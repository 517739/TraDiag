# train_aiops3c6c.py
# -*- coding: utf-8 -*-
import os, json, argparse
from collections import Counter
import numpy as np
import torch
import torch.nn as nn

from utils import *
from model import TraceClassifier

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--save-dir",  required=True, help="输出目录（将保存 ckpt 与评测报告）")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch",  type=int, default=64)
    ap.add_argument("--lr",     type=float, default=1e-3)
    ap.add_argument("--seed",   type=int, default=2025)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--min-type-support", type=int, default=200,
                    help="superfine 训练/评测保留的最小样本数")
    ap.add_argument("--use-class-weights", action="store_true",
                    help="在 coarse-3 与 superfine 使用按频次反比的类别权重")
    ap.add_argument("--ignore-loops", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    tr = os.path.join(args.data_root, "train.jsonl")
    va = os.path.join(args.data_root, "val.jsonl")
    te = os.path.join(args.data_root, "test.jsonl")

    # === 统计 latency 标准化（基于 train）
    fit = TraceDataset(tr, task="multi", fit_stats=True)
    stats = fit.stats
    ds_tr = TraceDataset(tr, task="multi", fit_stats=False, stats=stats)
    ds_va = TraceDataset(va, task="multi", fit_stats=False, stats=stats)
    ds_te = TraceDataset(te, task="multi", fit_stats=False, stats=stats)

    # 词表与类别名
    api_vocab, status_vocab, fine_names, superfine_names = vocab_sizes_from_meta(args.data_root)
    if superfine_names is None:
        raise RuntimeError("vocab.json 缺少 superfine_classes，请先用 make_aiops3c6c.py 生成。")
    class_names_type = superfine_names
    class_names_c3   = ["normal", "structural", "latency"]
    K_type = len(class_names_type)

    # === 统计 kept_types（按 train）
    cnt_type = Counter(int(r["superfine_label"]) for r in ds_tr.items
                       if r.get("superfine_label") is not None and r.get("superfine_label") >= 0)
    kept_types = {k for k, v in cnt_type.items() if v >= args.min_type_support}
    print(f"[Superfine] keep_types (>= {args.min_type_support}): {sorted(kept_types)}")

    # DataLoader
    mk = lambda ds, shuf: torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=shuf, collate_fn=collate, num_workers=0
    )
    tr_loader = mk(ds_tr, True); va_loader = mk(ds_va, False); te_loader = mk(ds_te, False)

    # 模型
    model = TraceClassifier(api_vocab, status_vocab, num_superfine=K_type,
                            ignore_loops=args.ignore_loops).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # 损失
    bce = nn.BCEWithLogitsLoss()
    if args.use_class_weights:
        cnt_c3 = Counter(int(r["coarse_label"]) for r in ds_tr.items if r.get("coarse_label") is not None)
        w_c3 = class_weights_from_counts(cnt_c3, 3).to(args.device)
        w_type = class_weights_from_counts(cnt_type, K_type).to(args.device)
        ce_c3   = nn.CrossEntropyLoss(weight=w_c3)
        ce_type = nn.CrossEntropyLoss(weight=w_type)
        print(f"[ClassWeights] c3={w_c3.tolist()} | type={w_type.tolist()}")
    else:
        ce_c3   = nn.CrossEntropyLoss()
        ce_type = nn.CrossEntropyLoss()

    def run_epoch(loader, train: bool):
        if train: model.train()
        else:     model.eval()
        totL = totA_bin = totA_c3 = totF_type = n = 0
        for g, lab, *_ in loader:
            g = g.to(args.device)
            out = model(g)

            # --- binary
            yb = lab["y_bin"].to(args.device).float().view(-1)
            loss_bin = bce(out["logit_bin"].view(-1), yb)

            # --- coarse-3
            yc = lab["y_c3"].to(args.device)
            loss_c3 = ce_c3(out["logits_c3"], yc)

            # --- superfine（仅有标签 & 且在 kept_types 的样本参与损失）
            yt = lab["y_type"].to(args.device)
            mt = lab["m_type"].to(args.device)
            keep_mask = mt.clone()
            if kept_types:
                kset = torch.tensor(sorted(kept_types), device=args.device, dtype=torch.long)
                # 标注在 kept 中？
                keep_mask = keep_mask & ( (yt.unsqueeze(1) == kset.unsqueeze(0)).any(dim=1) )
            loss_type = torch.tensor(0.0, device=args.device)
            if keep_mask.any():
                loss_type = ce_type(out["logits_type"][keep_mask], yt[keep_mask])

            # 平均权重
            loss = (loss_bin + loss_c3 + loss_type) / 3.0

            if train:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()

            b = yc.size(0)
            totL += loss.item() * b; n += b
            # 简单指标（可选）
            totA_bin += ((out["logit_bin"].view(-1) > 0).long().cpu() == yb.long().cpu()).float().sum().item()
            totA_c3  += (out["logits_c3"].argmax(1).cpu() == yc.cpu()).float().sum().item()
            # macro-F1（type）这里略（完全评测放到 evaluate 阶段）
        return totL / max(n,1), totA_bin / max(n,1), totA_c3 / max(n,1)

    # 训练
    best_val = 1e9; best_state = None
    os.makedirs(args.save_dir, exist_ok=True)
    for ep in range(1, args.epochs + 1):
        trL, trAb, trAc = run_epoch(tr_loader, True)
        vaL, vaAb, vaAc = run_epoch(va_loader, False)
        print(f"[Epoch {ep:02d}] Ltr={trL:.4f} | Lval={vaL:.4f} | "
              f"Acc(bin) tr/va={trAb:.4f}/{vaAb:.4f} | Acc(c3) tr/va={trAc:.4f}/{vaAc:.4f}")
        if vaL < best_val:
            best_val = vaL
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # --- 评测：coarse-3 全类混淆 ---
    _ = evaluate_detailed(model, va_loader, args.device, class_names_c3, head="c3",
                          save_csv_path=os.path.join(args.save_dir, "c3_val_confusion.csv"))
    _ = evaluate_detailed(model, te_loader, args.device, class_names_c3, head="c3",
                          save_csv_path=os.path.join(args.save_dir, "c3_test_confusion.csv"))

    # --- 评测与导出：superfine（筛类） ---
    evaluate_and_save_superfine(model, va_loader, args.device, class_names_type,
                                kept_types, os.path.join(args.save_dir, "val"), "val")
    evaluate_and_save_superfine(model, te_loader, args.device, class_names_type,
                                kept_types, os.path.join(args.save_dir, "test"), "test")

    # --- 保存 checkpoint（打包 stats/kept_types 等） ---
    ckpt = {
        "state_dict": model.state_dict(),
        "stats": stats,  # (mu_dict, sd_dict)
        "kept_types": sorted(kept_types),
        "class_names_superfine": class_names_type,
        "class_names_c3": class_names_c3,
        "epoch": args.epochs
    }
    ckpt_path = os.path.join(args.save_dir, "ckpt.pt")
    torch.save(ckpt, ckpt_path)
    print(f"[OK] saved checkpoint to {ckpt_path}")

if __name__ == "__main__":
    main()
