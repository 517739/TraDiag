# train_aiops3c6c.py
# -*- coding: utf-8 -*-
import os, json, argparse
from collections import Counter
import numpy as np
import torch
import torch.nn as nn

from utils import *
from model import TraceClassifier, TreeLSTMReadout

def accuracy(logits, y):
    return (logits.argmax(1) == y).float().mean().item()


def macro_f1(logits, y, C):
    pred = logits.argmax(1).cpu().numpy()
    true = y.cpu().numpy()
    f1s = []
    for c in range(C):
        tp = np.sum((pred == c) & (true == c))
        fp = np.sum((pred == c) & (true != c))
        fn = np.sum((pred != c) & (true == c))
        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        f1s.append(2 * p * r / (p + r + 1e-9))
    return float(np.mean(f1s))


@torch.no_grad()
def eval_epoch(model, loader, device, criterion, C):
    model.eval()
    totL = totA = totF = n = 0
    for g, y, order, _ in loader:
        g = g.to(device); y = y.to(device)
        logits = model(g)
        loss = criterion(logits, y)
        b = y.size(0)
        totL += loss.item() * b
        totA += accuracy(logits, y) * b
        totF += macro_f1(logits, y, C) * b
        n += b
    return totL / n, totA / n, totF / n


def train_epoch(model, loader, opt, device, criterion, C):
    model.train()
    totL = totA = totF = n = 0
    for g, y, order, _ in loader:
        g = g.to(device); y = y.to(device)
        logits = model(g)
        loss = criterion(logits, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        b = y.size(0)
        totL += loss.item() * b
        totA += accuracy(logits, y) * b
        totF += macro_f1(logits, y, C) * b
        n += b
    return totL / n, totA / n, totF / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="../../dataset/aiops3c6csp")
    ap.add_argument("--save_dir", default="../../dataset/aiops3c6csp/1020", help="摘要与图片输出目录")
    ap.add_argument("--task", choices=["coarse", "fine", "superfine"], default="superfine")
    ap.add_argument("--min-type-support", type=int, default=200,
                    help="细类（fine）最小样本数，低于此阈值的类别不参与训练期与评测期的细类报告")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--use-class-weights", action="store_true")
    # ----- 调试 & 容错 -----
    ap.add_argument("--ignore-loops", action="store_true",
                    help="TreeLSTM loop 时以零向量兜底继续训练（否则抛错）")
    ap.add_argument("--debug-dump", default="logs/bad_traces.jsonl",
                    help="把问题 trace 的统计写入 JSONL 文件")
    args = ap.parse_args()

    set_seed(args.seed)
    tr = os.path.join(args.data_root, "train.jsonl")
    va = os.path.join(args.data_root, "val.jsonl")
    te = os.path.join(args.data_root, "test.jsonl")

    # 统计基于 train 的延迟标准化
    fit = TraceDataset(tr, task=args.task, fit_stats=True)
    stats = fit.stats
    ds_tr = TraceDataset(tr, task=args.task, fit_stats=False, stats=stats)
    ds_va = TraceDataset(va, task=args.task, fit_stats=False, stats=stats)
    ds_te = TraceDataset(te, task=args.task, fit_stats=False, stats=stats)

    # 类别分布（打印）
    key = {"coarse":"coarse_label","fine":"fine_label","superfine":"superfine_label"}[args.task]
    def dist(ds):
        return Counter(int(r[key]) for r in ds.items if r.get(key) is not None and r.get(key) >= 0)
    print("[Label Dist] Train:", dict(dist(ds_tr)))
    print("[Label Dist] Val  :", dict(dist(ds_va)))
    print("[Label Dist] Test :", dict(dist(ds_te)))

    mk = lambda ds, shuf: torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=shuf, collate_fn=collate, num_workers=0
    )
    tr_loader = mk(ds_tr, True); va_loader = mk(ds_va, False); te_loader = mk(ds_te, False)

    # 词表与类别名（从 vocab.json 读取）
    api_vocab, status_vocab, fine_names, superfine_names = vocab_sizes_from_meta(args.data_root)
    if args.task == "coarse":
        class_names = ["normal", "structural", "latency"]
    elif args.task == "fine":
        class_names = fine_names or ["S1_fail_call","S2_net_struct","S3_other_struct","L1_net_delay","L2_jvm_perf","L3_resource_stress"]
    else:
        if superfine_names is None:
            raise RuntimeError("superfine_names not found in vocab.json；请先用含 superfine 的脚本重建数据集")
        class_names = superfine_names
    C = len(class_names)

    # === 仅在 superfine 时：按训练集统计 keep_types ===
    keep_types = None
    if args.task == "superfine":
        key = "superfine_label"
        cnt = Counter(int(r[key]) for r in ds_tr.items if r.get(key) is not None and r.get(key) >= 0)
        keep_types = {k for k, v in cnt.items() if v >= args.min_superfine_support}
        print(f"[Superfine] min_support={args.min_type_support}, keep_types={sorted(keep_types)}")

    os.makedirs(args.save_dir, exist_ok=True)
    if args.task == "superfine":
        with open(os.path.join(args.save_dir, "kept_types.json"), "w", encoding="utf-8") as f:
            json.dump({
                "kept_types": sorted(keep_types),
                "kept_names": [class_names[i] for i in sorted(keep_types)],
                "min_support": args.min_type_support
            }, f, ensure_ascii=False, indent=2)

    # 模型
    model = TraceClassifier(api_vocab, status_vocab, num_classes=C,
                            ignore_loops=args.ignore_loops,
                            debug_dump_path=args.debug_dump).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # 类别权重（可选）
    weights = torch.ones(C, dtype=torch.float)
    if args.use_class_weights:
        cnt = dist(ds_tr); weights = class_weights_from_counts(cnt, C)
        print(f"[Class Weights] {weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=weights.to(args.device))

    best_val = 1e9; best_state = None
    for ep in range(1, args.epochs + 1):
        trL, trA, trF = train_epoch(model, tr_loader, opt, args.device, criterion, C)
        vaL, vaA, vaF = eval_epoch(model, va_loader, args.device, criterion, C)
        print(f"[Epoch {ep:02d}] train loss {trL:.4f} acc {trA:.4f} f1 {trF:.4f} | val loss {vaL:.4f} acc {vaA:.4f} f1 {vaF:.4f}")
        if vaL < best_val:
            best_val = vaL; best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)

    # 详细评估与导出
    evaluate_detailed(model, va_loader, args.device, class_names,
                      save_csv_path=os.path.join(args.save_dir, f"{args.task}_val_confusion.csv"))
    evaluate_detailed(model, te_loader, args.device, class_names,
                      save_csv_path=os.path.join(args.save_dir, f"{args.task}_test_confusion.csv"))

    # 仅在 superfine 做筛类后的逐类报告
    if args.task == "superfine":
        evaluate_and_save_superfine(model, va_loader, args.device, class_names, keep_types,
                                    os.path.join(args.save_dir, "val"), "val")
        evaluate_and_save_superfine(model, te_loader, args.device, class_names, keep_types,
                                    os.path.join(args.save_dir, "test"), "test")

    out = os.path.join(args.save_dir, f"aiops_{args.task}_cls.pth")
    torch.save(model.state_dict(), out)
    print(f"[OK] saved model to {out}")
    print(f"[Stat] TreeLSTM loops ignored = {TreeLSTMReadout.get_loop_count()}")

if __name__ == "__main__":
    main()
