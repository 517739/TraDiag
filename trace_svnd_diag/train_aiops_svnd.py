# train_aiops_svnd.py  (v0.3-nodectx)
# -*- coding: utf-8 -*-
"""
三头训练：det(二分类)/c3(三分类)/type(细类)
- 训练前在 train.jsonl 上拟合 stats（z-score）
- 自动筛掉训练支持数 < --type_min_support 的细类
- 训练结束：在测试集评测，并保存 run_summary.json + 完整 ckpt(.pt)
"""
import os, argparse, torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import (
    set_seed, TraceDataset, collate_multi, vocab_sizes_from_meta,
    derive_keep_types, evaluate_detailed, print_per_class_reports,
    collect_per_class_reports, save_run_summary, save_ckpt
)
from model import TraceClassifier

def main():
    ap = argparse.ArgumentParser("AIOps Multi-Head Training")
    ap.add_argument("--data-root", default="dataset/aiops_svnd")
    ap.add_argument("--save_dir",  default="dataset/aiops_svnd/1019", help="摘要与图片输出目录")
    ap.add_argument("--save_pt",   default="dataset/aiops_svnd/1019/aiops_nodectx_multihead.pt", help="ckpt 路径")
    ap.add_argument("--type_min_support", type=int, default=200, help="细类参与训练/评测的最小支持数")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-progress", action="store_true", default=False)
    ap.add_argument("--early_stop_patience", type=int, default=5)
    ap.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    args = ap.parse_args()

    set_seed(args.seed)
    tr = os.path.join(args.data_root, "train.jsonl")
    va = os.path.join(args.data_root, "val.jsonl")
    te = os.path.join(args.data_root, "test.jsonl")
    api_sz, st_sz, node_sz, type_names, ctx_dim = vocab_sizes_from_meta(args.data_root)

    # ====== 数据：拟合 stats + 派生 keep_types ======
    ds_fit = TraceDataset(tr, task="multihead", fit_stats=True)
    stats  = ds_fit.stats                       # (mu_dict, sd_dict)
    keep_types = derive_keep_types(ds_fit.items, args.type_min_support)
    print(f"[TypeFilter] keep_types={sorted(list(keep_types))}  (min_support={args.type_min_support})")

    # 三份数据集都用同一套 stats & keep_types（保证一致）
    ds_tr = TraceDataset(tr, task="multihead", fit_stats=False, stats=stats, keep_types=keep_types)
    ds_va = TraceDataset(va, task="multihead", fit_stats=False, stats=stats, keep_types=keep_types)
    ds_te = TraceDataset(te, task="multihead", fit_stats=False, stats=stats, keep_types=keep_types)
    mk = lambda ds, shuf: DataLoader(ds, batch_size=args.batch, shuffle=shuf, collate_fn=collate_multi, num_workers=0)
    tr_loader, va_loader, te_loader = mk(ds_tr, True), mk(ds_va, False), mk(ds_te, False)

    # ====== 模型/优化器 ======
    device=torch.device(args.device)
    model = TraceClassifier(api_sz, st_sz, node_sz, n_types=len(type_names), ctx_dim=ctx_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()
    ce  = nn.CrossEntropyLoss()

    show_progress = (not args.no_progress)

    def run_epoch(loader, train=True, show_progress: bool = True):
        model.train(train)
        tot = 0.0
        iterator = tqdm(loader, total=len(loader), dynamic_ncols=True, leave=False) if show_progress else loader
        for g, lab, *_ in iterator:
            g = g.to(device)
            yb = lab["y_bin"].float().to(device)
            yc = lab["y_c3"].to(device)
            yt = lab["y_type"].to(device)
            mt = lab["m_type"].to(device)

            out = model(g)  # dict: logits
            l1 = bce(out["logit_bin"], yb)
            l2 = ce(out["logits_c3"], yc)
            if mt.sum() > 0:
                # 只在有细类标签的样本上统计
                l3 = (ce(out["logits_type"], yt) * mt).sum() / (mt.sum() + 1e-6)
            else:
                l3 = l1 * 0.0
            loss = 0.1 * l1 + 0.2 * l2 + 0.7 * l3

            if train:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()

            tot += float(loss.item())
            if show_progress:
                iterator.set_postfix(loss=float(loss.item()))
        return tot / max(1, len(loader))

    best = float("inf")
    best_state = None
    no_improve = 0

    # ====== 训练 & 验证 ======
    for ep in range(1, args.epochs + 1):
        trL = run_epoch(tr_loader, train=True, show_progress=show_progress)
        vaL = run_epoch(va_loader, train=False, show_progress=show_progress)
        print(f"[Epoch {ep:02d}] train {trL:.4f} | val {vaL:.4f}")

        # 验证集总体指标（type 评测自动只在 m_type==1 样本上；并屏蔽未保留类）
        evaluate_detailed(model, va_loader, device, type_names, keep_types=keep_types)

        # 早停监控：以 val loss 为准
        if vaL < best - args.early_stop_min_delta:
            best = vaL
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
                print(f"[EarlyStop] 连续 {no_improve} 轮无显著改进，提前停止在 Epoch {ep:02d}。")
                break

    # ====== 测试集评测 & 保存 ======
    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = evaluate_detailed(model, te_loader, device, type_names, keep_types=keep_types)
    print_per_class_reports(model, te_loader, device, type_names, keep_types=keep_types)

    # 保存 run_summary.json
    os.makedirs(args.save_dir, exist_ok=True)
    from utils import save_run_summary, save_ckpt
    reports = collect_per_class_reports(model, te_loader, device, type_names, keep_types=keep_types)
    save_run_summary(save_dir=args.save_dir, args=vars(args), data_root=args.data_root,
                     metrics_dict=metrics, reports_dict=reports,
                     stats=stats, type_names=type_names, keep_types=keep_types)

    # 保存 ckpt：打包 state_dict + stats + args + type_names + keep_types
    save_ckpt(path=args.save_pt, model_state=model.state_dict(), stats=stats,
              args_dict=vars(args), type_names=type_names, keep_types=keep_types)
    print("[OK] saved", args.save_pt)

if __name__ == "__main__":
    main()
