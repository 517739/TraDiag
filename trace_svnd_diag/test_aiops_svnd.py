# test_aiops_svnd.py  (v0.3)
# -*- coding: utf-8 -*-
"""
独立测试脚本：
- 从 ckpt 读取 stats 与 keep_types；
- 构造数据集（无需读取训练集），评测并打印逐类表（含 Pred 列）。
"""
import os, argparse, torch
from torch.utils.data import DataLoader

from utils import (
    set_seed, TraceDataset, collate_multi, vocab_sizes_from_meta,
    evaluate_detailed, print_per_class_reports
)
from model import TraceClassifier

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

def main():
    parser = argparse.ArgumentParser("AIOps Trace Multi-Head Test")
    parser.add_argument("--data_root", type=str, default="dataset/aiops_svnd",
                        help="包含 split.jsonl 和 vocab.json 的目录")
    parser.add_argument("--ckpt", type=str, default="dataset/aiops_svnd/1019/aiops_nodectx_multihead.pt",
                        help="*.pt 模型权重路径")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    # ====== 加载 ckpt：stats + keep_types + 可选 type_names ======
    ckpt = torch.load(args.ckpt, map_location=device)
    stats = ckpt["stats"]
    keep_types = set(ckpt.get("keep_types", [])) if ckpt.get("keep_types", None) is not None else None
    ckpt_args = ckpt.get("args", {})

    # ====== 词表/类型名 ======
    api_sz, st_sz, node_sz, type_names, ctx_dim = vocab_sizes_from_meta(args.data_root)

    # ====== 数据 ======
    jsonl_path = os.path.join(args.data_root, f"{args.split}.jsonl")
    ds = TraceDataset(jsonl_path, task="multihead", fit_stats=False, stats=stats, keep_types=keep_types)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_multi)

    # ====== 模型 ======
    model = TraceClassifier(api_sz, st_sz, node_sz, n_types=len(type_names), ctx_dim=ctx_dim).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # ====== 评测 ======
    print(f"\n[Eval] Running {args.split}.jsonl ...")
    metrics = evaluate_detailed(model, loader, device, type_names, keep_types=keep_types)
    print_per_class_reports(model, loader, device, type_names, keep_types=keep_types)

    print("\n[OK] 测试完成。")

if __name__ == "__main__":
    main()
