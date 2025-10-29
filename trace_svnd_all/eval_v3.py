# -*- coding: utf-8 -*-
"""
eval_v3.py — TraDiag v3 统一级联评测（A→B→C）

特性：
1) 阈值标定：优先在 A_val.jsonl 上标定 A 的阈值；无则回退到 val.jsonl。
2) 统一评测：优先使用 unified_test.jsonl；若不存在可选 --unify_source {unified,test,test+val}。
3) 仅在 “被 A 检出为异常” 的子集上评 B（故障细类）和 C（根因定位）。
4) 输出端到端结果到 result_unified.txt。

用法示例：
  python eval_v3.py \
    --data_root dataset/aiops_v3 \
    --ckpt_a runs/v3_try1/stageA_best.pt \
    --ckpt_b runs/v3_try1/stageB_best.pt \
    --device cuda:0 \
    --unify_source auto \
    --topk 0.2 --alpha_lat 2.0 --beta_stat 1.0 --recall_floor 0.95
"""
import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# 依赖你已经存在的 v3 代码
from utils_v3 import (
    JSONLDataset, collate_batch,
    evaluate_stage_a, evaluate_stage_b, evaluate_stage_c
)
from model_v3 import TraceUnifiedModelV3


def _load_ckpt(path, device):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device)
    return ckpt


class _SubsetByIndex(Dataset):
    """对任意 Dataset 做索引子集，保持与 JSONLDataset 兼容的 __getitem__ 返回。"""
    def __init__(self, base, indices):
        self.base = base
        self.idxs = list(indices)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        return self.base[self.idxs[i]]


def _pick_split_path(root: str, prefer: str, fallback: str):
    """选择优先 split，若不存在则回退。返回 split 名称（用于 JSONLDataset）。"""
    prefer_path = os.path.join(root, f"{prefer}.jsonl")
    if os.path.isfile(prefer_path):
        return prefer
    fb_path = os.path.join(root, f"{fallback}.jsonl")
    if os.path.isfile(fb_path):
        return fallback
    raise FileNotFoundError(f"neither {prefer}.jsonl nor {fallback}.jsonl exists under {root}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--ckpt_a", type=str, required=True)
    ap.add_argument("--ckpt_b", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--batch", type=int, default=4)

    # A 的评估/阈值参数（需与训练时保持一致）
    ap.add_argument("--topk", type=float, default=0.2)
    ap.add_argument("--alpha_lat", type=float, default=2.0)
    ap.add_argument("--beta_stat", type=float, default=1.0)
    ap.add_argument("--recall_floor", type=float, default=0.95)

    # 模型宽度（需与训练时一致）
    ap.add_argument("--emb", type=int, default=64)
    ap.add_argument("--gc_hidden", type=int, default=128)

    # 统一评测数据来源：auto=优先 unified_test，若无则 test；也可明确指定
    ap.add_argument("--unify_source", type=str, default="auto",
                    choices=["auto", "unified", "test", "test+val"])

    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---------------- 1) 装载 Stage A（用于检出 + 阈值） ----------------
    model = TraceUnifiedModelV3(emb=args.emb, gc_hidden=args.gc_hidden).to(dev)
    ckptA = _load_ckpt(args.ckpt_a, dev)
    if "model" in ckptA:
        model.load_state_dict(ckptA["model"], strict=False)
    mu, sd = ckptA.get("mu", {}), ckptA.get("sd", {})

    # 阈值标定 split：A_val 优先，否则用 val
    aval_split = _pick_split_path(args.data_root, "A_val", "val")
    ds_Aval = JSONLDataset(args.data_root, aval_split, cache_size=50000)
    loader_Aval = DataLoader(
        ds_Aval, batch_size=args.batch, shuffle=False,
        collate_fn=collate_batch, num_workers=0, pin_memory=False, persistent_workers=False
    )

    pr, roc, f1r, thr = evaluate_stage_a(
        model, loader_Aval, mu, sd, dev,
        topk=args.topk, alpha_lat=args.alpha_lat,
        beta_stat=args.beta_stat, recall_floor=args.recall_floor
    )
    print(
        "[Unified][A@VAL] ROC-AUC={:.4f}  PR-AUC={:.4f}  F1@R≥{:.0f}%={:.4f}  thr={:.6f}"
        .format(roc, pr, args.recall_floor * 100, f1r, thr)
    )

    # ---------------- 2) 统一评测的数据集选择 ----------------
    if args.unify_source == "auto":
        # 优先 unified_test.jsonl；若不存在则 test.jsonl
        try:
            unify_split = _pick_split_path(args.data_root, "unified_test", "test")
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"{e}\n提示：如果你希望端到端评测，请先用 make_aiops_v3.py 生成 unified_test.jsonl；"
                f"或者使用 --unify_source test/test+val。"
            )
        ds_U = JSONLDataset(args.data_root, unify_split, cache_size=50000)
    elif args.unify_source == "unified":
        unify_split = "unified_test"
        ds_U = JSONLDataset(args.data_root, unify_split, cache_size=50000)
    elif args.unify_source == "test":
        unify_split = "test"
        ds_U = JSONLDataset(args.data_root, "test", cache_size=50000)
    else:  # test+val
        # 将 test 与 val 级联为一个“统一池”
        ds_te = JSONLDataset(args.data_root, "test", cache_size=50000)
        ds_va = JSONLDataset(args.data_root, "val",  cache_size=50000)
        # 简单拼接：用一个临时 Dataset 做视图
        class _Concat(Dataset):
            def __init__(self, a, b): self.a, self.b = a, b
            def __len__(self): return len(self.a) + len(self.b)
            def __getitem__(self, i):
                return (self.a if i < len(self.a) else self.b)[i if i < len(self.a) else (i - len(self.a))]
        ds_U = _Concat(ds_te, ds_va)
        unify_split = "test+val"

    loader_U = DataLoader(
        ds_U, batch_size=args.batch, shuffle=False,
        collate_fn=collate_batch, num_workers=0, pin_memory=False, persistent_workers=False
    )

    # ---------------- 3) 在统一集上用 A 做故障检出（依据 thr） ----------------
    model.eval()
    y_true, scores = [], []
    with torch.no_grad():
        for g, y in loader_U:
            g = g.to(dev)
            out = model(g, vae_mode=True)
            # 与 utils_v3.evaluate_stage_a 同步的打分方式（标准化后重建误差）
            api = g.ndata["api"]
            lat = g.ndata["lat_ms"]
            api_np = api.cpu().numpy()
            mu_v = torch.tensor([mu.get(int(a), float(lat.mean())) for a in api_np], device=lat.device)
            sd_v = torch.tensor([sd.get(int(a), 1.0) for a in api_np], device=lat.device).clamp(min=1e-6)
            z = (lat - mu_v) / sd_v
            lat_hat_z = (out["lat_hat"] - mu_v) / sd_v
            e_lat = torch.abs(lat_hat_z - z).mean().item()
            e_stat = torch.abs(out["stat_hat"] - g.ndata["status"]).mean().item()
            s = args.alpha_lat * e_lat + args.beta_stat * e_stat
            y_true.append(int(y["y_bin"].item()))
            scores.append(float(s))

    scores = np.asarray(scores, dtype=np.float64)
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    pred_pos_mask = scores >= float(thr)
    pos_idx = np.nonzero(pred_pos_mask)[0].tolist()
    print(f"[Unified] A 预测异常 {pred_pos_mask.sum()} / {len(scores)} = {pred_pos_mask.mean():.4%}")

    # ---------------- 4) 在异常子集上评 Stage B（细类分类） ----------------
    ckptB = _load_ckpt(args.ckpt_b, dev)
    # 复用同一实例装 B 权重（你的 model_v3 是同一大模型，上层共享）
    if "model" in ckptB:
        model.load_state_dict(ckptB["model"], strict=False)

    ds_U_pos = _SubsetByIndex(ds_U, pos_idx)

    # 仅统计有细类标签的样本（y_type>=0），与 utils_v3 的评测对齐
    valid_pos = []
    for i in pos_idx:
        try:
            r = ds_U[i]  # JSONLDataset.__getitem__ 返回 (graph, label_dict)
            if isinstance(r, tuple) and len(r) == 2:
                y_type = int(r[1].get("y_type", -1))
            else:
                # 兼容可能的 (dict/json) 访问
                y_type = int(r.get("y_type", -1)) if isinstance(r, dict) else -1
        except Exception:
            y_type = -1
        if y_type >= 0:
            valid_pos.append(i)

    ds_U_pos_type = _SubsetByIndex(ds_U, valid_pos)
    loader_U_pos_type = DataLoader(
        ds_U_pos_type, batch_size=args.batch, shuffle=False,
        collate_fn=collate_batch, num_workers=0, pin_memory=False, persistent_workers=False
    )

    # 读取 type_names（若存在）
    try:
        with open(os.path.join(args.data_root, "vocab.json"), "r", encoding="utf-8") as f:
            type_names = json.load(f).get("type_names", [])
    except Exception:
        type_names = []

    accB, metrics = evaluate_stage_b(
        model, loader_U_pos_type, dev,
        return_details=True, type_names=type_names
    )
    print(f"[Unified][B@A-pos] type-acc={accB:.4f}  (样本={len(valid_pos)})")
    if metrics:
        print("[Unified][B per-type]")
        for m in metrics:
            print(
                "  - {name} (idx={index:02d}): support={support} predicted={predicted} "
                "precision={precision:.4f} recall={recall:.4f} F1={f1:.4f}".format(**m)
            )

    # ---------------- 5) 在异常子集上评 Stage C（根因定位） ----------------
    loader_U_pos = DataLoader(
        ds_U_pos, batch_size=1, shuffle=False,
        collate_fn=collate_batch, num_workers=0, pin_memory=False, persistent_workers=False
    )
    rca = evaluate_stage_c(
        model, loader_U_pos, mu, sd, dev,
        alpha_lat=args.alpha_lat, beta_stat=args.beta_stat
    )
    print("[Unified][C@A-pos] top1={top1:.4f} top3={top3:.4f} top5={top5:.4f} covered={covered:d}".format(**rca))

    # ---------------- 6) 汇总到文件 ----------------
    out_txt = os.path.join(args.data_root, "result_unified.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(
            "[Unified][A@VAL] ROC-AUC={:.4f} PR-AUC={:.4f} F1@R≥{:.0f}%={:.4f} thr={:.6f}\n".format(
                roc, pr, args.recall_floor * 100, f1r, thr
            )
        )
        f.write("[Unified][A] pos={}/{}\n".format(int(pred_pos_mask.sum()), len(scores)))
        f.write("[Unified][B@A-pos] type-acc={:.4f} samples={}\n".format(accB, len(valid_pos)))
        for m in (metrics or []):
            f.write(
                "[B per-type] {name} idx={index} support={support} predicted={predicted} "
                "precision={precision:.4f} recall={recall:.4f} F1={f1:.4f}\n".format(**m)
            )
        f.write("[Unified][C@A-pos] top1={top1:.4f} top3={top3:.4f} top5={top5:.4f} covered={covered:d}\n".format(**rca))
    print(f"[Unified] done. → {out_txt}")


if __name__ == "__main__":
    main()
