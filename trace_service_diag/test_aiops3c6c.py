# test_aiops3c6c.py
# -*- coding: utf-8 -*-
import os, json, argparse, csv
import torch
import torch.nn as nn

from utils import set_seed, TraceDataset, collate, vocab_sizes_from_meta
from model import TraceClassifier

@torch.no_grad()
def evaluate_and_save_fine(model, loader, device, class_names, keep_types, out_dir):
    ce = nn.CrossEntropyLoss()
    model.eval()
    all_logits, all_labels = [], []
    total_loss, n = 0.0, 0

    for g, y, *_ in loader:
        g = g.to(device); y = y.to(device)
        logits = model(g)
        loss = ce(logits, y)
        b = y.size(0)
        total_loss += loss.item() * b
        n += b
        all_logits.append(logits.detach().cpu())
        all_labels.append(y.detach().cpu())

    if n == 0:
        print("[test] empty loader."); return

    logits = torch.cat(all_logits, 0)
    labels = torch.cat(all_labels, 0)

    K = logits.size(-1)
    if keep_types is not None:
        mask = torch.full((K,), float("-inf"))
        mask[list(sorted(keep_types))] = 0.0
        logits = logits + mask

    preds = logits.argmax(1)

    keep = sorted(keep_types) if keep_types is not None else list(range(K))
    name_keep = [class_names[i] for i in keep]

    import numpy as _np
    cm = _np.zeros((len(keep), len(keep)), dtype=int)
    for t, p in zip(labels.numpy().tolist(), preds.numpy().tolist()):
        if t in keep and p in keep:
            cm[keep.index(t), keep.index(p)] += 1

    rows = []
    macro_p = []; macro_r = []; macro_f = []
    for i, k in enumerate(keep):
        tp = int(cm[i, i])
        support = int(cm[i, :].sum())
        predcnt = int(cm[:, i].sum())
        p = tp / (predcnt + 1e-9)
        r = tp / (support + 1e-9)
        f = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
        rows.append([name_keep[i], tp, support, predcnt, p, r, f])
        if support > 0:
            macro_p.append(p); macro_r.append(r); macro_f.append(f)

    overall_acc = float((labels.numpy() == preds.numpy()).mean())
    macroP = float(_np.mean(macro_p)) if macro_p else 0.0
    macroR = float(_np.mean(macro_r)) if macro_r else 0.0
    macroF = float(_np.mean(macro_f)) if macro_f else 0.0

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "fine_test_confusion.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow([""] + name_keep)
        for i, nm in enumerate(name_keep):
            w.writerow([nm] + cm[i, :].tolist())

    with open(os.path.join(out_dir, "fine_test_per_class.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class","TP","Support","Pred","Precision","Recall","F1"])
        w.writerows(rows)

    with open(os.path.join(out_dir, "fine_test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "overall_acc": overall_acc,
            "macro_precision": macroP,
            "macro_recall": macroR,
            "macro_f1": macroF,
            "kept_types": keep,
            "kept_names": name_keep,
            "loss": total_loss / max(n, 1)
        }, f, ensure_ascii=False, indent=2)

    print(f"[Fine-test] acc={overall_acc:.4f} macroF1={macroF:.4f} kept={len(keep)}/{K} out_dir={out_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="../../dataset/aiops3c6csp", help="包含 test.jsonl 与 vocab.json 的目录")
    ap.add_argument("--model-path", default="../../dataset/aiops3c6csp/aiops_superfine_cls.pth", help="训练好的权重文件 .pth")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--min-type-support", type=int, default=200)
    ap.add_argument("--run-name", default="trace_only")
    args = ap.parse_args()

    set_seed(args.seed)

    # 读取类别名
    api_vocab, status_vocab, fine_names, superfine_names = vocab_sizes_from_meta(args.data_root)
    if fine_names is None:
        raise RuntimeError("fine_names 为空：请确认 data-root 含 vocab.json 且带 fine_label_map/superfine_classes")

    # 数据集（只用 test）
    te_path = os.path.join(args.data_root, "test.jsonl")
    ds_te = TraceDataset(te_path, task="superfine")
    te_loader = torch.utils.data.DataLoader(ds_te, batch_size=args.batch, shuffle=False, collate_fn=collate, num_workers=0)

    # 计算 keep_types（按训练集统计通常更合理；若无训练集统计，这里用 test 近似）
    # 建议你传入一个 JSON（或从训练的 metrics.json 里读），这里给出 fallback：仅根据 test
    from collections import Counter
    cnt = Counter(int(r["fine_label"]) for r in ds_te.items if r.get("fine_label") is not None and int(r["fine_label"]) >= 0)
    keep_types = {k for k, v in cnt.items() if v >= args.min_type_support}
    print(f"[Fine] (fallback) min_support={args.min_type_support}, keep_types={sorted(keep_types)}")

    # 模型
    num_classes = len(fine_names)
    model = TraceClassifier(api_vocab, status_vocab, num_classes).to(args.device)
    state = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(state, strict=True)

    # 输出目录
    out_dir = os.path.join(args.data_root, "runs", args.run_name, "fine_test")
    evaluate_and_save_fine(model, te_loader, args.device, fine_names, keep_types, out_dir)

if __name__ == "__main__":
    main()
