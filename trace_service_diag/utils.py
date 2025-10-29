# utils.py
# -*- coding: utf-8 -*-
import os, re, json, random, hashlib
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import dgl
from torch.utils.data import Dataset
import csv
import torch.nn as nn
import hashlib, struct

# ===================== 1) 标签归并：coarse / fine =====================
STRUCTURAL_TYPES = {
    "code error", "pod failure", "pod kill", "node disk fill",
    "network corrupt", "network loss", "dns error", "target port misconfig",
    "jvm exception", "io fault",
}
LATENCY_TYPES = {
    "jvm latency", "network delay",
    "cpu stress", "memory stress", "node cpu stress", "node memory stress",
    "jvm gc", "jvm cpu",
}

FINE_GROUPS = {
    "S1_fail_call": {"code error", "pod failure", "pod kill", "dns error", "target port misconfig"},
    "S2_net_struct": {"network corrupt", "network loss"},
    "S3_other_struct": {"io fault", "node disk fill", "jvm exception"},
    "L1_net_delay": {"network delay"},
    "L2_jvm_perf": {"jvm latency", "jvm gc"},
    "L3_resource_stress": {"cpu stress", "memory stress", "node cpu stress", "node memory stress", "jvm cpu"},
}
FINE_LABELS = list(FINE_GROUPS.keys())
FINE_INDEX = {name: i for i, name in enumerate(FINE_LABELS)}

def map_coarse(ft: Optional[str]) -> Optional[int]:
    if not ft: return None
    k = ft.strip().lower()
    if k in STRUCTURAL_TYPES: return 1
    if k in LATENCY_TYPES:    return 2
    return None

def map_fine(ft: Optional[str]) -> Optional[int]:
    if not ft: return None
    k = ft.strip().lower()
    for name, s in FINE_GROUPS.items():
        if k in s: return FINE_INDEX[name]
    return None

# ===================== 2) URL 归一 & 词表键 =====================
_UUID = re.compile(r"[0-9a-fA-F]{8}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{12}")
_NUM  = re.compile(r"(?<![A-Za-z])[0-9]{2,}(?![A-Za-z])")
_HEX  = re.compile(r"0x[0-9a-fA-F]+")

def url_template(u: str) -> str:
    if not isinstance(u, str): return "NA"
    core = u.split("?")[0].split("#")[0]
    core = _UUID.sub("{uuid}", core)
    core = _HEX.sub("{hex}", core)
    core = _NUM.sub("{num}", core)
    core = re.sub(r"/{2,}", "/", core)
    return core

def make_api_key(service: str, url_tmpl: str) -> str:
    s = str(service) if service is not None else "NA_SVC"
    t = str(url_tmpl) if url_tmpl is not None else "NA_URL"
    return f"{s}||{t}"

# ===================== 3) 延迟标准化 =====================
def fit_latency_stats(items: List[dict]) -> Tuple[Dict[int, float], Dict[int, float]]:
    api_vals = defaultdict(list)
    for r in items:
        for nd in r["nodes"]:
            api_vals[int(nd["api_id"])].append(float(nd["latency_ms"]))
    mu, sd = {}, {}
    for k, vals in api_vals.items():
        v = np.asarray(vals, np.float32)
        p99 = np.percentile(v, 99)
        v = v[v < p99] if np.any(v < p99) else v
        mu[k] = float(np.mean(v))
        sd[k] = max(float(np.std(v)), 1e-3)
    return mu, sd

def z_latency(api_id: int, lat_ms: float, mu: Dict[int,float], sd: Dict[int,float]) -> float:
    return (lat_ms - mu.get(api_id, 0.0)) / sd.get(api_id, 1.0)

# ===================== 4) 构图：GCN / TreeLSTM（DAG） =====================
def build_parent_from_edges(edges: List[List[int]], n: int) -> List[int]:
    parent = [-1]*n
    for p, c in edges:
        if 0 <= p < n and 0 <= c < n:
            parent[c] = p
    return parent

def enforce_dag_parent(parent: List[int]) -> List[int]:
    parent = list(parent)
    n = len(parent)
    state = [0]*n  # 0=unseen,1=visiting,2=done
    def dfs(u):
        state[u] = 1
        p = parent[u]
        if p >= 0:
            if state[p] == 0:
                dfs(p)
            elif state[p] == 1:
                parent[u] = -1
        state[u] = 2
    for u in range(n):
        if state[u] == 0: dfs(u)
    return parent

def make_gcn_graph(edges: List[List[int]], n: int) -> dgl.DGLGraph:
    if edges:
        src = torch.tensor([e[0] for e in edges], dtype=torch.long)
        dst = torch.tensor([e[1] for e in edges], dtype=torch.long)
        g = dgl.graph((src, dst), num_nodes=n)
    else:
        g = dgl.graph(([], []), num_nodes=n)
    g = dgl.to_bidirected(g, copy_ndata=False)
    g = dgl.add_self_loop(g)
    return g

def make_treelstm_graph_from_parent(parent: List[int], n: int, device) -> dgl.DGLGraph:
    parent = enforce_dag_parent(list(parent))
    mask = torch.tensor([p >= 0 for p in parent], dtype=torch.bool)
    if mask.any():
        src = torch.tensor([parent[i] for i, m in enumerate(mask.tolist()) if m], dtype=torch.long, device=device)
        dst = torch.tensor([i for i, m in enumerate(mask.tolist()) if m], dtype=torch.long, device=device)
        g = dgl.graph((src, dst), num_nodes=n, device=device)
    else:
        g = dgl.graph(([], []), num_nodes=n, device=device)
    return g

# ===================== 5) 调试：parent 环检测 & 落盘 =====================
def has_cycle_by_parent(parent: torch.Tensor) -> bool:
    parent = parent.tolist()
    n = len(parent)
    state = [0]*n  # 0=unseen,1=visiting,2=done
    def dfs(u):
        state[u] = 1
        p = parent[u]
        if p >= 0:
            if state[p] == 0:
                if dfs(p): return True
            elif state[p] == 1:
                return True
        state[u] = 2
        return False
    for u in range(n):
        if state[u] == 0:
            if dfs(u): return True
    return False

def dump_bad_trace(jsonl_path: str, trace_id, num_nodes: int, has_parent_cycle: Optional[bool]):
    try:
        os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "trace_id": trace_id,
                "num_nodes": int(num_nodes),
                "has_parent_cycle": bool(has_parent_cycle) if has_parent_cycle is not None else None
            }, ensure_ascii=False) + "\n")
    except Exception:
        pass

# ===================== 6) DataSet & Collate =====================
class TraceDataset(Dataset):
    """支持 task in {'coarse','fine','superfine'}；fine/superfine 任务会丢弃 normal（label=-1 或缺失）。"""
    def __init__(self, path: str, task="coarse", fit_stats=False, stats=None):
        self.task = task
        self.items=[]
        with open(path,"r",encoding="utf-8") as f:
            for ln in f:
                r=json.loads(ln)
                if not r["nodes"] or len(r["nodes"])<2: continue
                if self.task=="fine":
                    if r.get("fine_label") is None or r.get("fine_label",-1)<0: continue
                elif self.task=="superfine":
                    if r.get("superfine_label") is None or r.get("superfine_label",-1)<0: continue
                self.items.append(r)
        if fit_stats:
            mu,sd=fit_latency_stats(self.items); self.stats=(mu,sd)
        else:
            self.stats=stats

    def __len__(self): return len(self.items)

    def __getitem__(self, idx:int):
        r=self.items[idx]; n=len(r["nodes"])
        api=torch.tensor([int(nd["api_id"]) for nd in r["nodes"]],dtype=torch.long)
        st =torch.tensor([int(nd["status_id"]) for nd in r["nodes"]],dtype=torch.long)
        mu,sd=self.stats
        lat=[]
        for nd in r["nodes"]:
            a=int(nd["api_id"]); l=(float(nd["latency_ms"])-mu.get(a,0.0))/sd.get(a,1.0)
            lat.append(l)
        lat=torch.tensor(lat,dtype=torch.float)[:,None]
        # parent/depth/order
        parent=[-1]*n
        if r["edges"]:
            for p,c in r["edges"]:
                parent[c]=p
        depth=[0]*n
        order=r.get("dfs_order", list(range(n)))
        for u in order:
            p=parent[u]; depth[u]=0 if p<0 else (depth[p]+1)
        # graphs
        if r["edges"]:
            src=torch.tensor([e[0] for e in r["edges"]],dtype=torch.long)
            dst=torch.tensor([e[1] for e in r["edges"]],dtype=torch.long)
            g=dgl.graph((src,dst), num_nodes=n)
        else:
            g=dgl.graph(([],[]), num_nodes=n)
        g=dgl.to_bidirected(g, copy_ndata=False)
        g=dgl.add_self_loop(g)
        g.ndata["api_id"]=api
        g.ndata["status_id"]=st
        g.ndata["lat"]=lat
        g.ndata["depth"]=torch.tensor(depth,dtype=torch.long)
        g.ndata["pos"]=torch.arange(n,dtype=torch.long)
        g.ndata["parent"]=torch.tensor(parent,dtype=torch.long)
        # 稳定 trace_id（优先直接 int，否则 md5 前 8 字节）
        tid_raw = r.get("trace_id", f"trace_{idx}")
        try:
            tid_num = int(tid_raw)
        except Exception:
            # 否则做稳定映射：MD5 -> 8字节 -> 有符号 int64
            digest = hashlib.md5(str(tid_raw).encode("utf-8")).digest()[:8]
            tid_num = struct.unpack(">q", digest)[0]  # 保证落在 int64 范围

        g.ndata["trace_id"] = torch.full((n,), tid_num, dtype=torch.long)

        # label
        if self.task=="coarse":
            y=int(r["coarse_label"])
        elif self.task=="fine":
            y=int(r["fine_label"])
        else:
            y=int(r["superfine_label"])
        return g, torch.tensor(y,dtype=torch.long), torch.tensor(order,dtype=torch.long), r.get("trace_id", f"trace_{idx}")

def collate(samples):
    gs, ys, orders, tids = zip(*samples)
    bg=dgl.batch(gs); y=torch.stack(ys,0)
    import numpy as _np
    offsets=_np.cumsum([0]+[g.num_nodes() for g in gs[:-1]]).tolist()
    flat=[]
    for off,ord_i in zip(offsets,orders):
        flat.extend([int(o)+off for o in ord_i.tolist()])
    return bg, y, torch.tensor(flat,dtype=torch.long), list(tids)

# ===================== 7) 其他工具 =====================
def set_seed(seed=2025):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def class_weights_from_counts(counts: Dict[int,int], num_classes: int) -> torch.Tensor:
    total=sum(counts.values()) or 1
    inv=[total/max(counts.get(c,1),1) for c in range(num_classes)]
    mean=sum(inv)/len(inv)
    return torch.tensor([v/mean for v in inv], dtype=torch.float)

def vocab_sizes_from_meta(root: str):
    """
    返回：(api_vocab_size, status_vocab_size, fine_names, superfine_names)
    fine_names / superfine_names 均可能为 None
    """
    meta=os.path.join(root,"vocab.json")
    if not os.path.exists(meta): return 0,0,None, None
    with open(meta,"r",encoding="utf-8") as f: m=json.load(f)
    api=int(m.get("api_vocab_size",0)); status=int(m.get("status_vocab_size",0))
    fine_names=None
    if "fine_label_map" in m:
        fine_names=[nm for nm,_ in sorted(m["fine_label_map"].items(), key=lambda x:x[1])]
    superfine_names = None
    if "superfine_classes" in m:
        # 已按 index 顺序写入，直接使用
        superfine_names = m["superfine_classes"]
    elif "superfine_label_map" in m:
        superfine_names=[nm for nm,_ in sorted(m["superfine_label_map"].items(), key=lambda x:x[1])]
    return api,status,fine_names,superfine_names


@torch.no_grad()
def evaluate_detailed(model, loader, device, class_names, save_csv_path=None):
    """
    打印/导出：混淆矩阵 + 每类 TP/Support/Precision/Recall/F1 + overall Acc/Macro-F1
    """
    ce = nn.CrossEntropyLoss()
    model.eval()
    all_logits, all_labels = [], []
    total_loss, n = 0.0, 0

    for batch in loader:
        g, y = batch[0], batch[1]
        g = g.to(device); y = y.to(device)
        logits = model(g)
        loss = ce(logits, y)
        b = y.size(0)
        total_loss += loss.item() * b
        n += b
        all_logits.append(logits.detach().cpu())
        all_labels.append(y.detach().cpu())

    if n == 0:
        print("[evaluate_detailed] empty loader."); return

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    preds = logits.argmax(1)

    import numpy as _np
    C = len(class_names)
    cm = _np.zeros((C, C), dtype=int)
    for t, p in zip(labels.numpy().tolist(), preds.numpy().tolist()):
        cm[t, p] += 1

    # per-class metrics
    per_class = []
    for c in range(C):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        support = cm[c, :].sum()
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        per_class.append((tp, support, prec, rec, f1))

    acc = (preds.numpy() == labels.numpy()).mean()
    macroF1 = float(_np.mean([x[4] for x in per_class]))

    # pretty print
    print("\n===== Detailed Evaluation =====")
    print("Confusion Matrix (rows=true, cols=pred):")
    print("         " + "".join([f"{n:^12}" for n in class_names]))
    for i, nm in enumerate(class_names):
        print(" "+f"{nm:<8}" + "".join([f"{cm[i,j]:^12d}" for j in range(C)]))

    print("\nPer-class metrics:")
    print(" class        TP     Support   Precision   Recall      F1")
    for i, nm in enumerate(class_names):
        tp, sup, pre, rec, f1 = per_class[i]
        print(f" {nm:<11}{tp:>6d}   {sup:>8d}   {pre:>9.4f}  {rec:>8.4f}  {f1:>8.4f}")
    print(f"\nOverall:  Acc={acc:.4f} | Macro-F1={macroF1:.4f}\n")

    # optional csv
    if save_csv_path:
        os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
        with open(save_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow([""] + class_names)
            for i, nm in enumerate(class_names):
                w.writerow([nm] + cm[i, :].tolist())

    return {
        "loss": total_loss / max(n, 1),
        "acc": acc,
        "macro_f1": macroF1,
        "confusion": cm,
        "per_class": per_class,
    }

@torch.no_grad()
def evaluate_and_save_superfine(model, loader, device, class_names, keep_types, out_dir, split):
    import csv, json, numpy as _np
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss, n = 0.0, 0
    all_logits, all_labels = [], []

    for g, y, *_ in loader:
        g = g.to(device); y = y.to(device)
        logits = model(g)
        loss = ce(logits, y); b = y.size(0)
        total_loss += loss.item() * b; n += b
        all_logits.append(logits.detach().cpu()); all_labels.append(y.detach().cpu())

    if n == 0:
        print(f"[{split}] empty loader."); return

    logits = torch.cat(all_logits, 0); labels = torch.cat(all_labels, 0)
    K = logits.size(-1)

    # 只在 keep_types 上评测 & 预测屏蔽
    if keep_types is not None:
        mask = torch.full((K,), float("-inf"))
        mask[list(sorted(keep_types))] = 0.0
        logits = logits + mask

    preds = logits.argmax(1)
    kept = sorted(keep_types) if keep_types is not None else list(range(K))
    kept_names = [class_names[i] for i in kept]

    # 混淆矩阵（仅 kept×kept）
    idx = {k:i for i,k in enumerate(kept)}
    cm = _np.zeros((len(kept), len(kept)), dtype=int)
    for t, p in zip(labels.tolist(), preds.tolist()):
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1

    # 逐类指标 & 宏平均（仅 support>0）
    rows = []; mp=[]; mr=[]; mf=[]
    for i, k in enumerate(kept):
        tp = int(cm[i,i])
        support = int(cm[i,:].sum())
        predcnt = int(cm[:,i].sum())
        P = tp / (predcnt + 1e-9)
        R = tp / (support + 1e-9)
        F = 0.0 if (P+R)==0 else 2*P*R/(P+R)
        rows.append([kept_names[i], tp, support, predcnt, P, R, F])
        if support > 0: mp.append(P); mr.append(R); mf.append(F)

    overall_acc = float((labels.numpy()==preds.numpy()).mean())
    macroP = float(_np.mean(mp)) if mp else 0.0
    macroR = float(_np.mean(mr)) if mr else 0.0
    macroF = float(_np.mean(mf)) if mf else 0.0

    os.makedirs(out_dir, exist_ok=True)
    # 保存混淆矩阵
    with open(os.path.join(out_dir, f"superfine_{split}_confusion.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow([""] + kept_names)
        for i, nm in enumerate(kept_names): w.writerow([nm] + cm[i,:].tolist())
    # 保存逐类
    with open(os.path.join(out_dir, f"superfine_{split}_per_class.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["class","TP","Support","Pred","Precision","Recall","F1"]); w.writerows(rows)
    # 保存摘要
    with open(os.path.join(out_dir, f"superfine_{split}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "overall_acc": overall_acc,
            "macro_precision": macroP,
            "macro_recall": macroR,
            "macro_f1": macroF,
            "kept_types": kept,
            "kept_names": kept_names,
            "loss": total_loss / max(n,1)
        }, f, ensure_ascii=False, indent=2)
    print(f"[Superfine-{split}] acc={overall_acc:.4f} macroF1={macroF:.4f} kept={len(kept)}/{K} out_dir={out_dir}")

