# utils.py
# -*- coding: utf-8 -*-
import os, re, json, random, struct, hashlib
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import dgl
from torch.utils.data import Dataset
import torch.nn as nn
import csv

# ===================== 1) 标签归并：coarse / fine（如需 6 类） =====================
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

# ===================== 4) 构图工具 =====================
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

# ===================== 5) Dataset & Collate（支持三头） =====================
class TraceDataset(Dataset):
    """
    task in {'multi','coarse','superfine'}：
      - 'multi'      ：联合训练（保留 normal & fault）
      - 'coarse'     ：仅三分类（保留 normal & fault）
      - 'superfine'  ：仅细类（故障），normal 丢弃
    返回：(g, lab, order, trace_id)，lab = {'y_bin','y_c3','y_type','m_type'}
    """
    def __init__(self, path: str, task="multi", fit_stats=False, stats=None):
        self.task = task
        self.items=[]
        with open(path,"r",encoding="utf-8") as f:
            for ln in f:
                r=json.loads(ln)
                if not r["nodes"] or len(r["nodes"])<2: continue
                if self.task=="superfine":
                    if r.get("superfine_label") is None or r.get("superfine_label",-1)<0: continue
                # 'multi' 与 'coarse' 都保留 normal
                self.items.append(r)
        if fit_stats:
            mu,sd=fit_latency_stats(self.items); self.stats=(mu,sd)
        else:
            self.stats=stats or ({}, {})

    def __len__(self): return len(self.items)

    def __getitem__(self, idx:int):
        r=self.items[idx]; n=len(r["nodes"])
        api=torch.tensor([int(nd["api_id"]) for nd in r["nodes"]],dtype=torch.long)
        st =torch.tensor([int(nd["status_id"]) for nd in r["nodes"]],dtype=torch.long)

        mu,sd=self.stats
        lat = torch.tensor([
            (float(nd["latency_ms"]) - mu.get(int(nd["api_id"]), 0.0)) / sd.get(int(nd["api_id"]), 1.0)
            for nd in r["nodes"]
        ], dtype=torch.float)[:,None]

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

        # 稳定 trace_id（优先 int，否则 md5->int64）
        tid_raw = r.get("trace_id", f"trace_{idx}")
        try: tid_num = int(tid_raw)
        except: tid_num = struct.unpack(">q", hashlib.md5(str(tid_raw).encode()).digest()[:8])[0]
        g.ndata["trace_id"] = torch.full((n,), tid_num, dtype=torch.long)

        # labels
        y_c3   = int(r["coarse_label"])
        y_bin  = 0 if y_c3 == 0 else 1
        y_type = int(r.get("superfine_label", -1))
        m_type = 1 if y_type >= 0 else 0

        lab = {
            "y_bin": torch.tensor(y_bin,  dtype=torch.long),
            "y_c3":  torch.tensor(y_c3,   dtype=torch.long),
            "y_type":torch.tensor(max(y_type,0), dtype=torch.long),  # 仅在 m_type==1 时有效
            "m_type":torch.tensor(m_type, dtype=torch.bool),
        }
        return g, lab, torch.tensor(order,dtype=torch.long), r.get("trace_id", f"trace_{idx}")

def collate(samples):
    gs, labs, orders, tids = zip(*samples)
    bg=dgl.batch(gs)
    # 合并 labels
    y_bin  = torch.stack([lb["y_bin"]  for lb in labs], 0)
    y_c3   = torch.stack([lb["y_c3"]   for lb in labs], 0)
    y_type = torch.stack([lb["y_type"] for lb in labs], 0)
    m_type = torch.stack([lb["m_type"] for lb in labs], 0)
    lab = {"y_bin":y_bin, "y_c3":y_c3, "y_type":y_type, "m_type":m_type}

    import numpy as _np
    offsets=_np.cumsum([0]+[g.num_nodes() for g in gs[:-1]]).tolist()
    flat=[]
    for off,ord_i in zip(offsets,orders):
        flat.extend([int(o)+off for o in ord_i.tolist()])
    return bg, lab, torch.tensor(flat,dtype=torch.long), list(tids)

# ===================== 6) Vocabulary helper =====================
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
    返回：(api_vocab_size, status_vocab_size, fine_names(6类或None), superfine_names)
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
        superfine_names = m["superfine_classes"]
    elif "superfine_label_map" in m:
        superfine_names=[nm for nm,_ in sorted(m["superfine_label_map"].items(), key=lambda x:x[1])]
    return api,status,fine_names,superfine_names

# ===================== 7) 评测 =====================
@torch.no_grad()
def evaluate_detailed(model, loader, device, class_names, head="c3", save_csv_path=None):
    """
    打印/导出：混淆矩阵 + 每类 TP/Support/P/R/F1 + overall Acc/Macro-F1
    head: 'c3' 使用 logits_c3；'bin' 使用 logit_bin（2类打印）；'type' 使用 logits_type（全类）
    """
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()

    model.eval()
    all_logits, all_labels = [], []
    total_loss, n = 0.0, 0

    for g, lab, *_ in loader:
        g = g.to(device)
        out = model(g)
        if head == "bin":
            logits = out["logit_bin"].view(-1)
            y = lab["y_bin"].to(device).float().view(-1)
            loss = bce(logits, y)
            # 转换为 2 类
            pred = (logits > 0).long().cpu()
            ycpu = y.long().cpu()
            all_logits.append(torch.stack([1-pred, pred], dim=1))
            all_labels.append(ycpu)
        elif head == "c3":
            logits = out["logits_c3"]
            y = lab["y_c3"].to(device)
            loss = ce(logits, y)
            all_logits.append(logits.detach().cpu())
            all_labels.append(y.detach().cpu())
        else:  # 'type'
            logits = out["logits_type"]
            y = lab["y_type"].to(device)
            loss = ce(logits, y)
            all_logits.append(logits.detach().cpu())
            all_labels.append(y.detach().cpu())

        b = y.size(0)
        total_loss += loss.item() * b
        n += b

    if n == 0:
        print("[evaluate_detailed] empty loader."); return

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    preds = logits.argmax(1)

    import numpy as _np
    C = logits.size(-1)
    cm = _np.zeros((C, C), dtype=int)
    for t, p in zip(labels.numpy().tolist(), preds.numpy().tolist()):
        cm[t, p] += 1

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

    if save_csv_path:
        os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
        with open(save_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            names = class_names if class_names else [str(i) for i in range(C)]
            w.writerow([""] + names)
            for i, nm in enumerate(names):
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
    import numpy as _np
    ce = nn.CrossEntropyLoss()
    model.eval()
    all_logits, all_labels = [], []
    total_loss, n = 0.0, 0

    for g, lab, *_ in loader:
        g = g.to(device); y = lab["y_type"].to(device)
        logits = model(g)["logits_type"]
        loss = ce(logits, y)
        b = y.size(0)
        total_loss += loss.item() * b; n += b
        all_logits.append(logits.detach().cpu()); all_labels.append(y.detach().cpu())

    if n == 0:
        print(f"[{split}] empty loader."); return

    logits = torch.cat(all_logits, 0); labels = torch.cat(all_labels, 0)
    K = logits.size(-1)

    if keep_types is not None:
        mask = torch.full((K,), float("-inf"))
        mask[list(sorted(keep_types))] = 0.0
        logits = logits + mask

    preds = logits.argmax(1)
    kept = sorted(keep_types) if keep_types is not None else list(range(K))
    kept_names = [class_names[i] for i in kept]
    idx = {k:i for i,k in enumerate(kept)}
    cm = _np.zeros((len(kept), len(kept)), dtype=int)
    for t, p in zip(labels.tolist(), preds.tolist()):
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1

    rows = []; mp=[]; mr=[]; mf=[]
    for i, k in enumerate(kept):
        tp = int(cm[i,i]); support = int(cm[i,:].sum()); predcnt = int(cm[:,i].sum())
        P = tp / (predcnt + 1e-9); R = tp / (support + 1e-9)
        F = 0.0 if (P+R)==0 else 2*P*R/(P+R)
        rows.append([kept_names[i], tp, support, predcnt, P, R, F])
        if support>0: mp.append(P); mr.append(R); mf.append(F)

    overall_acc = float((labels.numpy()==preds.numpy()).mean())
    macroP = float(_np.mean(mp)) if mp else 0.0
    macroR = float(_np.mean(mr)) if mr else 0.0
    macroF = float(_np.mean(mf)) if mf else 0.0

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"superfine_{split}_confusion.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow([""] + kept_names)
        for i, nm in enumerate(kept_names): w.writerow([nm] + cm[i,:].tolist())
    with open(os.path.join(out_dir, f"superfine_{split}_per_class.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(
            ["class","TP","Support","Pred","Precision","Recall","F1"]); w.writerows(rows)
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
    print(f"[Superfine-{split}] acc={overall_acc:.4f} macroF1={macroF:.4f} "
          f"kept={len(kept)}/{K} out_dir={out_dir}")
