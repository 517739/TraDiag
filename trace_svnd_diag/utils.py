# utils.py  (v0.3)
# -*- coding: utf-8 -*-
import os, json, hashlib, random
from typing import List, Dict, Tuple, Optional, Set
import numpy as np
import torch
import dgl
from torch.utils.data import Dataset

# ========== 统计延迟（供 z-score 标准化） ==========
def fit_latency_stats(items: List[dict]):
    """在训练集 items 上拟合每个 api_id 的 (mu, sigma)。"""
    api_vals = {}
    for r in items:
        for nd in r["nodes"]:
            api_vals.setdefault(int(nd["api_id"]), []).append(float(nd["latency_ms"]))
    mu, sd = {}, {}
    for k, v in api_vals.items():
        arr = np.asarray(v, np.float32)
        # 简单去极值：截去 >p99
        p99 = np.percentile(arr, 99)
        arr = arr[arr < p99] if np.any(arr < p99) else arr
        mu[k] = float(np.mean(arr))
        sd[k] = max(float(np.std(arr)), 1e-3)
    return mu, sd  # 注意：是 (mu_dict, sd_dict) 二元组

# ========== 细类支持数与保留集合 ==========
def type_support_from_items(items: List[dict]) -> Dict[int, int]:
    """统计 train items 中每个细类(y_type>=0)的支持数。"""
    cnt = {}
    for r in items:
        t = int(r.get("y_type", -1))
        if t >= 0:
            cnt[t] = cnt.get(t, 0) + 1
    return cnt

def derive_keep_types(items: List[dict], min_support: int = 200) -> Set[int]:
    """得到参与训练/评测的细类集合（支持数 >= min_support）。"""
    cnt = type_support_from_items(items)
    kept = {t for t, c in cnt.items() if c >= min_support}
    return kept

# ========== Dataset ==========
class TraceDataset(Dataset):
    def __init__(self, path: str, task="multihead", fit_stats=False, stats=None, keep_types: Optional[Set[int]]=None):
        self.task = task
        self.items=[]
        with open(path,"r",encoding="utf-8") as f:
            for ln in f:
                r=json.loads(ln)
                if not r.get("nodes"): continue
                if len(r["nodes"]) < 2: continue
                self.items.append(r)
        if fit_stats:
            self.stats = fit_latency_stats(self.items)
        else:
            self.stats = stats  # (mu_dict, sd_dict) 或 None
        self.keep_types = set(keep_types) if keep_types is not None else None

    def __len__(self): return len(self.items)

    def __getitem__(self, idx: int):
        r = self.items[idx]; n=len(r["nodes"])
        mu, sd = self.stats if self.stats is not None else ({},{})

        api  = torch.tensor([int(nd["api_id"])   for nd in r["nodes"]], dtype=torch.long)
        stat = torch.tensor([int(nd["status_id"])for nd in r["nodes"]], dtype=torch.long)
        node = torch.tensor([int(nd.get("node_id",0)) for nd in r["nodes"]], dtype=torch.long)
        # z-score latency
        lat  = []
        for nd in r["nodes"]:
            a=int(nd["api_id"]); l=(float(nd["latency_ms"])-mu.get(a,0.0))/sd.get(a,1.0)
            lat.append(l)
        lat  = torch.tensor(lat, dtype=torch.float).unsqueeze(-1)

        # parent/depth/pos
        parent=[-1]*n
        if r.get("edges"):
            for p,c in r["edges"]:
                parent[c]=p
        order = r.get("dfs_order", list(range(n)))
        depth=[0]*n
        for u in order:
            p=parent[u]; depth[u]=0 if p<0 else (depth[p]+1)

        # 调用图
        if r.get("edges"):
            src=torch.tensor([e[0] for e in r["edges"]], dtype=torch.long)
            dst=torch.tensor([e[1] for e in r["edges"]], dtype=torch.long)
            g=dgl.graph((src,dst), num_nodes=n)
        else:
            g=dgl.graph(([],[]), num_nodes=n)
        g=dgl.to_bidirected(g, copy_ndata=False)
        g=dgl.add_self_loop(g)

        # ndata
        g.ndata["api_id"]=api; g.ndata["status_id"]=stat; g.ndata["node_id"]=node
        g.ndata["lat"]=lat
        g.ndata["depth"]=torch.tensor(depth, dtype=torch.long)
        g.ndata["pos"]=torch.arange(n, dtype=torch.long)
        g.ndata["parent"]=torch.tensor(parent, dtype=torch.long)
        # ctx（图级 → 复制到每节点；readout=mean 等价于原向量）
        ctx = r.get("ctx", None)
        if isinstance(ctx, list) and len(ctx)>0:
            g.ndata["ctx"] = torch.tensor(ctx, dtype=torch.float).repeat(n,1)

        # labels
        y_bin  = int(r.get("y_bin", 0))
        y_c3   = int(r.get("y_c3",  0))
        y_type = int(r.get("y_type",-1))
        keep_mask = (y_type >= 0) and (self.keep_types is None or y_type in self.keep_types)
        labels = {
            "y_bin":  torch.tensor(y_bin, dtype=torch.long),
            "y_c3":   torch.tensor(y_c3,  dtype=torch.long),
            "y_type": torch.tensor(max(y_type,0), dtype=torch.long),
            "m_type": torch.tensor(1 if keep_mask else 0, dtype=torch.float),
        }
        tid = r.get("trace_id", f"trace_{idx}")
        return g, labels, torch.tensor(order, dtype=torch.long), tid

def collate_multi(samples):
    gs, labs, orders, tids = zip(*samples)
    bg = dgl.batch(gs)
    y_bin  = torch.stack([l["y_bin"] for l in labs], 0)
    y_c3   = torch.stack([l["y_c3"]  for l in labs], 0)
    y_type = torch.stack([l["y_type"]for l in labs], 0)
    m_type = torch.stack([l["m_type"]for l in labs], 0).float()
    labels = {"y_bin":y_bin,"y_c3":y_c3,"y_type":y_type,"m_type":m_type}
    # 展平 DFS 序（可视化用）
    import numpy as _np
    offsets=_np.cumsum([0]+[g.num_nodes() for g in gs[:-1]]).tolist()
    flat=[]
    for off, ord_i in zip(offsets, orders):
        flat.extend([int(o)+off for o in ord_i.tolist()])
    return bg, labels, torch.tensor(flat, dtype=torch.long), list(tids)

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def vocab_sizes_from_meta(root: str):
    meta=os.path.join(root,"vocab.json")
    with open(meta,"r",encoding="utf-8") as f:
        m=json.load(f)
    return int(m["api_vocab_size"]), int(m["status_vocab_size"]), int(m["node_vocab_size"]), m.get("type_names",[]), int(m.get("ctx_dim",0))

# ========== 指标（含 keep_types 掩码） ==========
@torch.no_grad()
def evaluate_detailed(model, loader, device, type_names, keep_types: Optional[Set[int]]=None):
    """打印 det/c3/type 总体指标。type 在 m_type==1 的样本上评估；若 keep_types 给定，会屏蔽未保留类的预测。"""
    model.eval()
    all_bin_y=[]; all_bin_p=[]
    all_c3_y =[]; all_c3_p=[]
    all_ty_y=[]; all_ty_p=[]
    for g, lab, *_ in loader:
        g=g.to(device)
        out=model(g)
        # bin
        all_bin_y.append(lab["y_bin"].numpy())
        all_bin_p.append(torch.sigmoid(out["logit_bin"]).cpu().numpy())
        # c3
        all_c3_y.append(lab["y_c3"].numpy())
        all_c3_p.append(out["logits_c3"].argmax(1).cpu().numpy())
        # type（mask）
        m=lab["m_type"].numpy().astype(bool)
        if m.any():
            ty_y = lab["y_type"].numpy()[m]
            logits = out["logits_type"]
            if keep_types is not None:
                K = logits.size(-1)
                mask = torch.full((K,), float('-inf'), device=logits.device)
                mask[list(keep_types)] = 0.0
                logits = logits + mask
            ty_p = logits.argmax(1).cpu().numpy()[m]
            all_ty_y.append(ty_y); all_ty_p.append(ty_p)

    import numpy as _np
    def bin_metrics(y,p,thr=0.5):
        y=_np.concatenate(y); p=_np.concatenate(p); pr=(p>=thr).astype(int)
        acc=float((pr==y).mean())
        tp=int(((pr==1)&(y==1)).sum()); fp=int(((pr==1)&(y==0)).sum()); fn=int(((pr==0)&(y==1)).sum())
        prec=tp/(tp+fp) if tp+fp>0 else 0.0; rec=tp/(tp+fn) if tp+fn>0 else 0.0
        f1=2*prec*rec/(prec+rec) if prec+rec>0 else 0.0
        return acc,f1
    def multi_metrics(y,p,C):
        y=_np.concatenate(y); p=_np.concatenate(p)
        acc=float((y==p).mean()); f1s=[]
        for c in range(C):
            yt=(y==c); yp=(p==c)
            tp=int((yt&yp).sum()); fp=int((~yt&yp).sum()); fn=int((yt&~yp).sum())
            pr=tp/(tp+fp) if tp+fp>0 else 0.0; rc=tp/(tp+fn) if tp+fn>0 else 0.0
            f1=2*pr*rc/(pr+rc) if pr+rc>0 else 0.0; f1s.append(f1)
        return acc, float(_np.mean(f1s))
    det_acc, det_f1 = bin_metrics(all_bin_y, all_bin_p)
    c3_acc,  c3_f1  = multi_metrics(all_c3_y, all_c3_p, 3)
    if all_ty_y:
        ty_acc, ty_f1 = multi_metrics(all_ty_y, all_ty_p, len(type_names) if type_names else 1)
    else:
        ty_acc, ty_f1 = 0.0, 0.0
    print(f"[Eval] det_acc={det_acc:.3f} det_f1={det_f1:.3f} | c3_acc={c3_acc:.3f} c3_f1={c3_f1:.3f} | type_acc={ty_acc:.3f} type_f1={ty_f1:.3f}")
    return dict(det_acc=det_acc, det_f1=det_f1, c3_acc=c3_acc, c3_f1=c3_f1, type_acc=ty_acc, type_f1=ty_f1)

@torch.no_grad()
def print_per_class_reports(model, loader, device, type_names,
                            out_dir=None, keep_types: Optional[Set[int]]=None):
    """
    打印三张逐类表：Binary / Coarse-3 / Fine-Type
    每类包含 TP / Support / Pred / Precision / Recall / F1
    Fine-Type 仅统计 m_type==1 的样本；若 keep_types 给定，还会在预测时屏蔽未保留的类。
    """
    model.eval()
    yb_list, pb_list = [], []
    yc_list, pc_list = [], []
    yt_list, pt_list, mt_list = [], [], []

    for g, lab, *_ in loader:
        g = g.to(device)
        out = model(g)

        # Binary
        yb = lab["y_bin"].to(device).view(-1).long()
        pb = (out["logit_bin"].view(-1) > 0).long()
        yb_list.append(yb.cpu()); pb_list.append(pb.cpu())

        # Coarse-3
        yc = lab["y_c3"].to(device).view(-1).long()
        pc = out["logits_c3"].argmax(dim=1).view(-1).long()
        yc_list.append(yc.cpu()); pc_list.append(pc.cpu())

        # Fine-Type
        mt = lab["m_type"].to(device).view(-1).bool()
        yt = lab["y_type"].to(device).view(-1).long()
        logits_t = out["logits_type"]
        if keep_types is not None:
            K = logits_t.size(-1)
            mask = torch.full((K,), float('-inf'), device=logits_t.device)
            mask[list(keep_types)] = 0.0
            logits_t = logits_t + mask
        pt = logits_t.argmax(dim=1).view(-1).long()
        yt_list.append(yt.cpu()); pt_list.append(pt.cpu()); mt_list.append(mt.cpu())

    def _per_class_table(y_true_list, y_pred_list, class_names,
                         valid_mask_list=None,
                         drop_rows_with_zero_support: bool = True,
                         macro_exclude_zero_support: bool = True):
        """返回 rows(list[dict])，以及 overall_acc、macroP、macroR、macroF1。"""
        y_true = torch.cat(y_true_list) if len(y_true_list) else torch.empty(0, dtype=torch.long)
        y_pred = torch.cat(y_pred_list) if len(y_pred_list) else torch.empty(0, dtype=torch.long)
        if valid_mask_list is not None:
            m = torch.cat(valid_mask_list).bool()
            y_true = y_true[m];
            y_pred = y_pred[m]

        K = len(class_names)
        tp = torch.zeros(K, dtype=torch.long)
        sup = torch.zeros(K, dtype=torch.long)  # ground-truth count
        pre = torch.zeros(K, dtype=torch.long)  # predicted count

        for k in range(K):
            mk_t = (y_true == k)
            mk_p = (y_pred == k)
            sup[k] = mk_t.sum()
            pre[k] = mk_p.sum()
            tp[k] = (mk_t & mk_p).sum()

        P = tp.float() / pre.clamp_min(1).float()
        R = tp.float() / sup.clamp_min(1).float()
        F = 2 * P * R / (P + R + 1e-12)

        # ---- 只在 support>0 的类上做宏平均 ----
        if macro_exclude_zero_support:
            mask = (sup > 0)
            macroP = float(P[mask].mean().item()) if mask.any() else 0.0
            macroR = float(R[mask].mean().item()) if mask.any() else 0.0
            macroF1 = float(F[mask].mean().item()) if mask.any() else 0.0
        else:
            macroP, macroR, macroF1 = float(P.mean().item()), float(R.mean().item()), float(F.mean().item())

        # ---- 逐类行：可选地隐藏 support=0 的类别 ----
        rows = []
        for i, n in enumerate(class_names):
            if drop_rows_with_zero_support and sup[i].item() == 0:
                continue
            rows.append({
                "name": n,
                "TP": int(tp[i]),
                "Support": int(sup[i]),
                "Pred": int(pre[i]),
                "P": float(P[i]),
                "R": float(R[i]),
                "F1": float(F[i]),
            })

        overall = float((y_true == y_pred).float().mean().item()) if y_true.numel() else 0.0
        return rows, overall, macroP, macroR, macroF1

    rows_b, acc_b, mPb, mRb, mFb = _per_class_table(yb_list, pb_list, ["normal", "anomaly"])
    rows_c, acc_c, mPc, mRc, mFc = _per_class_table(yc_list, pc_list, ["normal", "struct", "temporal"])
    rows_t, acc_t, mPt, mRt, mFt = _per_class_table(
        yt_list, pt_list, type_names, valid_mask_list=mt_list,
        drop_rows_with_zero_support=True, macro_exclude_zero_support=True
    )

    def _print_table(title, rows, acc, mP, mR, mF):
        if title:
            print(f"[Report] {title}\n")
        print(f"{'class':22s} {'TP':>6s} {'Support':>8s} {'Pred':>6s} {'Precision':>10s} {'Recall':>8s} {'F1':>8s}")
        for r in rows:
            print(f"{r['name']:22s} {r['TP']:6d} {r['Support']:8d} {r['Pred']:6d} "
                  f"{r['P']:10.4f} {r['R']:8.4f} {r['F1']:8.4f}")
        print("-"*70)
        print(f"overall_acc                     {acc:.4f}")
        print(f"macro(P/R/F1)                   {mP:.4f}   {mR:.4f}   {mF:.4f}\n")

    _print_table("Binary (det) — 正常/异常", rows_b, acc_b, mPb, mRb, mFb)
    _print_table("Coarse-3 (c3)", rows_c, acc_c, mPc, mRc, mFc)
    print("[Report] Fine Type (type) — 仅统计 m_type==1 的样本\n")
    _print_table("", rows_t, acc_t, mPt, mRt, mFt)

@torch.no_grad()
def collect_per_class_reports(model, loader, device, type_names, keep_types: Optional[Set[int]]=None):
    """
    和 print_per_class_reports 同源：返回 dict，不打印。
    每个头包含 overall_acc、macroP、macroR、macroF1、rows(含 TP/Support/Pred/P/R/F1)。
    """
    model.eval()
    yb_list, pb_list = [], []
    yc_list, pc_list = [], []
    yt_list, pt_list, mt_list = [], [], []

    for g, lab, *_ in loader:
        g = g.to(device)
        out = model(g)

        yb = lab["y_bin"].to(device).view(-1).long()
        pb = (out["logit_bin"].view(-1) > 0).long()
        yb_list.append(yb.cpu()); pb_list.append(pb.cpu())

        yc = lab["y_c3"].to(device).view(-1).long()
        pc = out["logits_c3"].argmax(dim=1).view(-1).long()
        yc_list.append(yc.cpu()); pc_list.append(pc.cpu())

        mt = lab["m_type"].to(device).view(-1).bool()
        yt = lab["y_type"].to(device).view(-1).long()
        logits_t = out["logits_type"]
        if keep_types is not None:
            K = logits_t.size(-1)
            mask = torch.full((K,), float('-inf'), device=logits_t.device)
            mask[list(keep_types)] = 0.0
            logits_t = logits_t + mask
        pt = logits_t.argmax(dim=1).view(-1).long()
        yt_list.append(yt.cpu()); pt_list.append(pt.cpu()); mt_list.append(mt.cpu())

    def _per_class_table(y_true_list, y_pred_list, names, valid_mask_list=None):
        y_true = torch.cat(y_true_list) if len(y_true_list) else torch.empty(0, dtype=torch.long)
        y_pred = torch.cat(y_pred_list) if len(y_pred_list) else torch.empty(0, dtype=torch.long)
        if valid_mask_list is not None:
            m = torch.cat(valid_mask_list).bool()
            y_true = y_true[m]; y_pred = y_pred[m]

        K = len(names)
        tp = torch.zeros(K, dtype=torch.long)
        sup = torch.zeros(K, dtype=torch.long)
        pre = torch.zeros(K, dtype=torch.long)

        for k in range(K):
            mk_t = (y_true == k)
            mk_p = (y_pred == k)
            sup[k] = mk_t.sum()
            pre[k] = mk_p.sum()
            tp[k]  = (mk_t & mk_p).sum()

        P = tp.float() / pre.clamp_min(1).float()
        R = tp.float() / sup.clamp_min(1).float()
        F = 2 * P * R / (P + R + 1e-12)

        rows = [{
            "name": names[i],
            "TP": int(tp[i]),
            "Support": int(sup[i]),
            "Pred": int(pre[i]),
            "P": float(P[i]),
            "R": float(R[i]),
            "F1": float(F[i]),
        } for i in range(K)]

        overall = float((y_true == y_pred).float().mean().item()) if y_true.numel() else 0.0
        macroP  = float(P.mean().item()) if K else 0.0
        macroR  = float(R.mean().item()) if K else 0.0
        macroF1 = float(F.mean().item()) if K else 0.0
        return {"overall_acc": overall, "macroP": macroP, "macroR": macroR, "macroF1": macroF1, "rows": rows}

    return {
        "binary": _per_class_table(yb_list, pb_list, ["normal","anomaly"]),
        "c3":     _per_class_table(yc_list, pc_list, ["normal","struct","temporal"]),
        "type":   _per_class_table(yt_list, pt_list, type_names, valid_mask_list=mt_list),
    }

# ========== 保存摘要与权重 ==========
def _sha1(path: str) -> Optional[str]:
    try:
        with open(path,"rb") as f:
            return hashlib.sha1(f.read()).hexdigest()
    except Exception:
        return None

def save_run_summary(save_dir: str, args: dict, data_root: str,
                     metrics_dict: dict, reports_dict: dict,
                     stats, type_names: List[str], keep_types: Optional[Set[int]]):
    """写 run_summary.json：总体指标+逐类表+参数+词表指纹+stats 概要。"""
    os.makedirs(save_dir, exist_ok=True)
    mu, sd = stats if stats is not None else ({}, {})
    summary = {
        "data_root": data_root,
        "vocab_sha1": _sha1(os.path.join(data_root, "vocab.json")),
        "args": args,
        "metrics_test": metrics_dict,
        "reports_test": reports_dict,
        "type_names": type_names,
        "keep_types": sorted(list(keep_types)) if keep_types is not None else None,
        "stats_keys": {"mu_cnt": len(mu), "sd_cnt": len(sd)}
    }
    out = os.path.join(save_dir, "run_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[OK] saved test summary -> {out}")

def save_ckpt(path: str, model_state, stats, args_dict: dict,
              type_names: List[str], keep_types: Optional[Set[int]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "state_dict": model_state,
        "stats": stats,                 # (mu_dict, sd_dict)
        "args": args_dict,
        "type_names": type_names,
        "keep_types": sorted(list(keep_types)) if keep_types is not None else None,
    }, path)
