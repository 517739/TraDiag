# -*- coding: utf-8 -*-
import os, json, csv
from typing import List, Dict, Tuple, Optional
from collections import OrderedDict, Counter
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.utils.data as tud
from torch.utils.data import Subset
import dgl
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_recall_curve

# [PATCH] NLL 常数项
import math
LOG_TWO_PI = math.log(2.0 * math.pi)

class JSONLDataset(Dataset):
    """
    与 v2 兼容：train/val/test.jsonl，每行一个 trace：
    {
      "trace_id": "...",
      "nodes": [{"api_id":..., "status_id":..., "node_id":..., "depth":..., "parent":..., "latency_ms"/"latency":...}],
      "edges": [[src, dst], ...],
      "y_bin": 0/1, "y_c3": 0/1/2, "y_type": int,
      "rca_idx": 可选
      "ctx": [ ... ]  # 可选，trace级上下文，加载时复制到每个节点
    }
    """
    def __init__(self, root: str, split: str, cache_size: int = 100000):
        super().__init__()
        self.path = os.path.join(root, f"{split}.jsonl")
        with open(self.path, 'r', encoding='utf-8') as f:
            self.items = [json.loads(l) for l in f]
        self._cache_size  = int(cache_size) if cache_size and cache_size > 0 else 0
        self._graph_cache = OrderedDict() if self._cache_size > 0 else None
        # 可选：从 vocab.json 读取细分类列表，构建 fault_type -> index 的映射
        self._type2idx: Dict[str, int] = {}
        self._idx2type: List[str] = []
        try:
            vocab_path = os.path.join(root, 'vocab.json')
            if os.path.isfile(vocab_path):
                with open(vocab_path, 'r', encoding='utf-8') as vf:
                    vocab = json.load(vf)
                type_names = vocab.get('type_names', None)
                if isinstance(type_names, list) and len(type_names) > 0:
                    self._idx2type = [str(name) for name in type_names]
                    self._type2idx = {name: idx for idx, name in enumerate(self._idx2type)}
                    for idx, name in enumerate(self._idx2type):
                        lower = name.lower()
                        if lower not in self._type2idx:
                            self._type2idx[lower] = idx
        except Exception:
            self._type2idx = {}
            self._idx2type = []

    def __len__(self): return len(self.items)

    def _build_graph(self, r: dict) -> dgl.DGLGraph:
        nodes = list(r.get('nodes', []))
        n = len(nodes)

        def _get_int(obj, keys, default=0):
            for k in keys:
                if k in obj and obj[k] is not None:
                    try:
                        return int(obj[k])
                    except (ValueError, TypeError):
                        try:
                            return int(float(obj[k]))
                        except (ValueError, TypeError):
                            continue
            return default

        def _get_float(obj, keys, default=0.0):
            for k in keys:
                if k in obj and obj[k] is not None:
                    try:
                        return float(obj[k])
                    except (ValueError, TypeError):
                        continue
            return default

        edges_raw = r.get('edges') or r.get('edge_index') or []
        src, dst = [], []
        for e in edges_raw:
            if isinstance(e, dict):
                u = _get_int(e, ('src', 'source', 'from'), 0)
                v = _get_int(e, ('dst', 'target', 'to'), 0)
            else:
                if not isinstance(e, (list, tuple)) or len(e) < 2:
                    continue
                u = _get_int({'value': e[0]}, ('value',), 0)
                v = _get_int({'value': e[1]}, ('value',), 0)
            if 0 <= u < n and 0 <= v < n:
                src.append(u)
                dst.append(v)

        api = []
        status = []
        node = []
        parent = []
        depth = []
        pos = []
        lat_list = []
        for i, nd in enumerate(nodes):
            api.append(_get_int(nd, ('api_id', 'api', 'apiId'), 0))
            status.append(_get_int(nd, ('status_id', 'status', 'statusId'), 0))
            node.append(_get_int(nd, ('node_id', 'node', 'nodeId'), 0))

            pval = nd.get('parent')
            if pval is None:
                pval = nd.get('parent_id') or nd.get('ParentID')
            try:
                p = int(pval) if pval is not None else -1
            except (ValueError, TypeError):
                p = -1
            parent.append(p if 0 <= p < n else -1)

            dval = nd.get('depth')
            try:
                depth.append(int(dval) if dval is not None else None)
            except (ValueError, TypeError):
                depth.append(None)

            pos.append(_get_int(nd, ('pos', 'position', 'order'), i))
            lat_list.append(_get_float(nd, ('latency_ms', 'lat_ms', 'latency'), 0.0))

        if any(d is None for d in depth) and n > 0:
            depth_cache = {}

            def compute_depth(idx: int) -> int:
                if idx in depth_cache:
                    return depth_cache[idx]
                d = depth[idx]
                if d is not None:
                    depth_cache[idx] = int(d)
                else:
                    p = parent[idx]
                    if p < 0 or p == idx:
                        depth_cache[idx] = 0
                    else:
                        depth_cache[idx] = compute_depth(p) + 1
                return depth_cache[idx]

            for i in range(n):
                compute_depth(i)
            depth = [depth_cache.get(i, 0) for i in range(n)]
        else:
            depth = [0 if d is None else int(d) for d in depth]

        if not src:
            for i, p in enumerate(parent):
                if 0 <= p < n:
                    src.append(p)
                    dst.append(i)

        src_t = torch.tensor(src, dtype=torch.int64) if src else torch.tensor([], dtype=torch.int64)
        dst_t = torch.tensor(dst, dtype=torch.int64) if dst else torch.tensor([], dtype=torch.int64)
        g = dgl.graph((src_t, dst_t), num_nodes=n)

        api_t = torch.tensor(api, dtype=torch.long) if n else torch.zeros(0, dtype=torch.long)
        status_t = torch.tensor(status, dtype=torch.long) if n else torch.zeros(0, dtype=torch.long)
        node_t = torch.tensor(node, dtype=torch.long) if n else torch.zeros(0, dtype=torch.long)
        depth_t = torch.tensor(depth, dtype=torch.long) if n else torch.zeros(0, dtype=torch.long)
        parent_t = torch.tensor(parent, dtype=torch.long) if n else torch.zeros(0, dtype=torch.long)
        pos_t = torch.tensor(pos if pos else list(range(n)), dtype=torch.long) if n else torch.zeros(0, dtype=torch.long)
        lat = torch.tensor(lat_list, dtype=torch.float32) if n else torch.zeros(0, dtype=torch.float32)

        g.ndata.update({'api': api_t, 'status': status_t, 'node': node_t, 'depth': depth_t,
                        'parent': parent_t, 'pos': pos_t, 'lat_ms': lat})
        if 'ctx' in r and isinstance(r['ctx'], list) and len(r['ctx']) > 0:
            g.ndata['ctx'] = torch.tensor(r['ctx'], dtype=torch.float32).repeat(n, 1)
        return g

    def __getitem__(self, idx: int):
        if self._graph_cache is not None and idx in self._graph_cache:
            g = self._graph_cache[idx]
            self._graph_cache.move_to_end(idx)
        else:
            r = self.items[idx]
            g = self._build_graph(r)
            if self._graph_cache is not None:
                self._graph_cache[idx] = g
                if len(self._graph_cache) > self._cache_size:
                    self._graph_cache.popitem(last=False)
        r = self.items[idx]

        def _to_int(val, default=0):
            try:
                return int(val)
            except (ValueError, TypeError):
                try:
                    return int(float(val))
                except (ValueError, TypeError):
                    return default

        y_bin = torch.tensor(_to_int(r.get('y_bin', 0), 0), dtype=torch.long)
        y_c3 = torch.tensor(_to_int(r.get('y_c3', r.get('y_c3_id', 0)), 0), dtype=torch.long)

        y_type_val = -1
        if 'fault_type' in r and isinstance(r.get('fault_type'), str):
            ft_raw = r.get('fault_type')
            ft = str(ft_raw).strip()
            if self._type2idx:
                key = ft if ft in self._type2idx else ft.lower()
                y_type_val = int(self._type2idx.get(key, -1))
        else:
            try:
                y_type_val = _to_int(r.get('y_type', -1), -1)
            except Exception:
                y_type_val = -1
        y_type = torch.tensor(int(y_type_val), dtype=torch.long)
        m_type = torch.tensor(_to_int(r.get('m_type', 1), 1), dtype=torch.long)
        y_rca = torch.tensor(_to_int(r.get('rca_idx', -1), -1), dtype=torch.long)
        return g, {
            'y_bin': y_bin,
            'y_c3': y_c3,
            'y_type': y_type,
            'm_type': m_type,
            'rca_idx': y_rca,
            'trace_idx': torch.tensor(idx, dtype=torch.long)
        }


def collate_batch(batch):
    gs, ys = zip(*batch)
    bg = dgl.batch(gs)
    keys = ys[0].keys()
    y = {k: torch.stack([d[k] for d in ys], dim=0) for k in keys}
    return bg, y


# ---------- 通用日志与数据统计 ----------
def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def resolve_type_names(dataset) -> Optional[List[str]]:
    if dataset is None:
        return None
    names = getattr(dataset, '_idx2type', None)
    if isinstance(names, list) and len(names) > 0:
        return list(names)
    if isinstance(dataset, Subset):
        return resolve_type_names(dataset.dataset)
    return None

def iter_items(ds):
    if isinstance(ds, Subset):
        base = ds.dataset
        for i in ds.indices:
            yield base.items[int(i)]
        return
    for r in getattr(ds, 'items', []):
        yield r

def node_stats(items: List[dict]) -> Dict[str, float]:
    sizes = [len(r.get('nodes', [])) for r in items]
    if not sizes:
        return dict(cnt=0, min=0, mean=0.0, p50=0.0, p90=0.0, max=0)
    v = np.asarray(sizes, dtype=np.int32)
    return dict(
        cnt=int(v.size), min=int(v.min()), mean=float(v.mean()),
        p50=float(np.percentile(v, 50)), p90=float(np.percentile(v, 90)), max=int(v.max())
    )

def log_stage_a_stats(ds_tr: JSONLDataset, ds_va, ds_te, normal_idx: List[int]):
    log(f"[A][data] train={len(ds_tr)}  val={len(ds_va)}  test={len(ds_te)}")
    log(f"[A][data] normal-only(train)={len(normal_idx)}")
    tr_pos = sum(int(r.get('y_bin', 0)) == 1 for r in ds_tr.items)
    log(f"[A][data] train pos={tr_pos}/{len(ds_tr)}")
    try:
        tr_norm_items = [ds_tr.items[i] for i in normal_idx]
    except Exception:
        tr_norm_items = []
    ns_tr = node_stats(tr_norm_items)
    ns_va = node_stats(list(iter_items(ds_va)))
    ns_te = node_stats(list(iter_items(ds_te)))
    log(f"[A][nodes] train_normal cnt={ns_tr['cnt']} min/mean/p50/p90/max={ns_tr['min']}/{ns_tr['mean']:.1f}/{ns_tr['p50']:.1f}/{ns_tr['p90']:.1f}/{ns_tr['max']}")
    log(f"[A][nodes] val cnt={ns_va['cnt']} min/mean/p50/p90/max={ns_va['min']}/{ns_va['mean']:.1f}/{ns_va['p50']:.1f}/{ns_va['p90']:.1f}/{ns_va['max']}")
    log(f"[A][nodes] test cnt={ns_te['cnt']} min/mean/p50/p90/max={ns_te['min']}/{ns_te['mean']:.1f}/{ns_te['p50']:.1f}/{ns_te['p90']:.1f}/{ns_te['max']}")

def log_stage_b_stats(ds_tr: JSONLDataset, ds_va, ds_te):
    t2i = getattr(ds_tr, '_type2idx', {}) if hasattr(ds_tr, '_type2idx') else {}
    i2t = getattr(ds_tr, '_idx2type', []) if hasattr(ds_tr, '_idx2type') else []
    fault_train = [r for r in ds_tr.items if int(r.get('y_bin', 0)) == 1]
    c_tr = Counter()
    for r in fault_train:
        ft = r.get('fault_type', None)
        if isinstance(ft, str):
            key = ft if ft in t2i else ft.lower()
            if key in t2i:
                c_tr[i2t[t2i[key]]] += 1
    total_fault = len(fault_train)
    labeled_fault = sum(c_tr.values())
    log(f"[B][data] train fault samples={total_fault}  with fine-type={labeled_fault}")
    if labeled_fault:
        top = ', '.join(f"{k}:{v}" for k, v in c_tr.most_common(5))
        log(f"[B][data] top types (train): {top}")

    def count_types(ds_any):
        c = Counter()
        for r in iter_items(ds_any):
            ft = r.get('fault_type', None)
            if isinstance(ft, str):
                key = ft if ft in t2i else ft.lower()
                if key in t2i:
                    c[i2t[t2i[key]]] += 1
        return c
    c_va = count_types(ds_va)
    c_te = count_types(ds_te)
    log(f"[B][data] val labeled={sum(c_va.values())}  test labeled={sum(c_te.values())}")
    if c_va:
        log(f"[B][data] top types (val): {', '.join(f'{k}:{v}' for k,v in c_va.most_common(5))}")
    if c_te:
        log(f"[B][data] top types (test): {', '.join(f'{k}:{v}' for k,v in c_te.most_common(5))}")

def log_stage_c_stats(ds_te):
    items = list(iter_items(ds_te))
    covered = 0
    for r in items:
        n = len(r.get('nodes', []))
        ridx = int(r.get('rca_idx', -1))
        if 0 <= ridx < max(1, n):
            covered += 1
    rate = covered / max(1, len(items))
    log(f"[C][data] test={len(items)}  rca-covered={covered}  cover_rate={rate:.3f}")


# ---------- 统计/权重 ----------
def fit_latency_stats(items: List[dict]) -> Tuple[Dict[int, float], Dict[int, float]]:
    api_vals = {}
    for r in items:
        for nd in r['nodes']:
            a = int(nd.get('api_id', 0))
            if 'latency_ms' in nd: v = float(nd['latency_ms'])
            elif 'latency' in nd:  v = float(nd['latency'])
            else:                  v = 0.0
            api_vals.setdefault(a, []).append(v)
    mu, sd = {}, {}
    for a, vals in api_vals.items():
        v = np.asarray(vals, dtype=np.float32)
        mu[a] = float(np.mean(v))
        sd[a] = float(np.std(v) + 1e-6)
    return mu, sd


def class_weights_from_dataset(ds: JSONLDataset, num_classes: int, normal_only: bool = True) -> torch.Tensor:
    cnt = np.zeros(num_classes, dtype=np.float32)
    for r in ds.items:
        if normal_only and int(r.get('y_bin', 0)) != 0:
            continue
        for nd in r['nodes']:
            sid = int(nd.get('status_id', 0))
            if 0 <= sid < num_classes:
                cnt[sid] += 1
    if cnt.sum() == 0:
        return torch.ones(num_classes, dtype=torch.float32)
    freq = cnt / cnt.sum()
    w = 1.0 / (freq + 1e-6)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


# ---------- Stage A 评估：两种口径（mse / nll） ----------
@torch.no_grad()
def _graph_anomaly_score_mse(g, lat_hat_ms, stat_hat, mu, sd, topk=0.2, alpha_lat=1.0, beta_stat=1.0):
    """log1p 域 z-MSE + 状态 NLL"""
    lat_log_obs = torch.log1p(torch.clamp(g.ndata['lat_ms'], min=0.0))
    lat_log_hat = torch.log1p(torch.clamp(lat_hat_ms, min=0.0))

    apinp = g.ndata['api'].cpu().numpy()
    mu_v = torch.tensor([mu.get(int(a), float(lat_log_obs.mean())) for a in apinp], device=lat_log_obs.device)
    sd_v = torch.tensor([sd.get(int(a), 0.5) for a in apinp], device=lat_log_obs.device).clamp_min(0.5)

    z     = (lat_log_obs - mu_v) / sd_v
    z_hat = (lat_log_hat - mu_v) / sd_v
    lat_err = (torch.clamp(z_hat, -6, 6) - torch.clamp(z, -6, 6)) ** 2

    logp = F.log_softmax(stat_hat, dim=-1)
    stat_nll = -logp[torch.arange(stat_hat.shape[0], device=stat_hat.device),
                     g.ndata['status'].to(stat_hat.device)]

    node_err = alpha_lat * lat_err + beta_stat * stat_nll
    node_err = torch.nan_to_num(node_err, nan=0.0, posinf=1e6, neginf=0.0)

    k = max(1, int(node_err.shape[0] * topk))
    return torch.topk(node_err, k).values.mean().detach().cpu().item()

@torch.no_grad()
def _graph_anomaly_score_nll(g, lat_mu_hat, lat_logvar_hat, stat_hat, topk=0.2, alpha_lat=1.0, beta_stat=1.0):
    """log1p 域 高斯 NLL（异方差，含 0.5·log(2π)）+ 状态 NLL"""
    lat_log_obs = torch.log1p(torch.clamp(g.ndata['lat_ms'], min=0.0))
    logvar = torch.clamp(
        torch.nan_to_num(lat_logvar_hat, nan=0.0, posinf=10.0, neginf=-10.0),
        min=torch.log(torch.tensor(0.5**2, device=lat_log_obs.device)),
        max=torch.log(torch.tensor(10.0**2, device=lat_log_obs.device))
    )
    lat_nll = 0.5 * (torch.exp(-logvar) * (lat_log_obs - lat_mu_hat)**2 + logvar + LOG_TWO_PI)

    logp = F.log_softmax(stat_hat, dim=-1)
    stat_nll = -logp[torch.arange(stat_hat.shape[0], device=stat_hat.device),
                     g.ndata['status'].to(stat_hat.device)]

    node_err = alpha_lat * lat_nll + beta_stat * stat_nll
    node_err = torch.nan_to_num(node_err, nan=0.0, posinf=1e6, neginf=0.0)

    k = max(1, int(node_err.shape[0] * topk))
    return torch.topk(node_err, k).values.mean().detach().cpu().item()

# ---------- Stage B ----------
@torch.no_grad()
def evaluate_stage_b(model, loader, device, return_details: bool = False, type_names: Optional[List[str]] = None):
    model.eval()
    tot, corr = 0, 0
    preds, gts = [], []
    for g, y in loader:
        g = g.to(device)
        out = model(g, vae_mode=False)
        pred = torch.argmax(out['logits_type'], dim=-1).cpu()
        gt = y['y_type']
        mask = gt >= 0
        corr += (pred[mask] == gt[mask]).sum().item()
        tot  += mask.sum().item()
        if return_details and mask.any():
            preds.append(pred[mask].numpy())
            gts.append(gt[mask].cpu().numpy())

    acc = corr / max(tot, 1)
    if not return_details:
        return acc

    if gts:
        gt_arr = np.concatenate(gts)
    else:
        gt_arr = np.empty((0,), dtype=np.int64)
    if preds:
        pred_arr = np.concatenate(preds)
    else:
        pred_arr = np.empty((0,), dtype=np.int64)

    if type_names is not None and len(type_names) > 0:
        label_names = list(type_names)
    else:
        max_idx = -1
        if gt_arr.size:
            max_idx = max(max_idx, int(gt_arr.max()))
        if pred_arr.size:
            max_idx = max(max_idx, int(pred_arr.max()))
        label_names = [f"type_{i}" for i in range(max_idx + 1)] if max_idx >= 0 else []

    metrics = []
    for idx, name in enumerate(label_names):
        support = int((gt_arr == idx).sum()) if gt_arr.size else 0
        predicted = int((pred_arr == idx).sum()) if pred_arr.size else 0
        tp = int(((gt_arr == idx) & (pred_arr == idx)).sum()) if gt_arr.size and pred_arr.size else 0
        precision = tp / predicted if predicted > 0 else 0.0
        recall = tp / support if support > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        metrics.append({
            'index': idx,
            'name': name,
            'support': support,
            'predicted': predicted,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
        })

    return acc, metrics


@torch.no_grad()
def evaluate_stage_a(model, loader, mu, sd, device, topk=0.2, alpha_lat=2.0, beta_stat=1.0, recall_floor=0.95,
                     loss_mode: str = 'nll'):  # [PATCH] 新增 loss_mode
    model.eval()
    ys, ps = [], []
    for g, y in loader:
        g = g.to(device)
        out = model(g, vae_mode=True)

        if loss_mode == 'mse':
            s = _graph_anomaly_score_mse(
                g, out['lat_hat'], out['stat_hat'], mu, sd,
                topk=topk, alpha_lat=alpha_lat, beta_stat=beta_stat
            )
        else:
            s = _graph_anomaly_score_nll(
                g, out['lat_mu_hat'], out['lat_logvar_hat'], out['stat_hat'],
                topk=topk, alpha_lat=alpha_lat, beta_stat=beta_stat
            )
        ys.append(int(y['y_bin'].item()))
        ps.append(float(s))

    y_true = np.asarray(ys, dtype=np.int32)
    y_score = np.nan_to_num(np.asarray(ps, dtype=np.float64), nan=0.0, posinf=1e6, neginf=0.0)

    mask = np.isfinite(y_score)
    y_true, y_score = y_true[mask], y_score[mask]

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        print("[StageA][WARN] y_true has <2 classes after filtering; skip AUC/PR.")
        prauc = float(np.mean(y_true)) if len(y_true) else 0.0
        roc = prauc
        thr = float(np.median(y_score)) if len(y_score) else 0.0
        y_pred = (y_score >= thr).astype(int) if len(y_score) else np.array([])
        f1r = float(f1_score(y_true, y_pred)) if (len(y_true) and y_true.sum() > 0) else 0.0
        return prauc, roc, f1r, thr

    prauc = float(average_precision_score(y_true, y_score))
    try:
        roc = float(roc_auc_score(y_true, y_score))
    except Exception:
        roc = float('nan')

    precision, recall, thr_list = precision_recall_curve(y_true, y_score)
    f1s = (2 * precision * recall) / (precision + recall + 1e-12)
    ix = int(np.nanargmax(f1s))
    thr = float(thr_list[max(ix - 1, 0)]) if len(thr_list) > 0 else float(np.median(y_score))
    y_pred = (y_score >= thr).astype(int)
    f1r = float(f1_score(y_true, y_pred))
    return prauc, roc, f1r, thr


# ---------- Stage C（RCA）：两种口径（mse / nll） ----------
@torch.no_grad()
def _node_recon_error_mse(model, g, mu, sd, alpha_lat=2.0, beta_stat=1.0):
    out = model(g, vae_mode=True)
    # 延迟：log1p 域 z-MSE
    lat_log_obs = torch.log1p(torch.clamp(g.ndata['lat_ms'], min=0.0))
    lat_log_hat = torch.log1p(torch.clamp(out['lat_hat'], min=0.0))
    apinp = g.ndata['api'].cpu().numpy()
    mu_v = torch.tensor([mu.get(int(a), float(lat_log_obs.mean())) for a in apinp], device=lat_log_obs.device)
    sd_v = torch.tensor([sd.get(int(a), 0.5) for a in apinp], device=lat_log_obs.device).clamp_min(0.5)
    z     = (lat_log_obs - mu_v) / sd_v
    z_hat = (lat_log_hat - mu_v) / sd_v
    lat_err = (torch.clamp(z_hat, -6, 6) - torch.clamp(z, -6, 6)) ** 2

    # 状态：NLL（交叉熵）
    logp = F.log_softmax(out['stat_hat'], dim=-1)
    stat_nll = -logp[torch.arange(logp.shape[0], device=logp.device),
                     g.ndata['status'].to(logp.device)]
    node_err = alpha_lat * lat_err + beta_stat * stat_nll
    return torch.nan_to_num(node_err, nan=0.0, posinf=1e6, neginf=0.0).detach()

@torch.no_grad()
def _node_recon_error_nll(model, g, mu, sd, alpha_lat=2.0, beta_stat=1.0):
    out = model(g, vae_mode=True)
    lat_log_obs = torch.log1p(torch.clamp(g.ndata['lat_ms'], min=0.0))
    mu_hat  = out['lat_mu_hat']
    logvar  = torch.clamp(
        torch.nan_to_num(out['lat_logvar_hat'], nan=0.0, posinf=10.0, neginf=-10.0),
        min=torch.log(torch.tensor(0.5**2, device=lat_log_obs.device)),
        max=torch.log(torch.tensor(10.0**2, device=lat_log_obs.device))
    )
    lat_nll = 0.5 * (torch.exp(-logvar) * (lat_log_obs - mu_hat)**2 + logvar + LOG_TWO_PI)

    logp = F.log_softmax(out['stat_hat'], dim=-1)
    stat_nll = -logp[
        torch.arange(logp.shape[0], device=logp.device),
        g.ndata['status'].to(logp.device)
    ]
    node_err = alpha_lat * lat_nll + beta_stat * stat_nll

    src, dst = g.edges()
    N = g.number_of_nodes()
    child_sum = torch.zeros(N, device=node_err.device).index_add_(0, src, node_err[dst])
    child_cnt = torch.zeros(N, device=node_err.device).index_add_(0, src, torch.ones_like(src, dtype=torch.float32))
    child_mean = child_sum / (child_cnt + 1e-6)

    lambda_deprop = 0.6  # 去传播强度，可调 0.4~0.8
    depth_decay = 0.03  # 深度轻度惩罚，可调 0.02~0.05
    depth = g.ndata['depth'].to(node_err.device).float()

    source_score = node_err - lambda_deprop * child_mean - depth_decay * depth
    return torch.nan_to_num(source_score, nan=0.0, posinf=1e6, neginf=0.0).detach()


@torch.no_grad()
def evaluate_stage_c(model, loader, mu, sd, device, alpha_lat=2.0, beta_stat=1.0, loss_mode: str = 'nll'):  # [PATCH]
    model.eval()
    top1 = top3 = top5 = 0
    covered = 0
    host_topk_all = []
    svc_topk_all = []
    for g, y in loader:
        g = g.to(device)
        if loss_mode == 'mse':
            node_err = _node_recon_error_mse(model, g, mu, sd, alpha_lat, beta_stat)
        else:
            node_err = _node_recon_error_nll(model, g, mu, sd, alpha_lat, beta_stat)
        rank = torch.argsort(node_err, descending=True).tolist()
        gt = int(y['rca_idx'].item()) if 'rca_idx' in y else -1
        if 0 <= gt < g.number_of_nodes():
            covered += 1
            if gt in rank[:1]: top1 += 1
            if gt in rank[:3]: top3 += 1
            if gt in rank[:5]: top5 += 1
        # 物理节点（node_id）Top-K
        host_topk = []
        if 'node' in g.ndata and g.number_of_nodes() > 0:
            node_ids = g.ndata['node'].detach().cpu().numpy()
            err_np = node_err.detach().cpu().numpy()
            host_scores = {}
            for nid, e in zip(node_ids, err_np):
                nid = int(nid)
                score = float(e)
                if nid in host_scores:
                    host_scores[nid] = max(host_scores[nid], score)
                else:
                    host_scores[nid] = score
            host_topk = sorted(host_scores.items(), key=lambda kv: kv[1], reverse=True)[:5]
        host_topk_all.append(host_topk)

        # 服务/接口 Top-K（若不存在 service 字段则回退至 api）
        svc_topk = []
        key_name = 'service' if 'service' in g.ndata else 'api'
        if key_name in g.ndata and g.number_of_nodes() > 0:
            svc_ids = g.ndata[key_name].detach().cpu().numpy()
            err_np = node_err.detach().cpu().numpy()
            svc_scores = {}
            for sid, e in zip(svc_ids, err_np):
                sid = int(sid)
                score = float(e)
                if sid in svc_scores:
                    svc_scores[sid] = max(svc_scores[sid], score)
                else:
                    svc_scores[sid] = score
            svc_topk = sorted(svc_scores.items(), key=lambda kv: kv[1], reverse=True)[:5]
        svc_topk_all.append(svc_topk)
    return {
        'top1': top1 / covered if covered else 0.0,
        'top3': top3 / covered if covered else 0.0,
        'top5': top5 / covered if covered else 0.0,
        'covered': covered,
        'host_topk': host_topk_all,
        'service_topk': svc_topk_all
    }


def append_result_txt(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(text if text.endswith('\n') else text + '\n')


def build_normal_index_from_csv(normal_csv_path: str, train_items: List[dict], out_json_path: str):
    if not os.path.isfile(normal_csv_path):
        raise FileNotFoundError(normal_csv_path)
    trace_set = set()
    with open(normal_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in ('TraceID', 'trace_id', 'traceId', 'traceID', 'id'):
                if k in row and row[k]:
                    trace_set.add(row[k].strip())
                    break
    idxs = [i for i, r in enumerate(train_items) if str(r.get('trace_id', '')) in trace_set]
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(idxs, f, ensure_ascii=False)
    return idxs

def loader_kwargs(args):
    """根据平台和 num_workers 构造 DataLoader kwargs，避免 prefetch_factor 报错"""
    nw = int(getattr(args, "num_workers", 0) or 0)
    pin = bool(getattr(args, "pin_memory", 0))
    if os.name == "nt" or nw <= 0:
        return dict(num_workers=0, pin_memory=False)
    kw = dict(num_workers=nw, pin_memory=pin)
    pf = int(getattr(args, "prefetch_factor", 0) or 0)
    if pf > 0:
        kw["prefetch_factor"] = pf
    return kw


# -------------可视化/统计（保持你之前的实现）---------------
def _count_pos(ds):
    base = ds.dataset if isinstance(ds, Subset) else ds
    idxs = ds.indices if isinstance(ds, Subset) else list(range(len(ds)))
    return sum(int(base.items[i].get('y_bin', 0)) for i in idxs)

@torch.no_grad()
def summarize_datasets(data_root: str) -> dict:
    out = {}
    # A
    A_tr = JSONLDataset(data_root, 'A_train_normal', cache_size=0)
    A_va = JSONLDataset(data_root, 'A_val',          cache_size=0)
    A_te = JSONLDataset(data_root, 'A_test',         cache_size=0)
    out['A'] = {
        'train': {'total': len(A_tr), 'fault': 0},
        'val':   {'total': len(A_va), 'fault': _count_pos(A_va)},
        'test':  {'total': len(A_te), 'fault': _count_pos(A_te)},
    }
    # B
    B_tr = JSONLDataset(data_root, 'B_train_fault',  cache_size=0)
    out['B'] = {'train': {'total': len(B_tr)}}
    try:
        B_va = JSONLDataset(data_root, 'B_val_fault',  cache_size=0)
        out['B']['val']  = {'total': len(B_va)}
    except Exception:
        out['B']['val']  = {'total': 0}
    try:
        B_te = JSONLDataset(data_root, 'B_test_fault', cache_size=0)
        out['B']['test'] = {'total': len(B_te)}
    except Exception:
        out['B']['test'] = {'total': 0}
    # U
    uni_path = os.path.join(data_root, 'unified_test.jsonl')
    out['Unified'] = {'test': {'total': 0}}
    if os.path.isfile(uni_path):
        U_te = JSONLDataset(data_root, 'unified_test', cache_size=0)
        out['Unified']['test']['total'] = len(U_te)
    return out

# ===== 你之前加过的 per-class 指标/表格输出（略） =====
def compute_class_metrics(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          class_names: list) -> pd.DataFrame:
    K = len(class_names)
    labels = list(range(K))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    support   = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    prec_c, rec_c, f1_c, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    prec_macro,  rec_macro,  f1_macro,  _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='macro', zero_division=0
    )
    prec_weight, rec_weight, f1_weight, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='weighted', zero_division=0
    )
    df = pd.DataFrame({
        'support(actual)': support.astype(float),
        'predicted':       predicted.astype(float),
        'precision':       np.round(prec_c, 4),
        'recall':          np.round(rec_c, 4),
        'F1':              np.round(f1_c,   4),
    }, index=class_names)
    total_support   = float(support.sum())
    total_predicted = float(predicted.sum())
    df_macro = pd.DataFrame([{
        'support(actual)': total_support,
        'predicted':       total_predicted,
        'precision':       np.round(prec_macro,  4),
        'recall':          np.round(rec_macro,   4),
        'F1':              np.round(f1_macro,    4),
    }], index=['(overall-macro)'])
    df_weighted = pd.DataFrame([{
        'support(actual)': total_support,
        'predicted':       total_predicted,
        'precision':       np.round(prec_weight, 4),
        'recall':          np.round(rec_weight,  4),
        'F1':              np.round(f1_weight,   4),
    }], index=['(overall-weighted)'])
    df = pd.concat([df, df_macro, df_weighted], axis=0)
    return df

def save_class_metrics_table(df: pd.DataFrame, out_csv: str, out_png: str):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, encoding='utf-8')
    df_show = df.reset_index().rename(columns={'index': 'class'})
    cols = df_show.columns.tolist()
    if cols[0] != 'class':
        cols = ['class'] + [c for c in cols if c != 'class']
        df_show = df_show[cols]
    fig_h = max(2.5, 0.45 * (len(df_show.index) + 1))
    fig_w = 9.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis('off')
    tbl = ax.table(cellText=df_show.values, colLabels=df_show.columns.tolist(), loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.0, 1.2)
    plt.tight_layout(); fig.savefig(out_png, dpi=220); plt.close(fig)

def append_dataset_and_stageA_report(result_path: str, stats: dict,
                                     a_val_auc=None, a_val_prauc=None, a_val_f1=None,
                                     a_test_auc=None, a_test_prauc=None, a_test_f1=None):
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'a', encoding='utf-8') as f:
        f.write("\n==== Dataset Overview ====\n")
        A = stats.get('A', {}); B = stats.get('B', {}); U = stats.get('Unified', {})
        f.write(f"A: train={A.get('train',{}).get('total',0)}"
                f"  val={A.get('val',{}).get('total',0)}(fault={A.get('val',{}).get('fault',0)})"
                f"  test={A.get('test',{}).get('total',0)}(fault={A.get('test',{}).get('fault',0)})\n")
        f.write(f"B: train={B.get('train',{}).get('total',0)}"
                f"  val={B.get('val',{}).get('total',0)}"
                f"  test={B.get('test',{}).get('total',0)}\n")
        f.write(f"Unified_test={U.get('test',{}).get('total',0)}\n")
        if any(x is not None for x in [a_val_auc, a_val_prauc, a_val_f1, a_test_auc, a_test_prauc, a_test_f1]):
            f.write("\n-- Stage A Metrics --\n")
            if a_val_auc is not None:
                f.write(f"VAL: ROC-AUC={a_val_auc:.4f}  PR-AUC={a_val_prauc:.4f}  F1={a_val_f1:.4f}\n")
            if a_test_auc is not None:
                f.write(f"TEST: ROC-AUC={a_test_auc:.4f}  PR-AUC={a_test_prauc:.4f}  F1={a_test_f1:.4f}\n")

def append_stageB_report(result_path: str, overall_acc: Optional[float],
                         df: pd.DataFrame, out_csv: str, out_png: str):
    with open(result_path, 'a', encoding='utf-8') as f:
        f.write("\n-- Stage B (Fine-grained Classification) --\n")
        if overall_acc is not None:
            f.write(f"TEST type-acc={overall_acc:.4f}\n")
        f.write(f"Per-class metrics (CSV): {out_csv}\n")
        f.write(f"Per-class table (PNG):  {out_png}\n")

# 阈值选择与基于阈值的二分类指标
def select_threshold_at_recall(y_true, scores, target_recall=0.95, fallback="max_f1"):
    """
    y_true: 0/1 (np.ndarray)
    scores: 连续分数，越大越“异常”
    返回: (thr, prec, rec, f1, k_idx)
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)

    order = np.argsort(-scores)           # 分数降序
    y = y_true[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    P  = max(1, int((y_true == 1).sum()))
    rec = tp / P
    prec = tp / np.maximum(tp + fp, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)

    ok = np.where(rec >= target_recall)[0]
    if len(ok) == 0:
        if fallback == "max_f1":
            k = int(np.argmax(f1))
        elif fallback == "youden":
            # Youden J = TPR-FPR
            fn = P - tp
            N  = len(y_true) - P
            tpr = rec
            fpr = fp / max(1, N)
            k = int(np.argmax(tpr - fpr))
        else:
            k = int(np.argmax(f1))
    else:
        k = ok[int(np.argmax(f1[ok]))]

    # 取到第 k 个的分数阈值（包含第 k 位）
    thr = scores[order][k] - 1e-12
    return float(thr), float(prec[k]), float(rec[k]), float(f1[k]), int(k)

def collect_bin_labels_and_scores(model, loader, device):
    """
    把二分类的真值与“异常分数”取出来（sigmoid(logit_bin)）
    返回: (y_all, s_all) 皆为 np.ndarray
    """
    model.eval()
    ys, ss = [], []
    with torch.no_grad():
        for g, lab, *_ in loader:
            g = g.to(device)
            out = model(g)
            y = lab["y_bin"].view(-1).cpu().numpy()
            s = torch.sigmoid(out["logit_bin"].view(-1)).cpu().numpy()
            ys.append(y); ss.append(s)
    return np.concatenate(ys), np.concatenate(ss)

def binary_metrics_from_scores(y_true, scores, thr):
    """
    用固定阈值 thr 把分数二值化，然后计算 Acc/Prec/Rec/F1 与混淆矩阵
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    y_pred = (scores >= thr).astype(int)

    acc  = float((y_pred == y_true).mean())
    tp   = int(((y_pred == 1) & (y_true == 1)).sum())
    tn   = int(((y_pred == 0) & (y_true == 0)).sum())
    fp   = int(((y_pred == 1) & (y_true == 0)).sum())
    fn   = int(((y_pred == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec + 1e-12)
    return {
        "thr": thr, "acc": acc, "prec": prec, "rec": rec, "f1": f1,
        "cm": {"TN": tn, "FP": fp, "FN": fn, "TP": tp}
    }
