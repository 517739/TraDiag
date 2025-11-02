#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_aiops_v2_plus.py  (基于 v2 的最小改版)
- 读取 Normal.csv / Service_fault.csv / Node_fault.csv
- 先按 TraceID 哈希切分 train/val/test，然后**仅用 train 构词表**；val/test OOV → <UNK>=0
- 第③/④步加入 tqdm 进度条；③步用 roll_vocabs（轻量加速）
- 输出（保持 v2 的 train/val/test，同时 [PATCH] 额外输出 A/B/Unified）：
    * train.jsonl / val.jsonl / test.jsonl / vocab.json
    * [PATCH] A_train_normal.jsonl, A_val.jsonl, A_test.jsonl
    * [PATCH] B_train_fault.jsonl, B_val_fault.jsonl, B_test_fault.jsonl
    * [PATCH] unified_test.jsonl（可选，默认写；与 A_test 等价）

保持与 v2 一致的字段：每个节点含 start_ms / end_ms / lat_ms；status 默认来自 HttpStatusCode（最小改动）。
所有 split 限额和统计均以 **Trace** 为单位。
"""
import argparse, os, json, random, hashlib, time
from typing import Optional, List, Dict, Tuple
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
from tqdm import tqdm

# ======= 默认路径/超参（与 v2 一致） =======
NORMAL_DIR = 'E:\ZJU\AIOps\Projects\TraDNN\dataset\SplitTrace06-07/2025-06-07_normal_traces.csv'
SERVICE_DIR = 'E:\ZJU\AIOps\Projects\TraDNN\dataset\SplitTrace06-07/2025-06-07_service.csv'
NODE_DIR    = 'E:\ZJU\AIOps\Projects\TraDNN\dataset\SplitTrace06-07/2025-06-07_node.csv'
OUT_DIR     = 'dataset/aiops_06-07_2e5'

PHASE_A_UNSUP_DEFAULT     = 100000    # 正常 trace cap（0 表示不截断）
PHASE_B_PER_FAULT_DEFAULT = 0        # 每细类 cap（0 表示不截断）
SPLIT = (0.8, 0.1, 0.1)              # train/val/test
SEED  = 2025
MIN_TRACE_SPANS = 2                  # 丢弃单 span trace

# 允许的细类（7+3）
SERVICE_FAULTS = {
    "code error","dns error","cpu stress","memory stress",
    "network corrupt","network delay","network loss",
}
NODE_FAULTS = {
    "node cpu stress","node memory stress","node disk fill",
}
IGNORE_SERVICE = {"pod failure","pod kill","target port misconfig"}

# 常见别名归一
SYN = {
    "network-loss":"network loss","network_loss":"network loss",
    "network-delay":"network delay","network delay(s)":"network delay",
    "dns-error":"dns error","code-error":"code error",
    "cpu-stress":"cpu stress","memory-stress":"memory stress",
    "node cpu":"node cpu stress","node memory":"node memory stress","node disk":"node disk fill",
}

# ======= 工具函数（与 v2 一致） =======
def md5mod(s: str, mod=100) -> int:
    h = hashlib.md5(s.strip().encode("utf-8")).hexdigest()
    return int(h[:8], 16) % mod

def url_template(u: str) -> str:
    if not isinstance(u, str): return "NA"
    core = u.split("?")[0].split("#")[0]
    core = core.replace("//","/").rstrip("/")
    return core or "NA"

def norm_fault(x: Optional[str]) -> Optional[str]:
    if x is None: return None
    s = str(x).strip().lower()
    return SYN.get(s, s)

def make_api_key(svc: str, url_tmpl: str) -> str:
    return f"{str(svc)}||{str(url_tmpl)}"

# ======= IO（与 v2 一致） =======
def load_csv(path: str) -> pd.DataFrame:
    cols = ["TraceID","SpanId","ParentID","NodeName","ServiceName","PodName","URL",
            "HttpStatusCode","StatusCode","SpanKind","StartTimeMs","EndTimeMs","fault_type","fault_instance"]
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        df = pd.read_csv(path, engine="python")
    for c in cols:
        if c not in df.columns: df[c] = np.nan
    for c in ["StartTimeMs","EndTimeMs","HttpStatusCode","StatusCode"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["lat_ms"] = (df["EndTimeMs"] - df["StartTimeMs"]).astype(float).clip(lower=0)
    df["fault_type"] = df["fault_type"].apply(norm_fault)
    df["url_tmpl"] = df["URL"].astype(str).apply(url_template)
    df["_node"] = df.get("NodeName","").fillna("").astype(str)
    return df

def dump_jsonl(path: str, items: List[dict], desc: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in tqdm(items, total=len(items), desc=desc, ncols=100):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ======= 分割（与 v2 一致） =======
def split_by_hash(df: pd.DataFrame, trace_col: str, seed=SEED, ratios=SPLIT):
    ids = df[trace_col].astype(str).unique().tolist()
    random.Random(seed).shuffle(ids)
    id_scores = [(tid, md5mod(tid, 1000000)) for tid in ids]
    id_scores.sort(key=lambda x: x[1])
    n = len(id_scores); n_tr = int(n*ratios[0]); n_va = int(n*ratios[1])
    tr_ids = set([tid for tid,_ in id_scores[:n_tr]])
    va_ids = set([tid for tid,_ in id_scores[n_tr:n_tr+n_va]])
    te_ids = set([tid for tid,_ in id_scores[n_tr+n_va:]])
    df_tr = df[df[trace_col].astype(str).isin(tr_ids)].copy()
    df_va = df[df[trace_col].astype(str).isin(va_ids)].copy()
    df_te = df[df[trace_col].astype(str).isin(te_ids)].copy()
    return df_tr, df_va, df_te

# ======= 轻量“滚词”（与 v2 一致） =======
def roll_vocabs(df: pd.DataFrame,
                cols: Dict[str,str],
                api_vocab: Dict[str,int],
                status_vocab: Dict[int,int],
                node_vocab: Dict[str,int],
                service_vocab: Dict[str,int],
                min_trace_size: int = MIN_TRACE_SPANS,
                desc: str = "roll vocabs"):
    trace_col, span_col = cols["trace"], cols["span"]
    svc_col= cols["svc"]
    nunique = int(df[trace_col].astype(str).nunique())
    pbar = tqdm(total=nunique, desc=desc, ncols=100)
    for tid, g in df.groupby(trace_col, sort=False):
        if len(g) < min_trace_size:
            pbar.update(1); continue
        # API / status / node / service
        svc = g[svc_col].astype(str).fillna("NA").tolist()
        url = g["url_tmpl"].astype(str).tolist()
        for s, u in zip(svc, url):
            key = make_api_key(s, u)
            if key not in api_vocab:
                api_vocab[key] = len(api_vocab) + 1
        # [保持 v2] 状态取 HttpStatusCode（最小改动；StatusCode 不参与词表）
        stat = pd.to_numeric(g["HttpStatusCode"], errors="coerce").fillna(0).astype(int).tolist()
        for sc in stat:
            if sc not in status_vocab:
                status_vocab[sc] = len(status_vocab) + 1
        nodes = g["_node"].astype(str).tolist()
        for nd in nodes:
            if nd not in node_vocab:
                node_vocab[nd] = len(node_vocab) + 1
        svcs = g[svc_col].astype(str).fillna("NA").tolist()
        for s in svcs:
            if s not in service_vocab:
                service_vocab[s] = len(service_vocab) + 1
        pbar.update(1)
    pbar.close()

# ======= 记录构建（与 v2 一致：保留 start_ms/end_ms/lat_ms 等） =======
def build_records(df: pd.DataFrame, cols: Dict[str,str],
                  api_vocab: Dict[str,int], status_vocab: Dict[int,int], node_vocab: Dict[str,int],
                  fixed_c3: Optional[int], fault_type_col: str,
                  freeze_vocab: bool,
                  min_trace_size: int = MIN_TRACE_SPANS,
                  service_vocab: Optional[Dict[str,int]] = None,
                  desc: str = "build records") -> List[dict]:
    trace_col = cols["trace"]; span_col=cols["span"]; parent_col=cols["parent"]
    svc_col=cols["svc"]; st_col=cols["start"]; et_col=cols["end"]

    records: List[dict] = []
    nunique = int(df[trace_col].astype(str).nunique())
    pbar = tqdm(total=nunique, desc=desc, ncols=100)

    for tid, g in df.groupby(trace_col, sort=False):
        g = g.sort_values(by=[st_col, et_col, span_col], kind="mergesort").reset_index(drop=True)
        if len(g) < min_trace_size:
            pbar.update(1); continue

        # 词表 & id 映射
        n = len(g)
        api_ids = np.zeros(n, dtype=np.int64)
        stat_ids= np.zeros(n, dtype=np.int64)
        node_ids= np.zeros(n, dtype=np.int64)
        service_ids = np.zeros(n, dtype=np.int64) if service_vocab is not None else None

        for i, row in g.iterrows():
            api_key = make_api_key(row[svc_col], row["url_tmpl"])
            if not freeze_vocab and api_key not in api_vocab:
                api_vocab[api_key] = len(api_vocab) + 1
            api_ids[i] = api_vocab.get(api_key, 0)

            # [保持 v2] 状态取 HttpStatusCode；NaN→0
            status = int(row["HttpStatusCode"]) if not pd.isna(row["HttpStatusCode"]) else 0
            if not freeze_vocab and status not in status_vocab:
                status_vocab[status] = len(status_vocab) + 1
            stat_ids[i] = status_vocab.get(status, 0)

            node = str(row["_node"])
            if not freeze_vocab and node not in node_vocab:
                node_vocab[node] = len(node_vocab) + 1
            node_ids[i] = node_vocab.get(node, 0)

            if service_vocab is not None:
                svc = str(row[svc_col]) if pd.notna(row[svc_col]) else "NA"
                if not freeze_vocab and svc not in service_vocab:
                    service_vocab[svc] = len(service_vocab) + 1
                service_ids[i] = service_vocab.get(svc, 0)

        # parent 索引
        id_to_idx = {str(sid): i for i, sid in enumerate(g[span_col].astype(str).tolist())}
        parent_idx = []
        for pid in g[parent_col].astype(str).tolist():
            j = id_to_idx.get(pid, None)
            parent_idx.append(-1 if (j is None or pid in ["", "nan", "NaN"]) else j)

        # 儿子表
        children = [[] for _ in range(n)]
        for c, p in enumerate(parent_idx):
            if p >= 0:
                children[p].append(c)

        # 根集合（允许多根）
        roots = [i for i,p in enumerate(parent_idx) if p < 0]
        if not roots:
            roots = [int(np.argmin(g[st_col].values))]

        # 全覆盖 DFS
        order: List[int] = []
        visited = [False]*n
        for r in sorted(roots, key=lambda j: (float(g.loc[j, st_col]) if pd.notna(g.loc[j, st_col]) else 0.0)):
            if visited[r]: continue
            stack=[r]
            while stack:
                u=stack.pop()
                if visited[u]: continue
                visited[u]=True
                order.append(u)
                for v in reversed(children[u]):
                    if not visited[v]:
                        stack.append(v)
        if len(order) < n:
            for i in range(n):
                if not visited[i]: order.append(i)

        pos_map = {i:p for p,i in enumerate(order)}
        depth=[0]*n
        for u in order:
            p=parent_idx[u]
            depth[u] = 0 if p<0 else (depth[p]+1)

        # 边（父子）
        edges = [(p,c) for c,p in enumerate(parent_idx) if p>=0]

        # 弱监督 rca: 最大延迟节点
        lat = g["lat_ms"].values.astype(float)
        rca_idx = int(np.argmax(lat)) if n>0 else 0

        # 标签：y_bin / y_c3 / fault_type
        ft = g[fault_type_col].iloc[0]
        ft = norm_fault(ft) if isinstance(ft, str) else None
        if ft in IGNORE_SERVICE:
            ft = None

        if isinstance(ft, str) and ft.strip():
            if ft in SERVICE_FAULTS:
                y_bin, y_c3 = 1, 1
            elif ft in NODE_FAULTS:
                y_bin, y_c3 = 1, 2
            else:
                y_bin, y_c3 = 0, 0
        else:
            y_bin, y_c3 = 0, 0

        nodes=[]
        for i in range(n):
            nodes.append({
                "api_id": int(api_ids[i]),
                "status_id": int(stat_ids[i]),
                "node_id": int(node_ids[i]),
                "latency_ms": float(lat[i]) if not np.isnan(lat[i]) else 0.0,
                "start_ms": float(g.loc[i, st_col]) if pd.notna(g.loc[i, st_col]) else 0.0,
                "end_ms": float(g.loc[i, et_col]) if pd.notna(g.loc[i, et_col]) else 0.0,
                "service": (str(g.loc[i, svc_col]) if pd.notna(g.loc[i, svc_col]) else "NA"),
                "url_tmpl": str(g.loc[i, "url_tmpl"]),
                "pos": int(pos_map.get(i, i)),
                "depth": int(depth[i]),
            })

        records.append({
            "trace_id": str(tid),
            "nodes": nodes,
            "edges": edges,
            "dfs_order": order,
            "y_bin": int(y_bin),
            "y_c3": int(y_c3),
            "fault_type": (ft if (ft and ft!="nan") else None),
            "rca_idx": int(rca_idx),
        })

        pbar.update(1)

    pbar.close()
    return records

# ======= 主入口（在 v2 基础上加入 A/B/Unified 产出） =======
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal", default=NORMAL_DIR)
    ap.add_argument("--service_fault", default=SERVICE_DIR)
    ap.add_argument("--node_fault", default=NODE_DIR)
    ap.add_argument("--outdir", default=OUT_DIR)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--phaseA_unsup_n", type=int, default=PHASE_A_UNSUP_DEFAULT)
    ap.add_argument("--phaseB_per_fault", type=int, default=PHASE_B_PER_FAULT_DEFAULT)
    ap.add_argument("--min_trace_spans", type=int, default=MIN_TRACE_SPANS)

    # [PATCH] 新增可选限额（均以 Trace 为单位；不影响原有参数语义）
    ap.add_argument("--A_train_normal_max", type=int, default=0, help="A 训练 normal 限额（0=不限；追加在 phaseA_unsup_n 之后）")
    ap.add_argument("--A_val_max", type=int, default=0, help="A 验证集 Trace 限额（0=不限）")
    ap.add_argument("--A_test_max", type=int, default=0, help="A 测试集 Trace 限额（0=不限）")
    ap.add_argument("--B_train_cap_per_type", type=int, default=0, help="B 训练每个 fault 类型的上限（0=不限；追加在 phaseB_per_fault 之后）")
    ap.add_argument("--B_val_max", type=int, default=0, help="B 验证集 Trace 限额（0=不限）")
    ap.add_argument("--B_test_max", type=int, default=0, help="B 测试集 Trace 限额（0=不限）")
    ap.add_argument("--write_unified_test", type=int, default=1, help="1=写 unified_test.jsonl（内容等同 A_test）")

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed)

    t0 = time.time()
    print("[1/6] 读取 CSV ...")
    df_n = load_csv(args.normal)
    df_s = load_csv(args.service_fault)
    df_f = load_csv(args.node_fault)

    # 归一化细类、筛选 7+3（与 v2 一致）
    df_s["fault_type"] = df_s["fault_type"].apply(norm_fault)
    df_f["fault_type"] = df_f["fault_type"].apply(norm_fault)
    df_s = df_s[df_s["fault_type"].isin(SERVICE_FAULTS)]
    df_f = df_f[df_f["fault_type"].isin(NODE_FAULTS)]

    # Phase-A 正常样本下采样（0=不截断）——作用在 split 之前（与 v2 一致）
    if args.phaseA_unsup_n > 0 and len(df_n["TraceID"].unique()) > args.phaseA_unsup_n:
        tids = df_n["TraceID"].astype(str).unique().tolist()
        random.shuffle(tids)
        keep = set(tids[:args.phaseA_unsup_n])
        df_n = df_n[df_n["TraceID"].astype(str).isin(keep)].copy()

    # Phase-B 每细类 ≤ K（0=不截断）——作用在 split 之前（与 v2 一致）
    def cap_per_fault(df: pd.DataFrame, k: int) -> pd.DataFrame:
        if k <= 0: return df
        out=[]
        for ft, g in df.groupby("fault_type"):
            tids = g["TraceID"].astype(str).unique().tolist()
            random.shuffle(tids)
            keep=set(tids[:k])
            out.append(g[g["TraceID"].astype(str).isin(keep)])
        return pd.concat(out, axis=0) if out else df
    df_s = cap_per_fault(df_s, args.phaseB_per_fault)
    df_f = cap_per_fault(df_f, args.phaseB_per_fault)

    df_n["__split_hint"] = "normal"
    df_s["__split_hint"] = "svc"
    df_f["__split_hint"] = "node"
    df_all = pd.concat([df_n, df_s, df_f], axis=0).reset_index(drop=True)

    print("[2/6] 按 TraceID 哈希切分 train/val/test ...")
    cols = {"trace":"TraceID","span":"SpanId","parent":"ParentID",
            "svc":"ServiceName","url":"URL","start":"StartTimeMs","end":"EndTimeMs"}
    df_tr, df_va, df_te = split_by_hash(df_all, trace_col=cols["trace"], seed=args.seed, ratios=SPLIT)
    print(f"   train traces≈{df_tr['TraceID'].nunique():>6}  val≈{df_va['TraceID'].nunique():>6}  test≈{df_te['TraceID'].nunique():>6}")

    # 第③步：仅用 train 构词（轻量滚词 + 进度条）
    print("[3/6] 用 train 构词（api/status/node/service） ...")
    t3 = time.time()
    api_vocab: Dict[str,int] = {}
    status_vocab: Dict[int,int] = {}
    node_vocab: Dict[str,int] = {}
    service_vocab: Dict[str,int] = {}
    roll_vocabs(df_tr, cols, api_vocab, status_vocab, node_vocab, service_vocab,
                min_trace_size=args.min_trace_spans,
                desc="[3/6] rolling vocabs (train-only)")
    print(f"   vocab sizes (train-only): api={len(api_vocab)} status={len(status_vocab)} node={len(node_vocab)} service={len(service_vocab)}  （不含<UNK>）")
    print(f"   done in {time.time()-t3:.1f}s")

    # 第④步：构建 JSONL（带进度）
    print("[4/6] 构建 JSONL 记录（train/val/test） ...")
    t4 = time.time()
    rec_train = build_records(df_tr, cols, api_vocab, status_vocab, node_vocab, fixed_c3=None,
                              fault_type_col="fault_type", freeze_vocab=True,
                              min_trace_size=args.min_trace_spans, service_vocab=service_vocab,
                              desc="[4/6] build records (train)")
    rec_val   = build_records(df_va, cols, api_vocab, status_vocab, node_vocab, fixed_c3=None,
                              fault_type_col="fault_type", freeze_vocab=True,
                              min_trace_size=args.min_trace_spans, service_vocab=service_vocab,
                              desc="[4/6] build records (val)")
    rec_test  = build_records(df_te, cols, api_vocab, status_vocab, node_vocab, fixed_c3=None,
                              fault_type_col="fault_type", freeze_vocab=True,
                              min_trace_size=args.min_trace_spans, service_vocab=service_vocab,
                              desc="[4/6] build records (test)")

    # 写 v2 的三份（保持原输出）
    dump_jsonl(os.path.join(args.outdir,"train.jsonl"), rec_train, desc="[4/6] write train.jsonl")
    dump_jsonl(os.path.join(args.outdir,"val.jsonl"),   rec_val,   desc="[4/6] write val.jsonl")
    dump_jsonl(os.path.join(args.outdir,"test.jsonl"),  rec_test,  desc="[4/6] write test.jsonl")
    print(f"   done in {time.time()-t4:.1f}s")

    # [PATCH] —— 第④步之后：从 v2 的三份中派生 A/B/Unified ----
    # A：train=normal-only；val/test=混合
    A_train = [r for r in rec_train if int(r.get("y_bin",0)) == 0]
    A_val   = rec_val   # 混合
    A_test  = rec_test  # 混合

    # 限额（以 Trace 为单位；不打乱，取前 N）
    def _cap_list(lst, cap):
        if cap and cap > 0 and len(lst) > cap:
            return lst[:cap]
        return lst

    A_train = _cap_list(A_train, args.A_train_normal_max if args.A_train_normal_max else 0)
    A_val   = _cap_list(A_val,   args.A_val_max if args.A_val_max else 0)
    A_test  = _cap_list(A_test,  args.A_test_max if args.A_test_max else 0)

    # B：仅 fault；按细类可进一步对 train 限额
    B_train_fault = [r for r in rec_train if int(r.get("y_bin",0)) == 1]
    if args.B_train_cap_per_type and args.B_train_cap_per_type > 0:
        per_type = defaultdict(list)
        for r in B_train_fault:
            per_type[r.get("fault_type","unknown")].append(r)
        capped=[]
        for k in sorted(per_type.keys()):
            capped.extend(per_type[k][:args.B_train_cap_per_type])
        B_train_fault = capped
    B_val_fault  = [r for r in rec_val  if int(r.get("y_bin",0)) == 1]
    B_test_fault = [r for r in rec_test if int(r.get("y_bin",0)) == 1]
    B_val_fault  = _cap_list(B_val_fault,  args.B_val_max if args.B_val_max else 0)
    B_test_fault = _cap_list(B_test_fault, args.B_test_max if args.B_test_max else 0)

    # 统一测试：等同 A_test（混合）
    unified_test = A_test if args.write_unified_test else []

    # 写出 A/B/Unified
    dump_jsonl(os.path.join(args.outdir,"A_train_normal.jsonl"), A_train, desc="[4/6] write A_train_normal.jsonl")
    dump_jsonl(os.path.join(args.outdir,"A_val.jsonl"),          A_val,   desc="[4/6] write A_val.jsonl")
    dump_jsonl(os.path.join(args.outdir,"A_test.jsonl"),         A_test,  desc="[4/6] write A_test.jsonl")

    dump_jsonl(os.path.join(args.outdir,"B_train_fault.jsonl"),  B_train_fault, desc="[4/6] write B_train_fault.jsonl")
    dump_jsonl(os.path.join(args.outdir,"B_val_fault.jsonl"),    B_val_fault,   desc="[4/6] write B_val_fault.jsonl")
    dump_jsonl(os.path.join(args.outdir,"B_test_fault.jsonl"),   B_test_fault,  desc="[4/6] write B_test_fault.jsonl")

    if unified_test:
        dump_jsonl(os.path.join(args.outdir,"unified_test.jsonl"), unified_test, desc="[4/6] write unified_test.jsonl")

    # 第⑤步：写 vocab.json（与 v2 一致）
    print("[5/6] 写入 vocab.json ...")
    type_names = sorted(list(SERVICE_FAULTS)) + sorted(list(NODE_FAULTS))
    vocab = {
        "api_vocab_size":     int(len(api_vocab)+1),   # +1 for <UNK>=0
        "status_vocab_size":  int(len(status_vocab)+1),
        "node_vocab_size":    int(len(node_vocab)+1),
        "service_vocab_size": int(len(service_vocab)+1),
        "type_names": type_names,
        "ctx_dim": 0,
        "split_seed": int(args.seed),
        "notes": "Vocab built on TRAIN only; OOV maps to 0 (<UNK>)"
    }
    with open(os.path.join(args.outdir,"vocab.json"),"w",encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    # 第⑥步：收尾统计（补充 A/B/Unified 概览）
    print("[6/6] 完成.")
    print(f"  输出目录: {args.outdir}")
    print(f"  traces: train={len(rec_train)} val={len(rec_val)} test={len(rec_test)}")
    print(f"  A: train={len(A_train)}  val={len(A_val)}  test={len(A_test)}")
    print(f"  B: train={len(B_train_fault)}  val={len(B_val_fault)}  test={len(B_test_fault)}")
    if unified_test:
        # A_test 是混合集；统计其中故障占比（pos）
        pos_test = sum(1 for r in A_test if int(r.get("y_bin",0))==1)
        print(f"  Unified_test={len(unified_test)}  (pos={pos_test})")
    cnt = Counter([r.get("fault_type") for r in rec_train+rec_val+rec_test if r.get('fault_type')])
    if cnt:
        print("  fine-type counts (all splits):", dict(sorted(cnt.items())))
    print(f"  type_names({len(type_names)}): {type_names}")
    print(f"总耗时：{time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
