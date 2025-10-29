# make_aiops_svnd.py  (v0.2-nodectx-fast)
# -*- coding: utf-8 -*-
"""
将 Node 物理信息与时间窗口上下文整合进 Trace 数据构造，产出三头训练所需样本。
- 输入：--normal Normal.csv --service Service_fault.csv --node Node_fault.csv
- 输出：train/val/test.jsonl、vocab.json（含 node_vocab_size/ctx_dim/type_names）
- 关键特性：
  * NEW: trace 级 _5xx_frac、svc_unique（自动计算，杜绝缺列报错）
  * NEW: Node 时间窗口上下文（±W 分钟）：双指针 + 前缀和（高效）
  * NEW: node_id 写入每个 span；ctx（7 维）写入每条 trace
  * NEW: 三头标签 y_bin / y_c3 / y_type
"""

import os, json, argparse, random
from collections import defaultdict, Counter, deque
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# ------------------ 基础工具 ------------------

def http_bucket(code):
    try:
        c = int(code)
    except Exception:
        return "other"
    if 200 <= c < 300: return "2xx"
    if 300 <= c < 400: return "3xx"
    if 400 <= c < 500: return "4xx"
    if 500 <= c < 600: return "5xx"
    return "other"

def url_template(u: str) -> str:
    if not isinstance(u, str): return "NA"
    core = u.split("?")[0].split("#")[0]
    core = core.replace("//", "/").rstrip("/")
    return core or "NA"

def make_api_key(service: str, url_tmpl: str) -> str:
    s = str(service) if service is not None else "NA_SVC"
    t = str(url_tmpl) if url_tmpl is not None else "NA_URL"
    return f"{s}||{t}"

def compute_depths(span_ids, parent_ids):
    parent = {}
    sids = set(map(str, span_ids))
    for sid, pid in zip(span_ids, parent_ids):
        pid = "" if pd.isna(pid) else str(pid)
        parent[str(sid)] = pid
    def is_root(sid, pid):
        return (pid in {"","0","-1"} or pid==sid or pid.lower()=="none" or pid not in sids)
    depths = {}
    children = defaultdict(list)
    for sid, pid in parent.items():
        if not is_root(sid, pid):
            children[pid].append(sid)
    roots = [sid for sid, pid in parent.items() if is_root(sid, pid)]
    for r in roots:
        depths[r] = 0
        q = deque([r])
        while q:
            u = q.popleft()
            for v in children.get(u, []):
                depths[v] = depths[u] + 1
                q.append(v)
    for sid in span_ids:
        if str(sid) not in depths:
            depths[str(sid)] = 0
    return [depths[str(sid)] for sid in span_ids]

# ------------------ 读取 & 统一列 ------------------

def load_and_tag(path, tag, cols):
    usecols = list(set([
        cols["trace"], cols["span"], cols["parent"], cols["svc"], cols["url"],
        cols["start"], cols["end"], "NodeName", "PodName",
        "HttpStatusCode", "StatusCode", "SpanKind",
        "Normalized_StartTime", "Normalized_EndTime",
        "fault_type", "fault_instance"
    ]))
    try:
        df = pd.read_csv(path, usecols=usecols, low_memory=False)
    except Exception:
        # 如果列不全，退回读全表
        df = pd.read_csv(path, low_memory=False)
    df['__set__'] = tag
    # 补齐缺失列
    need = set(usecols)
    for c in need:
        if c not in df.columns: df[c] = np.nan
    return df

# ------------------ 每 trace 核心统计 ------------------

def per_trace_core(df_t: pd.DataFrame, cols: Dict[str,str]) -> Dict[str, float]:
    n = len(df_t)
    df_t = df_t.copy()
    for c in [cols["start"], cols["end"]]:
        df_t[c] = pd.to_numeric(df_t[c], errors="coerce")
    df_t["lat_ms"] = (df_t[cols["end"]] - df_t[cols["start"]]).astype(float).clip(lower=0)

    # 主导节点（累计时长最大的 NodeName）
    node = df_t["NodeName"].fillna("").astype(str)
    node_dur = df_t.groupby(node)["lat_ms"].sum().to_dict() if n>0 else {}
    dominant_node = max(node_dur, key=node_dur.get) if node_dur else ""

    # 时间
    t0 = float(df_t[cols["start"]].min()) if n>0 else 0.0
    t1 = float(df_t[cols["end"]].max()) if n>0 else 0.0
    tmid = float(np.median(((df_t[cols["start"]].astype(float) + df_t[cols["end"]].astype(float))/2.0).values)) if n>0 else 0.0

    # 错误/HTTP
    http = pd.to_numeric(df_t["HttpStatusCode"], errors="coerce").fillna(0).astype(int)
    is_err = (http >= 500) | (pd.to_numeric(df_t["StatusCode"], errors="coerce").fillna(0).astype(int) != 0)
    http_b = pd.Categorical(http.map(http_bucket), categories=["2xx","3xx","4xx","5xx","other"])
    http_counts = pd.get_dummies(http_b).reindex(columns=["2xx","3xx","4xx","5xx","other"], fill_value=0).sum()

    # 深度
    depths = np.array(compute_depths(df_t[cols["span"]].astype(str).tolist(),
                                     df_t[cols["parent"]].astype(str).tolist()))
    row = {
        "TraceID": df_t[cols["trace"]].iloc[0],
        "dominant_node": dominant_node,
        "trace_t0": float(t0), "trace_t1": float(t1), "trace_tmid": float(tmid),
        "n_spans": float(n),
        "span_dur_sum": float(df_t["lat_ms"].sum()),
        "span_dur_mean": float(df_t["lat_ms"].mean()) if n>0 else 0.0,
        "span_dur_p90": float(np.percentile(df_t["lat_ms"], 90)) if n>0 else 0.0,
        "err_cnt": float(is_err.sum()),
        "err_rate": float(is_err.mean()) if n>0 else 0.0,
        "avg_depth": float(depths.mean()) if n>0 else 0.0,
        "max_depth": float(depths.max()) if n>0 else 0.0,
        "svc_unique": float(df_t[cols["svc"]].fillna("").astype(str).nunique()),
        "node_unique": float(node.nunique()),
        "_5xx_frac": float((http >= 500).mean() if n>0 else 0.0),  # ★ 确保存在，避免后续缺列
    }
    total = float(n) if n>0 else 1.0
    for b in ["2xx","3xx","4xx","5xx","other"]:
        row[f"http_frac_{b}"] = float(http_counts.get(b, 0))/total
    return row

# ------------------ Node 时间窗口上下文（高效版） ------------------

def build_window_context_fast(df_core: pd.DataFrame,
                              win_minutes: float = 3.0,
                              thr_err: float = 0.05,
                              approx_peak: bool = True) -> pd.DataFrame:
    """
    高效计算 Node-centered 时间窗口上下文：
      输入 df_core 必须包含：
      ['TraceID','dominant_node','trace_t0','trace_t1','trace_tmid',
       'span_dur_p90','err_rate','_5xx_frac','y_bin','svc_unique']
      若缺列会自动补 0.0。
      返回列：ctx_traces, ctx_services_unique, ctx_err_rate_mean,
             ctx_5xx_frac_mean, ctx_concurrency_peak, ctx_abn_ratio_error,
             ctx_p90_over_baseline
    approx_peak=True 用窗口大小近似并发峰值（极快，经验足够好）；
    False 时用“t_mid 瞬时并发”近似（starts/ends 二分，仍很快）。
    """
    # 兜底缺列
    for col in ["_5xx_frac", "svc_unique", "err_rate", "span_dur_p90", "y_bin"]:
        if col not in df_core.columns:
            df_core[col] = 0.0

    W = win_minutes * 60_000.0  # ms
    # 正常集 p90 基线
    if "y_bin" in df_core.columns and (df_core["y_bin"]==0).any():
        normal_p90 = df_core[df_core["y_bin"]==0]["span_dur_p90"].quantile(0.95)
    else:
        normal_p90 = df_core["span_dur_p90"].quantile(0.95)

    df_core = df_core.copy()
    df_core["_p90_over"] = (df_core["span_dur_p90"] - normal_p90).clip(lower=0.0)
    df_core["_abn_flag"] = (df_core["err_rate"] > thr_err).astype(np.int8)

    n_total = len(df_core)
    out = {
        "ctx_traces": np.zeros(n_total, np.float32),
        "ctx_services_unique": np.zeros(n_total, np.float32),
        "ctx_err_rate_mean": np.zeros(n_total, np.float32),
        "ctx_5xx_frac_mean": np.zeros(n_total, np.float32),
        "ctx_concurrency_peak": np.zeros(n_total, np.float32),
        "ctx_abn_ratio_error": np.zeros(n_total, np.float32),
        "ctx_p90_over_baseline": np.zeros(n_total, np.float32),
    }

    # 分 node 处理
    groups = defaultdict(list)
    for idx, r in enumerate(df_core.itertuples(index=False)):
        dom = getattr(r, "dominant_node", "")
        if dom: groups[dom].append(idx)

    for node_key, idxs in groups.items():
        arr = df_core.iloc[idxs].sort_values("trace_tmid").reset_index()  # index = 原 df_core 行号
        ridx = arr["index"].to_numpy()
        tmid = arr["trace_tmid"].to_numpy(np.float64)
        t0   = arr["trace_t0"].to_numpy(np.float64)
        t1   = arr["trace_t1"].to_numpy(np.float64)
        err  = arr["err_rate"].to_numpy(np.float32)
        f5   = arr["_5xx_frac"].to_numpy(np.float32)
        svcU = arr["svc_unique"].to_numpy(np.float32)
        p90o = arr["_p90_over"].to_numpy(np.float32)
        abn  = arr["_abn_flag"].to_numpy(np.int16)

        pref_err  = np.concatenate([[0.0], np.cumsum(err , dtype=np.float64)])
        pref_f5   = np.concatenate([[0.0], np.cumsum(f5  , dtype=np.float64)])
        pref_svcU = np.concatenate([[0.0], np.cumsum(svcU, dtype=np.float64)])
        pref_p90o = np.concatenate([[0.0], np.cumsum(p90o, dtype=np.float64)])
        pref_abn  = np.concatenate([[0]  , np.cumsum(abn , dtype=np.int64)])

        if not approx_peak:
            starts = np.sort(t0); ends = np.sort(t1)

        L = 0; R = 0
        n = len(arr)
        for i in range(n):
            left  = tmid[i] - W
            right = tmid[i] + W
            while L < n and tmid[L] < left:  L += 1
            while R < n and tmid[R] <= right: R += 1
            win = R - L
            if win <= 0: continue

            sum_err  = pref_err[R]  - pref_err[L]
            sum_f5   = pref_f5[R]   - pref_f5[L]
            sum_svcU = pref_svcU[R] - pref_svcU[L]
            sum_p90o = pref_p90o[R] - pref_p90o[L]
            cnt_abn  = pref_abn[R]  - pref_abn[L]

            out["ctx_traces"][ridx[i]]            = float(win)
            out["ctx_err_rate_mean"][ridx[i]]     = float(sum_err / win)
            out["ctx_5xx_frac_mean"][ridx[i]]     = float(sum_f5 / win)
            out["ctx_services_unique"][ridx[i]]   = float(sum_svcU / win)   # 均值近似
            out["ctx_abn_ratio_error"][ridx[i]]   = float(cnt_abn / win)
            out["ctx_p90_over_baseline"][ridx[i]] = float(sum_p90o / win)

            if approx_peak:
                out["ctx_concurrency_peak"][ridx[i]] = float(win)
            else:
                # 瞬时并发近似：#starts<=tmid - #ends<tmid
                import bisect
                s_le = bisect.bisect_right(starts, tmid[i])
                e_lt = bisect.bisect_left(ends,   tmid[i])
                out["ctx_concurrency_peak"][ridx[i]] = float(max(0, s_le - e_lt))

    return pd.DataFrame(out)

# 过滤函数
def drop_orphan_traces_df(df: pd.DataFrame, cols: Dict[str,str],
                          root_indicators=frozenset({"-1","0","","nan","none","None"})) -> pd.DataFrame:
    """
    删除包含“父不在本 Trace 的 SpanId 集合中”的 Trace。
    """
    t_col, s_col, p_col = cols["trace"], cols["span"], cols["parent"]
    keep_idx = []
    for tid, g in df.groupby(t_col, sort=False):
        sids = set(g[s_col].astype(str))
        has_orphan = False
        for pid in g[p_col].astype(str):
            if pid not in root_indicators and pid not in sids:
                has_orphan = True
                break
        if not has_orphan:
            keep_idx.append(g.index.values)
    if not keep_idx:
        return df.iloc[0:0].copy()
    import numpy as np
    return df.loc[np.concatenate(keep_idx)].copy()


# ------------------ 记录构造（含 node_id） ------------------

def build_records(df: pd.DataFrame, cols, api_vocab, status_vocab, node_vocab,
                  fixed_c3: Optional[int], fault_type_col: str,
                  min_trace_size=2) -> List[dict]:
    t_col, s_col, p_col = cols["trace"], cols["span"], cols["parent"]
    svc_col, url_col    = cols["svc"], cols["url"]
    st_col, et_col      = cols["start"], cols["end"]

    for c in [st_col, et_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["lat_ms"] = (df[et_col] - df[st_col]).astype(float).clip(lower=0)
    df["url_tmpl"] = df[url_col].apply(url_template)

    # node 词表
    df["_node"] = df.get("NodeName","").fillna("").astype(str)
    for nk in df["_node"].unique():
        if nk not in node_vocab:
            node_vocab[nk] = len(node_vocab) + 1

    records=[]
    for tid, g in df.groupby(t_col, sort=False):
        g = g.dropna(subset=[s_col, st_col, et_col]).copy()
        if len(g) < min_trace_size: continue
        g = g.reset_index(drop=True)

        # 索引与 parent/children
        idx_of = {str(g.loc[i, s_col]): i for i in range(len(g))}
        parent_idx=[]; children=[[] for _ in range(len(g))]; roots=set(range(len(g)))
        for i in range(len(g)):
            pid = g.loc[i, p_col]
            if pd.isna(pid) or str(pid) not in idx_of:
                parent_idx.append(-1)
            else:
                p = idx_of[str(pid)]
                parent_idx.append(p); children[p].append(i); roots.discard(i)

        # 生成DFS序
        starts = g[st_col].to_numpy()
        order = []
        seen = set()
        for root in sorted(roots if roots else [int(np.argmin(g[st_col].values))]):
            stack = [root]
            while stack:
                u = stack.pop()
                if u in seen:
                    continue
                seen.add(u);
                order.append(u)
                for v in reversed(sorted(children[u], key=lambda x: starts[x])):
                    stack.append(v)

        # 词表 id
        api_ids = np.zeros(len(g), dtype=int)
        st_ids  = np.zeros(len(g), dtype=int)
        node_ids= np.zeros(len(g), dtype=int)
        for i in range(len(g)):
            api_key = make_api_key(g.loc[i, svc_col], g.loc[i, "url_tmpl"])
            if api_key not in api_vocab: api_vocab[api_key] = len(api_vocab) + 1
            api_ids[i] = api_vocab[api_key]
            skey = int(pd.to_numeric(g.loc[i, "HttpStatusCode"], errors="coerce")) if "HttpStatusCode" in g.columns else 0
            skey = str(skey)
            if skey not in status_vocab: status_vocab[skey] = len(status_vocab) + 1
            st_ids[i] = status_vocab[skey]
            node_ids[i] = node_vocab.get(str(g.loc[i, "_node"]), 0)

        edges = [[int(parent_idx[i]), int(i)] for i in range(len(g)) if parent_idx[i] >= 0]

        # 标签（y_bin/y_c3）与 fault_type
        if fixed_c3 == 0:
            y_bin, y_c3, ft = 0, 0, None
        else:
            y_bin, y_c3 = 1, int(fixed_c3)
            ft = str(g[fault_type_col].iloc[0]).strip().lower() if fault_type_col in g.columns else None

        nodes=[]
        for i in range(len(g)):
            nodes.append({
                "span_id": str(g.loc[i, s_col]),
                "parent_id": (str(g.loc[i, p_col]) if pd.notna(g.loc[i, p_col]) and str(g.loc[i, p_col]) in idx_of else None),
                "api_id": int(api_ids[i]),
                "status_id": int(st_ids[i]),
                "node_id": int(node_ids[i]),           # NEW
                "latency_ms": float(g.loc[i, "lat_ms"]),
                "start_ms": float(g.loc[i, st_col]),
                "end_ms": float(g.loc[i, et_col]),
                "service": str(g.loc[i, svc_col]) if pd.notna(g.loc[i, svc_col]) else "NA",
                "url_tmpl": str(g.loc[i, "url_tmpl"]),
            })

        rec = {
            "trace_id": str(tid),
            "nodes": nodes,
            "edges": edges,
            "dfs_order": order,
            "y_bin": int(y_bin),
            "y_c3": int(y_c3),
            "fault_type": (ft if (ft and ft!="nan") else None),
        }
        records.append(rec)
    return records

# ------------------ 主流程 ------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal", default="E:\ZJU\AIOps\Projects\TraDNN\TraDiag/trace_svnd_all\dataset\Data/Normal.csv")
    ap.add_argument("--service", default="E:\ZJU\AIOps\Projects\TraDNN\dataset\SplitTrace\service\merged.csv")
    ap.add_argument("--node", default="E:/ZJU/AIOps/Projects/TraDNN/dataset/SplitTrace/node/merged.csv")
    ap.add_argument("--outdir", default="dataset/aiops_svnd")
    ap.add_argument("--win-minutes", type=float, default=3.0)
    ap.add_argument("--approx-peak", type=int, default=1, help="1=窗口大小近似并发峰值(快), 0=瞬时并发近似")
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--min-trace-size", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--drop-orphan-traces", type=int, default=1,
                    help="1=删除包含悬空父指针的 Trace，0=保留")
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed)

    cols = dict(
        trace="TraceID", span="SpanId", parent="ParentID",
        svc="ServiceName", url="URL",
        start="StartTimeMs", end="EndTimeMs"
    )

    print("[1/5] loading csv ...")
    df_n = load_and_tag(args.normal, 'normal', cols)
    df_s = load_and_tag(args.service, 'service_fault', cols)
    df_f = load_and_tag(args.node,   'node_fault', cols)

    if args.drop_orphan_traces:
        print("[1.5/5] dropping orphan-span traces ...")
        df_n = drop_orphan_traces_df(df_n, cols)
        df_s = drop_orphan_traces_df(df_s, cols)
        df_f = drop_orphan_traces_df(df_f, cols)

    # 统一 fault_type 小写（便于映射）
    for df in (df_s, df_f):
        if "fault_type" in df.columns:
            df["fault_type"] = df["fault_type"].apply(lambda x: str(x).strip().lower() if isinstance(x, str) else x)

    print("[2/5] per-trace core features ...")
    core_rows=[]
    for tag, df in [("normal", df_n), ("service", df_s), ("node", df_f)]:
        for i, (tid, g) in enumerate(df.groupby(cols["trace"], sort=False), 1):
            if i % 10000 == 0:
                print(f"  - {tag}: processed {i} traces")
            r = per_trace_core(g, cols)
            if tag == "normal":
                r["y_bin"]=0; r["y_c3"]=0; r["fault_type"]=None
            elif tag == "service":
                r["y_bin"]=1; r["y_c3"]=1; r["fault_type"]=str(g["fault_type"].iloc[0]).strip().lower() if "fault_type" in g.columns else None
            else:
                r["y_bin"]=1; r["y_c3"]=2; r["fault_type"]=str(g["fault_type"].iloc[0]).strip().lower() if "fault_type" in g.columns else None
            core_rows.append(r)
    df_core = pd.DataFrame(core_rows)

    # 细粒度类型表（来自 service+node）
    ft_all = pd.concat([df_s.get("fault_type"), df_f.get("fault_type")], axis=0).dropna().astype(str).str.strip().str.lower()
    type_names = [k for k,_ in sorted(Counter(ft_all.tolist()).items(), key=lambda x:(-x[1], x[0]))]
    type2id = {k:i for i,k in enumerate(type_names)}

    print("[3/5] node-window context (fast) ...")
    subset_cols = ["TraceID","dominant_node","trace_t0","trace_t1","trace_tmid",
                   "span_dur_p90","err_rate","_5xx_frac","y_bin","svc_unique"]
    for c in subset_cols:
        if c not in df_core.columns:
            # 兜底，避免 AttributeError
            df_core[c] = 0.0 if c!="dominant_node" and c!="TraceID" else ""
    ctx_df = build_window_context_fast(
        df_core[subset_cols].copy(),
        win_minutes=args.win_minutes,
        thr_err=0.05,
        approx_peak=bool(args.approx_peak)
    )
    df_core = pd.concat([df_core.reset_index(drop=True), ctx_df.reset_index(drop=True)], axis=1)

    # === 新增：对重复 TraceID 做归并（关键修复） ===
    def reduce_df_core_duplicates(df_core: pd.DataFrame) -> pd.DataFrame:
        """
        将重复 TraceID 聚合：
          - y_bin/y_c3 取最大（优先保留异常/节点级标签）
          - fault_type 取第一个非空（小写去空白）
          - dominant_node 取众数
          - 其他数值（包含 ctx_*）取均值
        """
        import pandas as pd
        agg = {}
        for c in df_core.columns:
            if c in ["TraceID", "fault_type", "dominant_node"]:
                continue
            if pd.api.types.is_numeric_dtype(df_core[c]):
                agg[c] = "mean"
        if "y_bin" in df_core.columns: agg["y_bin"] = "max"
        if "y_c3" in df_core.columns: agg["y_c3"] = "max"

        def first_non_empty(s: pd.Series):
            for v in s:
                if isinstance(v, str) and v.strip() and v.strip().lower() != "nan":
                    return v.strip().lower()
            return None

        def mode_or_first(s: pd.Series):
            vc = s.value_counts()
            return str(vc.index[0]) if len(vc) else ""

        agg["fault_type"] = first_non_empty
        agg["dominant_node"] = mode_or_first
        return df_core.groupby("TraceID", as_index=False).agg(agg)

    dup_cnt = int(df_core.duplicated(subset=["TraceID"]).sum())
    if dup_cnt > 0:
        print(f"[WARN] df_core has {dup_cnt} duplicate TraceID rows; reducing by TraceID ...")
        df_core_red = reduce_df_core_duplicates(df_core)
    else:
        df_core_red = df_core

    print("[4/5] build jsonl records ...")
    api_vocab, status_vocab, node_vocab = {}, {}, {}
    rec_n = build_records(df_n, cols, api_vocab, status_vocab, node_vocab, fixed_c3=0, fault_type_col="fault_type", min_trace_size=args.min_trace_size)
    rec_s = build_records(df_s, cols, api_vocab, status_vocab, node_vocab, fixed_c3=1, fault_type_col="fault_type", min_trace_size=args.min_trace_size)
    rec_f = build_records(df_f, cols, api_vocab, status_vocab, node_vocab, fixed_c3=2, fault_type_col="fault_type", min_trace_size=args.min_trace_size)
    traces = rec_n + rec_s + rec_f

    # 写回 y_type 与 ctx
    core_map = df_core_red.set_index("TraceID").to_dict(orient="index")
    assert df_core_red["TraceID"].is_unique, "TraceID still not unique after reduction"
    for r in traces:
        info = core_map.get(r["trace_id"], {})
        r["y_bin"] = int(info.get("y_bin", r.get("y_bin", 0)))
        r["y_c3"]  = int(info.get("y_c3",  r.get("y_c3", 0)))
        ft = r.get("fault_type", info.get("fault_type", None))
        r["y_type"] = int(type2id.get(ft, -1)) if (ft is not None and str(ft)!="") and r["y_bin"]==1 else -1
        r["ctx"] = [
            float(info.get("ctx_traces", 0.0)),
            float(info.get("ctx_services_unique", 0.0)),
            float(info.get("ctx_err_rate_mean", 0.0)),
            float(info.get("ctx_5xx_frac_mean", 0.0)),
            float(info.get("ctx_concurrency_peak", 0.0)),
            float(info.get("ctx_abn_ratio_error", 0.0)),
            float(info.get("ctx_p90_over_baseline", 0.0)),
        ]

    # ====== 新增：打印详细数据分布 =====
    def log_dataset_stats(df_core, type_names):
        print("\n=== Dataset Statistics ===")
        print(f"Total traces               : {len(df_core)}")
        print(f"Normal traces              : {(df_core['y_bin'] == 0).sum()}")
        print(f"Anomaly traces             : {(df_core['y_bin'] == 1).sum()}")
        print(f"Service-level faults       : {(df_core['y_c3'] == 1).sum()}")
        print(f"Node-level faults          : {(df_core['y_c3'] == 2).sum()}")
        print("----------------------------------------")
        # 各细类
        fine_cnt = df_core[df_core['y_bin'] == 1]['fault_type'].value_counts()
        print("Fault-type breakdown:")
        for ft, cnt in fine_cnt.items():
            print(f"  {ft or 'NULL'} : {cnt}")
        print("----------------------------------------")
        # 保存 csv（可选）
        fine_cnt.to_csv(os.path.join(args.outdir, "fault_type_count.csv"), header=['count'])
    # 先归并、打标签之后再统计
    log_dataset_stats(df_core_red, type_names)

    print("[5/5] split & save ...")
    random.shuffle(traces)
    n=len(traces); n_tr=int(n*args.train_ratio); n_va=int(n*args.val_ratio)
    train=traces[:n_tr]; val=traces[n_tr:n_tr+n_va]; test=traces[n_tr+n_va:]

    def dump_jsonl(path, items):
        with open(path,"w",encoding="utf-8") as f:
            for r in items:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump_jsonl(os.path.join(args.outdir, "train.jsonl"), train)
    dump_jsonl(os.path.join(args.outdir, "val.jsonl"),   val)
    dump_jsonl(os.path.join(args.outdir, "test.jsonl"),  test)

    with open(os.path.join(args.outdir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump({
            "api_vocab_size": len(api_vocab),
            "status_vocab_size": len(status_vocab),
            "node_vocab_size": len(node_vocab),
            "type_names": type_names,
            "ctx_dim": 7,
        }, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote to {args.outdir} | traces={n} train={len(train)} val={len(val)} test={len(test)} "
          f"| api_vocab={len(api_vocab)} status_vocab={len(status_vocab)} node_vocab={len(node_vocab)} "
          f"| types={len(type_names)}")

if __name__ == "__main__":
    main()
