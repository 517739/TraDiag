# make_aiops3c6c.py
# -*- coding: utf-8 -*-
"""
构建三分类（coarse）、六分类（fine，可选）与“按 fault_type 的多分类（superfine）”。
仅 Trace 通道；严格过滤：
  - orphan span（父不在本 trace）：整条 trace 丢弃
  - single-root 约束：根节点数量 != 1 的 trace 丢弃
输出：
- train/val/test JSONL（逐 trace）
- 扁平 CSV（逐 span）
- vocab.json（含 fine/superfine 标签映射）
- brief.txt（分布 + 丢弃原因统计）
"""
import os, json, argparse, random
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

from utils import (
    url_template, make_api_key,
    map_coarse, map_fine,
    STRUCTURAL_TYPES, LATENCY_TYPES, FINE_GROUPS, FINE_LABELS, FINE_INDEX
)

# --------------------------- core ---------------------------

def build_records(df: pd.DataFrame, cols, api_vocab, status_vocab,
                  label_scheme: str,
                  fault_type_series: pd.Series = None,
                  superfine_map: dict = None,
                  fixed_coarse=None,
                  min_trace_size=2,
                  drop_orphan_trace=True,
                  require_single_root=True,
                  drop_stats: dict = None):
    t_col, s_col, p_col = cols["trace"], cols["span"], cols["parent"]
    svc_col, url_col    = cols["svc"], cols["url"]
    st_col, et_col      = cols["start"], cols["end"]

    for c in [st_col, et_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["latency_ms"] = (df[et_col] - df[st_col]).astype(float)

    # 选择状态码列（优先前者，回退后者）
    if cols["status_pref"] in df.columns:
        status = df[cols["status_pref"]]
        if cols["status_fb"] in df.columns:
            status = status.fillna(df[cols["status_fb"]])
    else:
        status = df[cols["status_fb"]] if cols["status_fb"] in df.columns else None
    df["_status_key"] = status.astype("Int64").astype(str) if status is not None else "NA"

    df["url_tmpl"] = df[url_col].apply(url_template)

    records=[]
    for tid, g in df.groupby(t_col, sort=False):
        g = g.dropna(subset=[s_col, st_col, et_col, "latency_ms"]).copy()
        if len(g) < min_trace_size:
            if drop_stats is not None: drop_stats["too_short"] += 1
            continue
        g = g.reset_index(drop=False)  # 保存原行号
        orig_indices = g.iloc[:, 0]

        # ---- 建立 span 索引映射 ----
        idx_of = {str(g.loc[i, s_col]): i for i in range(len(g))}
        parent_idx = []
        children = [[] for _ in range(len(g))]
        roots = set(range(len(g)))
        orphan_found = False
        for i in range(len(g)):
            pid = g.loc[i, p_col]
            if pd.isna(pid) or str(pid) == "-1" or str(pid) not in idx_of:
                # 父 ID 缺失 / -1 / 不在本 trace
                if not pd.isna(pid) and str(pid) not in idx_of and drop_orphan_trace:
                    orphan_found = True
                parent_idx.append(-1)
            else:
                p = idx_of[str(pid)]
                parent_idx.append(p)
                children[p].append(i)
                if i in roots:
                    roots.discard(i)

        if orphan_found and drop_orphan_trace:
            if drop_stats is not None: drop_stats["orphan_span"] += 1
            continue

        # 单根约束
        if require_single_root:
            if len(roots) != 1:
                if drop_stats is not None: drop_stats["multi_or_zero_roots"] += 1
                continue
            root = list(roots)[0]
        else:
            root = min(roots) if roots else int(np.argmin(g[st_col].values))

        for u in range(len(g)):
            children[u].sort(key=lambda j: (g.loc[j, st_col], g.loc[j, s_col]))

        # ---- 词表 id ----
        api_ids = np.zeros(len(g), dtype=int)
        status_ids = np.zeros(len(g), dtype=int)
        for i in range(len(g)):
            api_key = make_api_key(g.loc[i, svc_col], g.loc[i, "url_tmpl"])
            if api_key not in api_vocab:
                api_vocab[api_key] = len(api_vocab) + 1
            api_ids[i] = api_vocab[api_key]
            skey = g.loc[i, "_status_key"] if pd.notna(g.loc[i, "_status_key"]) else "NA"
            if skey not in status_vocab:
                status_vocab[skey] = len(status_vocab) + 1
            status_ids[i] = status_vocab[skey]

        edges = [[int(parent_idx[i]), int(i)] for i in range(len(g)) if parent_idx[i] >= 0]

        # ---- DFS 顺序（供深度/位置计算&可视化） ----
        order=[]
        stack=[root]
        seen=set()
        while stack:
            u=stack.pop()
            if u in seen: continue
            seen.add(u); order.append(u)
            for v in reversed(children[u]): stack.append(v)

        # ---- 标签 ----
        fault_type=None; coarse=None; fine=None; sfl=None
        if fixed_coarse is not None:
            coarse = int(fixed_coarse)
        else:
            raw_idx0 = int(orig_indices.iloc[0])
            ft_val = None if fault_type_series is None else fault_type_series.iloc[raw_idx0] \
                     if raw_idx0 < len(fault_type_series) else None
            fault_type = str(ft_val).strip().lower() if isinstance(ft_val, str) else None
            coarse = map_coarse(fault_type)
            if coarse is None:
                if drop_stats is not None: drop_stats["unknown_fault_type"] += 1
                continue
            fine = map_fine(fault_type)
            if superfine_map and fault_type:
                sfl = superfine_map.get(fault_type, None)

        # ---- 写入策略 ----
        scheme = label_scheme
        if scheme == "coarse":
            pass
        elif scheme == "fine":
            if coarse == 0:      # normal 不参与 fine
                if drop_stats is not None: drop_stats["fine_drop_normal"] += 1
                continue
            if fine is None:
                if drop_stats is not None: drop_stats["fine_unmapped"] += 1
                continue
        elif scheme == "both":
            if coarse == 0:
                fine = -1
            else:
                if fine is None:
                    if drop_stats is not None: drop_stats["fine_unmapped"] += 1
                    continue
        elif scheme == "superfine":
            if coarse == 0:
                if drop_stats is not None: drop_stats["sfn_drop_normal"] += 1
                continue
            if sfl is None:
                if drop_stats is not None: drop_stats["sfn_unmapped"] += 1
                continue
        elif scheme == "both+superfine":
            if coarse == 0:
                fine = -1; sfl = -1
            else:
                if fine is None or sfl is None:
                    if drop_stats is not None: drop_stats["sfn_or_fine_unmapped"] += 1
                    continue
        else:
            raise ValueError(f"Unknown label_scheme: {scheme}")

        # ---- 节点打包 ----
        nodes=[]
        for i in range(len(g)):
            nodes.append({
                "span_id": str(g.loc[i, s_col]),
                "parent_id": (str(g.loc[i, p_col]) if pd.notna(g.loc[i, p_col]) and str(g.loc[i, p_col]) in idx_of else None),
                "api_id": int(api_ids[i]),
                "status_id": int(status_ids[i]),
                "latency_ms": float(g.loc[i, "latency_ms"]),
                "start_ms": float(g.loc[i, st_col]),
                "end_ms": float(g.loc[i, et_col]),
                "service": str(g.loc[i, cols["svc"]]) if pd.notna(g.loc[i, cols["svc"]]) else "NA",
                "url_tmpl": str(g.loc[i, "url_tmpl"]),
            })

        records.append({
            "trace_id": str(tid),
            "coarse_label": int(coarse) if coarse is not None else None,
            "fine_label":   int(fine)   if fine   is not None else None,    # -1 for normal if both/both+superfine
            "superfine_label": int(sfl) if sfl    is not None else None,    # -1 for normal if both+superfine
            "fine_name":    (FINE_LABELS[fine] if (fine is not None and fine >= 0) else None),
            "fault_type":   fault_type,
            "nodes": nodes,
            "edges": edges,
            "dfs_order": order
        })
    return records

def dump_jsonl(path, items):
    with open(path,"w",encoding="utf-8") as f:
        for r in items:
            f.write(json.dumps(r, ensure_ascii=False)+"\n")

def flatten_to_csv(path, items):
    rows=[]
    for r in items:
        tid=r["trace_id"]; cl=r.get("coarse_label"); fl=r.get("fine_label")
        sfl=r.get("superfine_label"); fn=r.get("fine_name"); ft=r.get("fault_type")
        for nd in r["nodes"]:
            rows.append({
                "TraceID": tid, "SpanId": nd["span_id"], "ParentID": nd["parent_id"],
                "ServiceName": nd["service"], "URL_Tmpl": nd["url_tmpl"],
                "API_ID": nd["api_id"], "Status_ID": nd["status_id"],
                "StartTimeMs": nd["start_ms"], "EndTimeMs": nd["end_ms"], "LatencyMs": nd["latency_ms"],
                "coarse_label": cl, "fine_label": fl, "superfine_label": sfl,
                "fine_name": fn, "fault_type": ft,
            })
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")

# --------------------------- CLI ---------------------------

def parse_args():
    ap=argparse.ArgumentParser()
    ap.add_argument("--normal", default="E:\ZJU\AIOps\Projects\TraDNN\TraDiag/trace_svnd_all\dataset\Data/Normal.csv")
    ap.add_argument("--fault",  default="E:\ZJU\AIOps\Projects\TraDNN\TraDiag/trace_svnd_all\dataset\Data/Service_fault.csv")
    ap.add_argument("--outdir", default="../../dataset/aiops_sv")
    ap.add_argument("--label-scheme", choices=["coarse","fine","both","superfine","both+superfine"], default="both+superfine")
    ap.add_argument("--id-cols", nargs=3, default=["TraceID","SpanId","ParentID"])
    ap.add_argument("--svc-col", default="ServiceName")
    ap.add_argument("--url-col", default="URL")
    ap.add_argument("--status-pref", default="statuscode")
    ap.add_argument("--status-fallback", default="HttpStatusCode")
    ap.add_argument("--start-col", default="StartTimeMs")
    ap.add_argument("--end-col", default="EndTimeMs")
    ap.add_argument("--fault-type-col", default="fault_type")
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--min-trace-size", type=int, default=2)
    ap.add_argument("--drop-orphan-trace", action="store_true", default=True)
    ap.add_argument("--require-single-root", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=2025)
    return ap.parse_args()

def main():
    args=parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed)

    t_col,s_col,p_col=args.id_cols
    cols=dict(trace=t_col,span=s_col,parent=p_col,
              svc=args.svc_col,url=args.url_col,
              status_pref=args.status_pref,status_fb=args.status_fallback,
              start=args.start_col,end=args.end_col)

    normal=pd.read_csv(args.normal)
    fault =pd.read_csv(args.fault)

    # 预构建 superfine 标签：只基于故障行
    ft_series = fault.get(args.fault_type_col, pd.Series([None]*len(fault)))
    ft_norm = ft_series.apply(lambda x: str(x).strip().lower() if isinstance(x, str) else None)
    valid_fault_types = [v for v in ft_norm if v and (map_coarse(v) in (1,2))]
    cnt = Counter(valid_fault_types)
    superfine_names = [k for k,_ in sorted(cnt.items(), key=lambda x:(-x[1], x[0]))]
    superfine_map = {k:i for i,k in enumerate(superfine_names)}

    api_vocab, status_vocab = {}, {}
    drop_stats = Counter()

    # 正常样本（固定 coarse=0）
    rec_normal = build_records(
        normal, cols, api_vocab, status_vocab, label_scheme=args.label_scheme,
        fault_type_series=None, superfine_map=superfine_map, fixed_coarse=0,
        min_trace_size=args.min_trace_size,
        drop_orphan_trace=args.drop_orphan_trace, require_single_root=args.require_single_root,
        drop_stats=drop_stats
    )
    # 故障样本
    rec_fault  = build_records(
        fault, cols, api_vocab, status_vocab, label_scheme=args.label_scheme,
        fault_type_series=ft_norm, superfine_map=superfine_map, fixed_coarse=None,
        min_trace_size=args.min_trace_size,
        drop_orphan_trace=args.drop_orphan_trace, require_single_root=args.require_single_root,
        drop_stats=drop_stats
    )

    traces = rec_normal + rec_fault
    random.shuffle(traces)
    n=len(traces); n_tr=int(n*args.train_ratio); n_v=int(n*args.val_ratio)
    train=traces[:n_tr]; val=traces[n_tr:n_tr+n_v]; test=traces[n_tr+n_v:]

    # 写 JSONL & CSV
    dump_jsonl(os.path.join(args.outdir,"train.jsonl"), train)
    dump_jsonl(os.path.join(args.outdir,"val.jsonl"),   val)
    dump_jsonl(os.path.join(args.outdir,"test.jsonl"),  test)
    flatten_to_csv(os.path.join(args.outdir,"traces_flat_train.csv"), train)
    flatten_to_csv(os.path.join(args.outdir,"traces_flat_val.csv"),   val)
    flatten_to_csv(os.path.join(args.outdir,"traces_flat_test.csv"),  test)

    # 统计分布
    def count(items, key):
        c=Counter([r.get(key) for r in items if r.get(key) is not None and r.get(key)>=0])
        return dict(sorted(c.items()))
    coarse_all=count(traces,"coarse_label")
    fine_all  =count(traces,"fine_label")
    sfn_all   =count(traces,"superfine_label")

    # 写 vocab.json
    with open(os.path.join(args.outdir,"vocab.json"),"w",encoding="utf-8") as f:
        json.dump({
            "api_vocab_size": len(api_vocab),
            "status_vocab_size": len(status_vocab),
            "coarse_label_map": {"normal":0,"structural":1,"latency":2},
            "fine_label_map": {name:i for i,name in enumerate(FINE_LABELS)},  # 仅当需要 6 类时使用
            "fine_groups": {k: sorted(list(v)) for k,v in FINE_GROUPS.items()},
            "structural_types": sorted(list(STRUCTURAL_TYPES)),
            "latency_types": sorted(list(LATENCY_TYPES)),
            "superfine_label_map": superfine_map,
            "superfine_classes": superfine_names,
            "label_scheme": args.label_scheme
        }, f, ensure_ascii=False, indent=2)

    # 写 brief.txt（含丢弃原因）
    with open(os.path.join(args.outdir,"brief.txt"),"w",encoding="utf-8") as f:
        f.write(f"Total traces: {n}\n")
        f.write(f"Train/Val/Test: {len(train)}/{len(val)}/{len(test)}\n")
        f.write(f"Coarse dist: {coarse_all}\n")
        f.write(f"Fine   dist: {fine_all}\n")
        f.write(f"Superfine (fault_type) dist: {sfn_all}\n")
        f.write(f"API vocab: {len(api_vocab)} | Status vocab: {len(status_vocab)}\n")
        if superfine_names:
            f.write("Superfine classes (index:name):\n")
            for k,v in sorted(superfine_map.items(), key=lambda x:x[1]):
                f.write(f"  {v}: {k}\n")
        if drop_stats:
            f.write("\n[Drop stats]\n")
            for k,v in drop_stats.items():
                f.write(f" {k}: {v}\n")

    print(f"[OK] wrote dataset to {args.outdir}")

if __name__ == "__main__":
    main()
