#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_split_1dir_tracecnt.py
单目录批处理 + 每文件 TraceID 计数
"""
import argparse, json, pathlib as pl, re
from datetime import datetime, timezone, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd

# ---------- 配置 ----------
NEED_COLS = [
    "TraceID", "SpanId", "ParentID",
    "NodeName", "ServiceName", "PodName", "URL",
    "HttpStatusCode", "StatusCode", "SpanKind",
    "StartTimeMs", "EndTimeMs",
    "Normalized_StartTime", "Normalized_EndTime"
]

SUB_DIRS = ["normal", "node", "service", "pod", "overlap"]

# ---------- 工具 ----------
def canon(x):
    return re.sub(r"[^a-z0-9]+", "_", str(x).lower()).strip("_") if x else ""

def iso_to_ms(s):
    if not s: return 0
    s = str(s).strip()
    if s.endswith("Z"):
        dt = datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    return int(datetime.fromisoformat(s).timestamp() * 1000)

def first_instance(raw) -> str:
    if not raw: return ''
    if isinstance(raw, list):
        return str(raw[0]).strip() if raw else ''
    return str(raw).split(';')[0].strip()

# ---------- 读 groundtruth ----------
def load_gt(gt_path: pl.Path, left_ms=60_000, right_ms=60_000):
    """
    输入：groundtruth.csv
    必要列（大小写不敏感）：
        instance_type, instance, start_time, end_time, fault_type
    时间格式支持：ISO-8601 带 Z 或普通格式
    """
    df = pd.read_csv(gt_path)
    # 统一列名小写
    df.columns = [c.lower() for c in df.columns]

    need = {"instance_type", "instance", "start_time", "end_time", "fault_type"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"groundtruth.csv 缺少列：{miss}")

    records = []
    for _, r in df.iterrows():
        level = str(r["instance_type"]).strip().lower()
        if level not in {"node", "service", "pod"}:
            continue
        inst = first_instance(r["instance"])
        if not inst:
            continue
        st, ed = iso_to_ms(r["start_time"]), iso_to_ms(r["end_time"])
        if st <= 0 or ed <= 0:
            continue
        records.append({
            "level": level,
            "instance_canon": canon(inst),
            "fault_type": str(r["fault_type"] or "").strip(),
            "start_ms": st - left_ms,
            "end_ms": ed + right_ms
        })
    return pd.DataFrame(records)

# ---------- 核心 ----------
def split_one_csv(csv_path: pl.Path, gt_df: pd.DataFrame, out_root: pl.Path, global_wrote_header):
    wrote_header = global_wrote_header

    # 1. 给 groundtruth 加序号
    gt_idx = {lvl: {} for lvl in {"node", "service", "pod"}}   # inst -> [(lo, hi, ftype, idx)]
    for idx_global, (_, r) in enumerate(gt_df.iterrows(), 0):
        inst = r["instance_canon"]
        gt_idx[r["level"]].setdefault(inst, []).append(
            (r["start_ms"], r["end_ms"], r["fault_type"], idx_global)
        )

    # 2. 命中收集
    hit_gt = {"node": set(), "service": set(), "pod": set()}
    hit_node, hit_svc, hit_pod = set(), set(), set()
    trace2fault = {}
    trace2inst = {}
    all_traces = set()

    hit_cnt = {"node": 0, "service": 0, "pod": 0}

    def collect(chunk: pd.DataFrame):
        chunk["TraceID"] = chunk["TraceID"].astype(str)
        all_traces.update(chunk["TraceID"].unique())
        chunk["NodeName_c"] = chunk["NodeName"].astype(str).apply(canon)
        chunk["ServiceName_c"] = chunk["ServiceName"].astype(str).apply(canon)
        chunk["PodName_c"] = chunk["PodName"].astype(str).apply(canon)

        def hit(level: str, col: str):
            res = set()
            if not gt_idx[level]:
                return res
            sub = chunk[chunk[col].isin(gt_idx[level])]
            if sub.empty:
                return res
            for inst, wins in gt_idx[level].items():
                ss = sub[sub[col] == inst]
                if ss.empty:
                    continue
                st, ed = ss["StartTimeMs"].values, ss["EndTimeMs"].values
                for lo, hi, ftype, g_id in wins:
                    mask = (np.maximum(st, float(lo)) <= np.minimum(ed, float(hi)))
                    if mask.any():
                        hit_gt[level].add(g_id)      # 记录序号
                        for tid in ss["TraceID"].astype(str).values[mask]:
                            res.add(tid)
                            trace2fault[tid] = ftype
                            if tid not in trace2inst:
                                trace2inst[tid] = inst
            return res

        nonlocal hit_node, hit_svc, hit_pod
        hit_node|= hit("node", "NodeName_c")
        hit_svc |= hit("service", "ServiceName_c")
        hit_pod |= hit("pod", "PodName_c")

    for chunk in pd.read_csv(csv_path, chunksize=2_000_000):
        collect(chunk)

    # 3. 分类
    any_overlap = (hit_node & hit_svc) | (hit_node & hit_pod) | (hit_svc & hit_pod)
    pure_node = hit_node - any_overlap
    pure_svc = hit_svc - any_overlap
    pure_pod = hit_pod - any_overlap
    pure_normal = all_traces - (hit_node | hit_svc | hit_pod)

    def tag(tid: str):
        if tid in pure_normal: return "normal"
        if tid in pure_node: return "node"
        if tid in pure_svc: return "service"
        if tid in pure_pod: return "pod"
        return "overlap"

    # 4. 写盘 + 计数
    # wrote_header = {s: False for s in SUB_DIRS}
    trace_cnt = defaultdict(set)  # sub -> {traceID}

    for chunk in pd.read_csv(csv_path, chunksize=2_000_000):
        part = chunk[NEED_COLS].copy()
        part["fault_type"] = part["TraceID"].map(trace2fault).fillna("")
        part["fault_instance"] = part["TraceID"].map(trace2inst).fillna("")
        part["TraceID"] = part["TraceID"].astype(str)
        part["_sub"] = part["TraceID"].map(tag)
        for sub, grp in part.groupby("_sub", sort=False):
            path = out_root / sub / "ALL_spans.csv"
            grp.drop(columns=["_sub"]).to_csv(
                path, mode="a" if wrote_header[sub] else "w",
                index=False, header=not wrote_header[sub],
                float_format="%.6f", encoding="utf-8"
            )
            wrote_header[sub] = True
            trace_cnt[sub].update(grp["TraceID"].unique())

    # # 5. 打印
    # total = sum(len(s) for s in trace_cnt.values())
    # print(f"[{csv_path.name}] 写入 TraceID 统计："
    #       f"normal={len(trace_cnt['normal'])} "
    #       f"node={len(trace_cnt['node'])} "
    #       f"service={len(trace_cnt['service'])} "
    #       f"pod={len(trace_cnt['pod'])} "
    #       f"overlap={len(trace_cnt['overlap'])} | 总计={total}")

    # 打印时带上命中条数
    total = sum(len(s) for s in trace_cnt.values())
    print(f"[{csv_path.name}] TraceID 统计："
          f"normal={len(trace_cnt['normal'])} "
          f"node={len(trace_cnt['node'])} "
          f"service={len(trace_cnt['service'])} "
          f"pod={len(trace_cnt['pod'])} "
          f"overlap={len(trace_cnt['overlap'])} | "
          f"总计 TraceID={total} 条 | "
          f"GT 注入记录命中 node={len(hit_gt['node'])} "
          f"service={len(hit_gt['service'])} "
          f"pod={len(hit_gt['pod'])} 条")

# ---------- 主入口 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", default='../../dataset/aiops25/processed/phasetwo', type=pl.Path, help="目录（不递归）")
    ap.add_argument("--groundtruth", default='../../dataset/Data/groundtruth.csv', type=pl.Path)
    ap.add_argument("--out-root", default="../../dataset/SplitTrace1729", type=pl.Path)
    ap.add_argument("--left-ms", type=int, default=0)
    ap.add_argument("--right-ms", type=int, default=0)
    args = ap.parse_args()

    gt_df = load_gt(args.groundtruth, args.left_ms, args.right_ms)
    args.out_root.mkdir(parents=True, exist_ok=True)
    for d in SUB_DIRS:
        (args.out_root / d).mkdir(exist_ok=True)

    csv_files = sorted(args.csv_dir.glob("*.csv"))
    if not csv_files:
        print("目录下未找到任何 CSV 文件"); return

    global_wrote_header = {s: False for s in SUB_DIRS}  # 全局
    for csv_file in csv_files:
        split_one_csv(csv_file, gt_df, args.out_root, global_wrote_header)

    print("\n[OK] 全部处理完成！")

if __name__ == "__main__":
    main()