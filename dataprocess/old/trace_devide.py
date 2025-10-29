#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trace_split_4classes.py

功能：
- 依据 groundtruth.jsonl（按天、实例、时间窗）将某一天的 Trace 拆为四类：
  normal / node / service / pod
- 输入优先使用 Data/YYYY-MM-DD.csv（你现成的日级 CSV）；
  若不存在，则回落到 aiops25/{day}/{day}/trace-parquet/*.parquet，
  解析 tags/process/references 时严格调用你已有的
  ndarray2dict / pick_tag / parent_from_ref（不做其它改动）。

输出：
- {out_root}/{YYYY-MM-DD}/{normal, node, service, pod}/{YYYY-MM-DD}_spans.csv
- 列与原 CSV 保持一致：
  TraceID,SpanId,ParentID,NodeName,ServiceName,PodName,URL,
  HttpStatusCode,StatusCode,SpanKind,StartTimeMs,EndTimeMs,
  Normalized_StartTime,Normalized_EndTime
"""

import argparse, json, ast, re, os
import pathlib as pl
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Iterable, Set

import numpy as np
import pandas as pd

# === 用你原来的解析函数 ===
from dataprocess import ndarray2dict, pick_tag, parent_from_ref  # ← 直接复用你的方法  :contentReference[oaicite:1]{index=1}


# ---------------- 工具函数（与解析无关，最小新增） ----------------
def canon(x: str) -> str:
    return re.sub(r"[^a-z0-9]+","_", str(x).lower()).strip("_") if x is not None else ""

def iso_to_ms(s: str) -> int:
    if not s: return 0
    s = str(s).strip()
    if s.endswith("Z"):
        dt = datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return int(dt.timestamp()*1000)
    try:
        return int(datetime.fromisoformat(s).timestamp()*1000)
    except Exception:
        s2 = s.replace("Z","")
        try: return int(datetime.fromisoformat(s2).timestamp()*1000)
        except Exception: return 0

def day_window(day_str: str) -> Tuple[int,int]:
    d0 = datetime.strptime(day_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    d1 = d0 + timedelta(days=1) - timedelta(milliseconds=1)
    return int(d0.timestamp()*1000), int(d1.timestamp()*1000)

def parse_instances_any(raw) -> List[str]:
    """支持 'a' / ['a'] / ['a','b'] / '[a,b]'；非法→[]"""
    if raw is None: return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    s = str(raw).strip()
    if not s: return []
    if s.startswith("[") and s.endswith("]"):
        val = None
        for parser in (json.loads, ast.literal_eval):
            try:
                val = parser(s); break
            except Exception:
                val = None
        if isinstance(val, list):
            return [str(x).strip() for x in val if str(x).strip()]
        # 兜底拆分
        ss = [t.strip().strip('"').strip("'") for t in s[1:-1].split(",")]
        return [x for x in ss if x]
    return [s]

def to_ms(x) -> float:
    """统一转毫秒（float 保精度）：>1e14 ns → ms；>1e11 ms → ms；否则 s → ms"""
    try:
        xi = int(x)
    except Exception:
        try:
            xi = int(float(x))
        except Exception:
            return 0.0
    if xi > 10**14:   # ns
        return xi / 1_000_000.0
    if xi > 10**11:   # ms
        return float(xi)
    return float(xi) * 1000.0  # s → ms

NEED_COLS = [
    "TraceID","SpanId","ParentID",
    "NodeName","ServiceName","PodName","URL",
    "HttpStatusCode","StatusCode","SpanKind",
    "StartTimeMs","EndTimeMs",
    "Normalized_StartTime","Normalized_EndTime"
]


# ---------------- groundtruth：当天 + 多实例 + 窗口 ----------------
def load_gt_for_day(gt_path: pl.Path, day: str, left_ms=60000, right_ms=60000) -> pd.DataFrame:
    lo_day, hi_day = day_window(day)
    items = []
    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            level = str(obj.get("instance_type") or obj.get("type") or "").strip().lower()
            if level not in ("node","service","pod"):
                continue
            insts = parse_instances_any(obj.get("instance"))
            if not insts:
                continue
            st = iso_to_ms(obj.get("start_time")); ed = iso_to_ms(obj.get("end_time"))
            if st<=0 or ed<=0:
                continue
            # 与当天有交集
            if ed < lo_day or st > hi_day:
                continue
            items.append({
                "level": level,
                "instances": insts,
                "instances_canon": [canon(x) for x in insts],
                "fault_type": str(obj.get("fault_type") or obj.get("fault") or "").strip(),
                "start_ms": st - left_ms, "end_ms": ed + right_ms
            })
    return pd.DataFrame(items)

# ---------------- 从日级 CSV 拆分（推荐路径） ----------------
def split_from_day_csv(day_csv: pl.Path, gt_day: pd.DataFrame, out_dir: pl.Path, day: str, chunksize=2_000_000):
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("normal","node","service","pod"):
        (out_dir/sub).mkdir(parents=True, exist_ok=True)

    # 构建 实例→时间窗 倒排索引
    idx: Dict[str, Dict[str, List[Tuple[int,int]]]] = {"node":{}, "service":{}, "pod":{}}
    for _, g in gt_day.iterrows():
        for inst_c in g["instances_canon"]:
            idx[g["level"]].setdefault(inst_c, []).append((g["start_ms"], g["end_ms"]))

    # —— 第一遍：只收集命中 trace 集合 + 全部 trace ——
    hit_node, hit_svc, hit_pod = set(), set(), set()
    all_traces: Set[str] = set()

    def collect_hits_chunk(chunk: pd.DataFrame):
        nonlocal hit_node, hit_svc, hit_pod, all_traces
        # 可能已经是字符串了，防止对象列
        chunk["TraceID"] = chunk["TraceID"].astype(str)
        all_traces.update(chunk["TraceID"].unique().tolist())

        # 规范名（向量化）
        chunk["NodeName_c"]    = chunk["NodeName"].astype(str).str.lower().str.replace(r"[^a-z0-9]+","_", regex=True).str.strip("_")
        chunk["ServiceName_c"] = chunk["ServiceName"].astype(str).str.lower().str.replace(r"[^a-z0-9]+","_", regex=True).str.strip("_")
        chunk["PodName_c"]     = chunk["PodName"].astype(str).str.lower().str.replace(r"[^a-z0-9]+","_", regex=True).str.strip("_")

        # 三层依次命中
        def hit(level: str, col: str) -> Tuple[Set[str], float, float]:
            """
            返回 (命中TraceID集合, 完全覆盖占比, 所有重合占比)
            占比 = 完全覆盖span数 / 所有重合span数  （所有重合>=完全覆盖）
            """
            res = set()
            if not idx[level]:
                return res, 0.0, 0.0

            sub = chunk[chunk[col].isin(idx[level].keys())]
            if sub.empty:
                return res, 0.0, 0.0

            # 用于占比统计
            full_covered = 0
            any_overlap = 0

            for inst, wins in idx[level].items():
                ss = sub[sub[col] == inst]
                if ss.empty:
                    continue
                st = ss["StartTimeMs"].values
                ed = ss["EndTimeMs"].values
                tids = ss["TraceID"].astype(str).values

                for lo, hi in wins:
                    # 所有重合
                    overlap_mask = (np.maximum(st, float(lo)) <= np.minimum(ed, float(hi)))
                    # 完全覆盖：span 开始 >= 窗口开始 且 结束 <= 窗口结束
                    full_mask = (st >= float(lo)) & (ed <= float(hi))

                    cnt_overlap = overlap_mask.sum()
                    cnt_full = full_mask.sum()

                    any_overlap += cnt_overlap
                    full_covered += cnt_full

                    if overlap_mask.any():
                        res.update(tids[full_mask])

                # 计算占比
                ratio_full = full_covered / any_overlap if any_overlap else 0.0
                ratio_any = 1.0  # 总是 1，方便外部统一打印
                return res, ratio_full, ratio_any

        hit_node, ratio_full_node, _ = hit("node", "NodeName_c")
        hit_svc, ratio_full_svc, _ = hit("service", "ServiceName_c")
        hit_pod, ratio_full_pod, _ = hit("pod", "PodName_c")

        print(f"[统计] 完全覆盖占比 → node:{ratio_full_node:.2%}  service:{ratio_full_svc:.2%}  pod:{ratio_full_pod:.2%}")

        # hit_node |= hit("node","NodeName_c")
        # hit_svc  |= hit("service","ServiceName_c")
        # hit_pod  |= hit("pod","PodName_c")

    # 第一遍
    for chunk in pd.read_csv(day_csv, chunksize=chunksize):
        # 只要必要列，防止意外列影响
        miss = set(NEED_COLS) - set(chunk.columns)
        if miss:
            raise RuntimeError(f"{day_csv} 缺少列：{miss}")
        collect_hits_chunk(chunk)

    # ---------- 2. 重叠集合 ----------
    pod_svc = hit_pod & hit_svc
    pod_node = hit_pod & hit_node
    svc_node = hit_svc & hit_node
    pod_svc_node = hit_pod & hit_svc & hit_node

    # 所有“两层及以上”重叠
    any_overlap = pod_svc | pod_node | svc_node | pod_svc_node

    # ---------- 3. 只中一种的“纯净”集合 ----------
    pure_pod = hit_pod - any_overlap
    pure_svc = hit_svc - any_overlap
    pure_node = hit_node - any_overlap
    pure_normal = all_traces - (hit_pod | hit_svc | hit_node)  # 完全没命中

    # ---------- 4. 写盘路径 ----------

    # 纯净目录
    pure_dirs = {
        "pod": out_dir / "pod",
        "service": out_dir / "service",
        "node": out_dir / "node",
        "normal": out_dir / "normal",
    }
    for d in pure_dirs.values():
        d.mkdir(exist_ok=True)

    # 重叠目录
    overlap_dir = out_dir / "overlap"
    overlap_dir.mkdir(exist_ok=True)

    # 每个组合一个文件（空组合不创建）
    overlap_files = {
        "pod+service": overlap_dir / "pod+service_spans.csv",
        "pod+node": overlap_dir / "pod+node_spans.csv",
        "service+node": overlap_dir / "service+node_spans.csv",
        "pod+service+node": overlap_dir / "pod+service+node_spans.csv",
    }

    # 写文件控制
    wrote_header = {**{k: False for k in pure_dirs},
                    **{k: False for k in overlap_files}}

    # ---------- 5. 第二遍写 chunk ----------
    def write_chunk(chunk: pd.DataFrame):
        part = chunk[NEED_COLS].copy()
        part = part.sort_values(["TraceID", "StartTimeMs", "EndTimeMs", "SpanId"])
        tids = part["TraceID"].astype(str)

        # 逐 span 分类
        cats = []
        for tid in tids:
            if tid in pure_normal: cats.append("normal"); continue
            if tid in pure_pod:    cats.append("pod"); continue
            if tid in pure_svc:    cats.append("service"); continue
            if tid in pure_node:   cats.append("node"); continue
            # 以下都是重叠
            if tid in pod_svc_node: cats.append("pod+service+node"); continue
            if tid in pod_svc:      cats.append("pod+service"); continue
            if tid in pod_node:     cats.append("pod+node"); continue
            if tid in svc_node:     cats.append("service+node"); continue
            # 理论上不会走到这里
            cats.append("normal")

        part["_cls"] = cats

        # 分组写盘
        for cls_name, grp in part.groupby("_cls", sort=False):
            if cls_name in pure_dirs:
                path = pure_dirs[cls_name] / f"{day}_spans.csv"
            else:
                path = overlap_files[cls_name]
            grp.drop(columns=["_cls"], inplace=True)
            grp.to_csv(path, mode=("a" if wrote_header[cls_name] else "w"),
                       index=False, header=(not wrote_header[cls_name]),
                       float_format="%.6f", encoding="utf-8")
            wrote_header[cls_name] = True

    # 开始第二遍扫描
    for chunk in pd.read_csv(day_csv, chunksize=chunksize):
        write_chunk(chunk)

    # ---------- 6. 打印纯净统计 ----------
    print(f"[纯净统计] {day} → "
          f"pod:{len(pure_pod)}  service:{len(pure_svc)}  "
          f"node:{len(pure_node)}  normal:{len(pure_normal)}  "
          f"重叠:{len(any_overlap)}")

# ---------------- 新增：目录读取 ----------------
def collect_csv_files(csv_source: pl.Path, recursive: bool = False) -> List[pl.Path]:
    """返回目录下所有 csv 文件路径（可递归）"""
    pattern = "**/*.csv" if recursive else "*.csv"
    files = sorted(csv_source.glob(pattern))
    if not files:
        raise FileNotFoundError(f"{csv_source} 未找到任何 csv 文件")
    return files

# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--day-csv", type=pl.Path, help="单个日级 CSV 文件")
    group.add_argument("--csv-dir", type=pl.Path, help="包含若干 CSV 的目录（与 --recursive 配合）")

    ap.add_argument("--groundtruth", default="../Data/groundtruth.txt", help="groundtruth.jsonl")
    ap.add_argument("--out-root", default="./DataSplit", help="输出根目录")
    ap.add_argument("--left-buffer-ms", type=int, default=60000)
    ap.add_argument("--right-buffer-ms", type=int, default=60000)
    ap.add_argument("--recursive", "-r", action="store_true", help="递归子目录")
    args = ap.parse_args()

    out_root = pl.Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # 1. 确定要处理的 csv 文件列表
    if args.day_csv:
        csv_files = [args.day_csv]
    else:
        csv_files = collect_csv_files(args.csv_dir, args.recursive)

    # 2. 逐个文件处理
    for csv_file in csv_files:
        day = csv_file.stem  # 用文件名（不含扩展名）当“日期”
        print(f"\n=== 处理 {csv_file} （识别为 {day}）===")
        gt_day = load_gt_for_day(pl.Path(args.groundtruth), day,
                                 args.left_buffer_ms, args.right_buffer_ms)
        out_dir = out_root / day
        out_dir.mkdir(parents=True, exist_ok=True)

        # 直接调用你原有的核心拆分函数
        split_from_day_csv(csv_file, gt_day, out_dir, day, chunksize=2_000_000)

    print("\n[OK] 全部处理完成！")

if __name__ == "__main__":
    main()
