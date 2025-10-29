#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
以 Trace 为单位打故障标签，并剔除同时命中多种故障类型的 Trace。
用法：
    python label_trace_clean.py spans.csv groundtruth.jsonl
"""
import os
import glob
import pandas as pd
import argparse, csv, json, sys
from datetime import datetime, timezone
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict

TIME_COL = "StartTimeMs"
TIME_PARSER = float
UTC = timezone.utc

# 统计计数器
CNT_STR, CNT_LIST, CNT_EMPTY = 0, 0, 0


def normalize_instance(raw) -> List[str]:
    global CNT_STR, CNT_LIST, CNT_EMPTY
    if isinstance(raw, str):
        if raw.strip() == "":
            CNT_EMPTY += 1
            return []
        CNT_STR += 1
        return [raw.strip().lower()]
    if isinstance(raw, list):
        if not raw:
            CNT_EMPTY += 1
            return []
        CNT_LIST += 1
        return [s.strip().lower() for s in raw if isinstance(s, str)]
    CNT_EMPTY += 1
    return []


def parse_ground_truth(path: str) -> List[Dict]:
    faults = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("instance_type") != "service":
                continue
            rec["service_list"] = normalize_instance(rec.get("instance"))
            rec["start_ts"] = datetime.fromisoformat(
                rec["start_time"].replace("Z", "+00:00")
            ).timestamp()
            rec["end_ts"] = datetime.fromisoformat(
                rec["end_time"].replace("Z", "+00:00")
            ).timestamp()
            faults.append(rec)
    print(f"[stats] instance format -> str: {CNT_STR}, list: {CNT_LIST}, empty: {CNT_EMPTY}", file=sys.stderr)
    return faults


def build_trace_fault_candidates(csv_path: str, faults: List[Dict]) -> Dict[str, Set[Tuple[str, str]]]:
    """
    返回 TraceID -> 命中(fault_category, fault_type)集合
    """
    trace_cands: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            svc = r["ServiceName"].lower()
            try:
                span_start = TIME_PARSER(r[TIME_COL]) / 1000.0
                span_end = TIME_PARSER(r["EndTimeMs"]) / 1000.0
            except Exception:
                continue
            for frec in faults:
                if svc not in frec["service_list"]:
                    continue
                if span_start >= frec["start_ts"] and span_end <= frec["end_ts"]:
                    tid = r["TraceID"]
                    trace_cands[tid].add((frec["fault_category"], frec["fault_type"]))
    return trace_cands


def process(csv_path: str, gt_path: str, out_path: Optional[str], verbose: bool = True):
    faults = parse_ground_truth(gt_path)
    print(f"Loaded {len(faults)} service-level fault records.", file=sys.stderr)
    trace_cands = build_trace_fault_candidates(csv_path, faults)

    # 决定每个 Trace 最终标签
    trace_label: Dict[str, Optional[Tuple[str, str]]] = {}
    removed_tids = []
    total_traces = len(trace_cands)  # ← 新增

    for tid, cset in trace_cands.items():
        if len(cset) >= 2:
            trace_label[tid] = None
            removed_tids.append(tid)
        elif len(cset) == 1:
            trace_label[tid] = next(iter(cset))
        else:
            trace_label[tid] = None

    # ← 新增统计
    removed_cnt = len(removed_tids)
    ratio = removed_cnt / total_traces if total_traces else 0.0
    print(f"[clean] 多故障 Trace 被剔除：{removed_cnt} / {total_traces} "
          f"({ratio:.2%})", file=sys.stderr)

    # 第二遍扫描写回
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames + ["fault_category", "fault_type"]
        for r in reader:
            tid = r["TraceID"]
            lbl = trace_label.get(tid)
            if lbl:
                r["fault_category"], r["fault_type"] = lbl
                rows.append(r)
            # else:
            #     r["fault_category"] = ""
            #     r["fault_type"] = ""
            # rows.append(r)

    out_path = out_path or csv_path
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # 打印清洗结果
    print(f"Traces with multi-fault (removed): {len(removed_tids)}", file=sys.stderr)
    if verbose and removed_tids:
        print("Removed TraceIDs:", *removed_tids, file=sys.stderr)
    print(f"Done! Clean labeled CSV -> {out_path}", file=sys.stderr)


def combine_service_csvs(root_dir, output_file):
    # 确保根目录存在
    if not os.path.exists(root_dir):
        raise ValueError(f"目录不存在: {root_dir}")

    # 查找所有service子目录中的CSV文件
    csv_pattern = os.path.join(root_dir, '*.csv')
    csv_files = glob.glob(csv_pattern)

    if not csv_files:
        raise ValueError(f"在 {root_dir} 下未找到任何CSV文件")

    print(f"找到 {len(csv_files)} 个CSV文件")

    # 读取并合并所有CSV文件
    dataframes = []
    for file_path in csv_files:
        try:
            # 读取CSV文件
            df = pd.read_csv(file_path)
            dataframes.append(df)
            print(f"已读取: {file_path} ({len(df)} 行)")
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {str(e)}")

    # 合并所有数据框
    if not dataframes:
        raise ValueError("没有成功读取任何CSV文件")

    combined_df = pd.concat(dataframes, ignore_index=True)

    # 保存合并后的数据
    combined_df.to_csv(output_file, index=False)
    print(f"合并完成! 总共 {len(combined_df)} 行数据已保存到: {output_file}")

    return combined_df


if __name__ == "__main__":
    # 执行合并操作
    # combine_service_csvs('../Data/Service', '../Data/combined_services.csv')

    csv_dir = "../Data/Merged/service.csv"
    gtd_dir = "../Data/groundtruth.jsonl"
    output = "../Data/process_dataset1.csv"

    process(csv_dir, gtd_dir, output)
