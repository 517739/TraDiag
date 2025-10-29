#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_pod_fault_dataset.py

用法示例：
python build_pod_fault_dataset.py \
  --data-root ./Data \
  --groundtruth ./groundtruth.jsonl \
  --out ./aiops25_pod_only \
  --left-buffer-ms 60000 \
  --right-buffer-ms 60000
"""
import argparse, json, os, re
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# ---------- 解析 groundtruth 时间（ISO8601 → 毫秒） ----------
def iso_to_ms(s: str) -> int:
    if not s:
        return 0
    s = str(s).strip()
    # 常见形式：2025-06-05T16:10:02Z
    if s.endswith("Z"):
        dt = datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")
        return int(dt.timestamp() * 1000)
    # 带偏移：2025-06-05T16:10:02+00:00
    try:
        return int(datetime.fromisoformat(s).timestamp() * 1000)
    except Exception:
        # 兜底：删尾巴重试
        s2 = s.replace("Z", "")
        try:
            return int(datetime.fromisoformat(s2).timestamp() * 1000)
        except Exception:
            return 0

# ---------- 规范化 fault_type / coarse/fine/superfine ----------
def canon(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")

COARSE_MAP = {
    # → structural
    "dns_error": "structural_anomaly",
    "io_fault": "structural_anomaly",
    "jvm_exception": "structural_anomaly",
    "pod_failure": "structural_anomaly",
    "pod_kill": "structural_anomaly",
    # → latency
    "cpu_stress": "latency_anomaly",
    "memory_stress": "latency_anomaly",
    "jvm_cpu": "latency_anomaly",
    "jvm_gc": "latency_anomaly",
    "jvm_latency": "latency_anomaly",
}
FINE_DEFAULT_STRUCT = "exception"
FINE_DEFAULT_LAT    = "network_delay"  # 只是占位，实际我们直接用 fault_type 作为 fine

def map_labels(fault_type_raw: str, instance: str):
    ft = canon(fault_type_raw)
    coarse = COARSE_MAP.get(ft, "latency_anomaly")
    # fine：直接用规范化 fault_type（区分度更高）
    fine = ft if ft else (FINE_DEFAULT_STRUCT if coarse=="structural_anomaly" else FINE_DEFAULT_LAT)
    # superfine：细化到 “fault_type:pod_name”
    inst = canon(instance)
    superfine = f"{fine}:{inst}" if inst else fine
    return coarse, fine, superfine

# ---------- 主流程 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="../Data", help="按天CSV所在的根目录（例如 Data/2025-06-06.csv 或 Data/2025-06-06/2025-06-06.csv 已经合并成上一层了的话，就传上层）")
    ap.add_argument("--groundtruth", default="../Data/groundtruth.jsonl", help="groundtruth.jsonl 路径")
    ap.add_argument("--out", default="./Data/", help="输出目录（写 manifest.csv / flat_spans.csv）")
    ap.add_argument("--left-buffer-ms", type=int, default=0, help="时间窗口左缓冲（毫秒）")
    ap.add_argument("--right-buffer-ms", type=int, default=0, help="时间窗口右缓冲（毫秒）")
    ap.add_argument("--csv_pattern", default=r"^\d{4}-\d{2}-\d{2}\.csv$", help="匹配 Data 下天粒度CSV文件名的正则")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # 1) 读取 groundtruth（只保留 pod 级）
    gts = []
    with open(args.groundtruth, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            inst_type = str(obj.get("instance_type") or obj.get("type") or "").strip().lower()
            if inst_type != "pod":
                continue

            raw_inst = obj.get("instance")
            if isinstance(raw_inst, str) and raw_inst.strip():
                # 纯字符串
                inst = raw_inst.strip()
            elif isinstance(raw_inst, list) and len(raw_inst) == 1:
                # 单元素列表
                inst = str(raw_inst[0]).strip()
            else:
                # 多元素列表或其他非法格式直接跳过
                continue
            ft   = str(obj.get("fault_type") or obj.get("fault") or "").strip()
            st   = iso_to_ms(obj.get("start_time"))
            ed   = iso_to_ms(obj.get("end_time"))
            if not inst or st<=0 or ed<=0:
                continue
            gts.append({
                "instance": inst,
                "fault_type": ft,
                "start_ms": st - args.left_buffer_ms,
                "end_ms":   ed + args.right_buffer_ms,
                "start_orig": st,
                "end_orig": ed,
            })
    if not gts:
        print("[WARN] groundtruth 中没有 pod 级条目，结束。")
        return
    print(f"共 {len(gts)} 种类型")
    gt_df = pd.DataFrame(gts)
    gt_df["inst_canon"] = gt_df["instance"].map(canon)

    # 2) 扫描 Data 根目录下的天粒度 CSV
    pat = re.compile(args.csv_pattern)
    day_csvs = sorted([p for p in data_root.glob("**/*.csv") if pat.search(p.name)])
    if not day_csvs:
        print(f"[WARN] 在 {data_root} 下未找到天级 CSV（匹配：{args.csv_pattern}）")
        return

    manifest_rows = []
    flat_parts = []

    # 为了提速：把 gt 按 instance 索引
    gt_by_inst = {}
    for i, row in gt_df.iterrows():
        gt_by_inst.setdefault(row["inst_canon"], []).append(row)

    for csv_path in day_csvs:
        try:
            spans = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[WARN] 读取 {csv_path} 失败：{e}")
            continue

        # 必要列检查
        needed = ["TraceID","SpanId","ParentID","PodName","StartTimeMs","EndTimeMs"]
        miss = [c for c in needed if c not in spans.columns]
        if miss:
            print(f"[WARN] {csv_path} 缺少列：{miss}，跳过。")
            continue

        # 预处理：规范化 PodName / Start/End
        spans["PodName_canon"] = spans["PodName"].astype(str).map(canon)
        # 兼容你的 StartTimeMs/EndTimeMs 为 float 的设定
        spans["StartTimeMs"] = pd.to_numeric(spans["StartTimeMs"], errors="coerce")
        spans["EndTimeMs"]   = pd.to_numeric(spans["EndTimeMs"],   errors="coerce")

        # 按 PodName candidate 加速匹配
        for inst_canon, gt_list in gt_by_inst.items():
            # 粗筛：同名 pod 的行
            sub = spans.loc[spans["PodName_canon"] == inst_canon]
            if sub.empty:
                continue
            # 进一步：按每个 gt 窗口筛
            sub_valid = []
            for gt in gt_list:
                lo, hi = gt["start_ms"], gt["end_ms"]
                mask = (sub["StartTimeMs"] >= lo) & (sub["StartTimeMs"] <= hi)
                ss = sub.loc[mask]
                if ss.empty:
                    continue
                # 命中的 TraceID
                tids = ss["TraceID"].astype(str).unique().tolist()
                if not tids:
                    continue
                # 标注每个 TraceID
                coarse, fine, superfine = map_labels(gt["fault_type"], gt["instance"])
                for tid in tids:
                    manifest_rows.append({
                        "TraceID": tid,
                        "label_coarse": coarse,
                        "label_fine": fine,
                        "label_superfine": superfine,
                        "fault_type": fine,
                        "instance": gt["instance"],
                        "instance_type": "pod",
                        "start_time_ms": gt["start_orig"],
                        "end_time_ms": gt["end_orig"],
                    })
                # 收集被命中的 span（后面合并输出）
                # 注意：按需求，flat_spans.csv 去掉 StartTimeMs/EndTimeMs
                keep_cols = [c for c in spans.columns if c not in ("StartTimeMs","EndTimeMs","PodName_canon")]
                flat_parts.append(spans.loc[spans["TraceID"].astype(str).isin(tids), keep_cols])

    if not manifest_rows:
        print("[WARN] 未匹配到任何 Pod 级故障 Trace。")
        return

    # 3) 汇总 & 去重
    manifest = pd.DataFrame(manifest_rows).drop_duplicates(subset=["TraceID"])
    # 有些 Trace 可能被多个窗口命中：保留最早/或第一条（可按需改策略）
    manifest = manifest.drop_duplicates(subset=["TraceID"], keep="first").reset_index(drop=True)

    # 4) 合并 flat spans（只保留命中的 Trace）
    flat_spans = pd.concat(flat_parts, axis=0, ignore_index=True) if flat_parts else pd.DataFrame()
    if not flat_spans.empty:
        # 只保留 manifest 里的 TraceID
        tids = set(manifest["TraceID"].astype(str).tolist())
        flat_spans = flat_spans.loc[flat_spans["TraceID"].astype(str).isin(tids)].copy()
        # 最终列顺序（尽量保持你当前列；已移除 StartTimeMs/EndTimeMs）
        pref = ["TraceID","SpanId","ParentID",
                "NodeName","ServiceName","PodName","URL",
                "HttpStatusCode","StatusCode","SpanKind",
                "Normalized_StartTime","Normalized_EndTime"]
        exist = [c for c in pref if c in flat_spans.columns]
        others = [c for c in flat_spans.columns if c not in exist]
        flat_spans = flat_spans[exist + others]

    # 5) 写盘
    manifest_out = Path(out/"manifest.csv"); manifest.to_csv(manifest_out, index=False, encoding="utf-8")
    flat_out = Path(out/"flat_spans.csv")
    if not flat_spans.empty:
        flat_spans.to_csv(flat_out, index=False, encoding="utf-8")
    print(f"[OK] manifest: {manifest_out}")
    print(f"[OK] flat_spans: {flat_out if not flat_spans.empty else '(no spans matched)'}")

if __name__ == "__main__":
    main()
