#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 基于你现有的简易脚本做的“最小改动”版：保留原有函数与取数方式，仅将处理范围扩展到 aiops25 下所有日期目录，
# 并按天输出到 Data/YYYY-MM-DD.csv；补充了 0~1 归一化列名对齐与拓扑序列号。
# 来源：你提供的 dataprocess.py（在此基础上轻量改造） :contentReference[oaicite:0]{index=0}

import argparse, json, pathlib, pandas as pd, numpy as np
from typing import List

# ---------- 工具函数（尽量保持原样） ----------
def ndarray2dict(obj) -> dict:
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except Exception:
            return {}
    if isinstance(obj, np.ndarray):
        return {d["key"]: d["value"] for d in obj if isinstance(d, dict) and "key" in d}
    if isinstance(obj, dict):
        return obj
    return {}

def pick_tag(obj, *keys):
    d = ndarray2dict(obj)
    for k in keys:
        if isinstance(d, dict) and d.get(k):
            return d[k]
    return ""

def parent_from_ref(obj) -> str:
    if isinstance(obj, str):
        data = json.loads(obj) if obj.startswith('[') else []
    elif isinstance(obj, np.ndarray):
        data = list(obj)
    elif isinstance(obj, list):
        data = obj
    else:
        data = []
    for ref in data:
        if isinstance(ref, dict) and str(ref.get("refType", "")).upper() in ("CHILD_OF", "CHILDOF", "FOLLOWS_FROM", "FOLLOWSFROM"):
            return str(ref.get("spanID") or ref.get("spanId") or "")
    return "-1"

def to_ms(x) -> float:
    """把各种时间单位统一转成毫秒整数：
       - >1e14 视为微秒 → ms
       - >1e11 视为毫秒 → ms
    """
    try:
        xi = int(x)
    except Exception:
        return 0.0
    if xi > 10 ** 14:  # μs
        return xi / 1_000.0
    if xi > 10 ** 11:  # ms
        return float(xi)

# ---------- 单个 parquet 读取并转行 ----------
def process_parquet_file(pf: pathlib.Path, rows: List[dict]):
    df = pd.read_parquet(pf)
    # 兼容不同导出字段名
    col_trace = "traceID" if "traceID" in df.columns else ("traceId" if "traceId" in df.columns else "trace_id")
    col_span  = "spanID"  if "spanID"  in df.columns else ("spanId"  if "spanId"  in df.columns else "span_id")
    col_oper  = "operationName" if "operationName" in df.columns else ("name" if "name" in df.columns else None)
    col_start = "startTime" if "startTime" in df.columns else ("start_time" if "start_time" in df.columns else None)
    col_dur   = "duration" if "duration" in df.columns else ("durationMs" if "durationMs" in df.columns else None)

    # process 列里一般包含 tags/servicename 等
    for _, r in df.iterrows():
        proc = ndarray2dict(r.get("process", {}))
        proc_tags = proc.get("tags", {})
        span_tags = ndarray2dict(r.get("tags", {}))
        status_code_str = pick_tag(span_tags, "status.code", "http.status_code")  # 优先 OTel，再兜底 HTTP
        try:
            status_code = int(status_code_str) if status_code_str not in ("", None) else 0
        except Exception:
            status_code = 0
        span_kind = pick_tag(span_tags, "span.kind")  # client/server/internal 等

        trace_id = str(r.get(col_trace, ""))
        span_id  = str(r.get(col_span,  ""))
        parent   = parent_from_ref(r.get("references", []))

        # 三层归属（按你的口径）
        node_name    = pick_tag(proc_tags, "node_name", "nodeName", "k8s.node.name", "host.name")
        service_name = proc.get("serviceName", "") or pick_tag(proc_tags, "service.name", "serviceName")
        pod_name     = pick_tag(proc_tags, "name", "podName", "k8s.pod.name", "pod.name")

        # URL / 操作名兜底
        url = str(r.get(col_oper, "")) if col_oper else ""
        if not url:
            url = pick_tag(ndarray2dict(r.get("tags", {})), "http.url", "url", "http.target", "rpc.method")

        http_code_str = pick_tag(
            span_tags,
            "http.status_code",  # 常见键
            "http.response.status_code"  # 兜底（有些导出用这个）
        )
        try:
            http_status_code = int(http_code_str) if http_code_str not in ("", None) else -1
        except Exception:
            http_status_code = -1  # 非 HTTP span 或解析失败，置为 -1

        # 开始/结束时间（毫秒）
        st = to_ms(r.get(col_start, 0)) if col_start else 0.0
        du = float(r.get(col_dur, 0)) if col_dur else 0.0
        ed = st + (du / 1000.0) # duration 微秒转毫秒

        rows.append({
            "TraceID"  : trace_id,
            "SpanId"   : span_id,
            "ParentID" : parent,
            "NodeName" : node_name,
            "ServiceName": service_name,
            "PodName"  : pod_name,
            "URL"      : url,
            "StatusCode": status_code,  # int, 0 表正常，非 0 视为错误
            "HttpStatusCode": http_status_code,
            "SpanKind": span_kind,
            "StartTimeMs": st,
            "EndTimeMs"  : ed,
        })

# ---------- 按天归一化与写盘 ----------
def finalize_and_write(rows: List[dict], out_csv: pathlib.Path):
    if not rows:
        print(f"[WARN] 无数据可写：{out_csv}")
        return
    spans = pd.DataFrame(rows)

    # 以 Trace 为单位做 0~1 归一化（与你现有管线一致）
    trace_range = spans.groupby("TraceID").agg(
        start_min=("StartTimeMs", "min"),
        end_max  =("EndTimeMs",   "max")
    )
    spans = spans.merge(trace_range, left_on="TraceID", right_index=True)
    dur = (spans["end_max"] - spans["start_min"]).clip(lower=1e-6)
    spans["Normalized_StartTime"] = (spans["StartTimeMs"] - spans["start_min"]) / dur
    spans["Normalized_EndTime"]   = (spans["EndTimeMs"]   - spans["start_min"]) / dur
    spans = spans.drop(columns=["start_min", "end_max"])

    # Trace 内按开始/结束/SpanId 稳定排序并生成拓扑序号
    spans = spans.sort_values(["TraceID", "StartTimeMs", "EndTimeMs", "SpanId"])
    # spans["Normalized_tree_span_ids"] = spans.groupby("TraceID").cumcount()

    # 输出列顺序（便于后续 groundtruth 对齐与特征构建）
    cols = [
        "TraceID","SpanId","ParentID",
        "NodeName","ServiceName","PodName","URL",
        "HttpStatusCode", "StatusCode", "SpanKind",
        "StartTimeMs", "EndTimeMs",
        "Normalized_StartTime", "Normalized_EndTime"
    ]
    spans = spans[cols]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    spans.to_csv(out_csv, index=False, float_format="%.8f")
    print(f"→ 写完 {out_csv}  共 {len(spans)} 行")


# ---------- 扫描 aiops25 目录下所有日期并处理 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../../dataset/aiops25/row", help="aiops25 根目录")
    ap.add_argument("--out-dir", default="../dataset/aiops25/processed", help="按天输出目录（Data/YYYY-MM-DD.csv）")
    ap.add_argument("--pattern", default="2025-??-??", help="日期目录匹配模式（默认 2025-??-??）")
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    out_root = pathlib.Path(args.out_dir)
    # 一层日期目录（如 aiops25/2025-06-06），其下一层同名目录（aiops25/2025-06-06/2025-06-06/trace-parquet）
    date_dirs = sorted(root.glob(args.pattern))

    if not date_dirs:
        print(f"[WARN] 在 {root} 下未找到符合 {args.pattern} 的日期 目录")
        return

    for d in date_dirs:
        inner = d / d.name / "trace-parquet"
        if not inner.exists():
            print(f"[WARN] 跳过：{inner} 不存在")
            continue

        # import shutil
        # # 🔥 清理同级非 trace-parquet 目录
        # parent_dir = inner.parent
        # for item in parent_dir.iterdir():
        #     if item.is_dir() and item.name != "trace-parquet":
        #         print(f"[CLEAN] 删除目录：{item}")
        #         shutil.rmtree(item, ignore_errors=True)

        # 本日输出文件
        out_csv = out_root / f"{d.name}.csv"
        print(f"\n=== 处理日期 {d.name} ===")
        print(f"扫描目录：{inner}")

        # 收集所有 parquet / parguet
        pq_files = sorted(list(inner.glob("*.parquet")))
        if not pq_files:
            print(f"[WARN] {inner} 下没有 parquet/parguet 文件，跳过")
            continue

        rows: List[dict] = []
        for i, pf in enumerate(pq_files, 1):
            print(f"[{i}/{len(pq_files)}] Read {pf.name}")
            try:
                process_parquet_file(pf, rows)
            except Exception as e:
                print(f"[ERROR] 读取 {pf.name} 失败：{e}")

        finalize_and_write(rows, out_csv)


if __name__ == "__main__":
    main()
