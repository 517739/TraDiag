#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 精确统计脚本：根据OpenTelemetry标准精确统计各种协议类型的Span数量

import argparse, json, pathlib, pandas as pd, numpy as np
from typing import Dict, Set
from collections import defaultdict


# ---------- 工具函数 ----------
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


# ---------- 精确协议类型判断 ----------
def is_http_span(span_kind, span_tags):
    """判断是否为HTTP Span"""
    if span_kind not in ["client", "server"]:
        return False

    # 检查是否存在HTTP相关标签
    http_tags = ["http.status_code", "http.url", "http.target", "http.method"]
    return any(pick_tag(span_tags, tag) for tag in http_tags)


def is_grpc_span(span_kind, span_tags):
    """判断是否为gRPC Span"""
    if span_kind not in ["client", "server"]:
        return False

    # 检查gRPC特定标签
    rpc_system = pick_tag(span_tags, "rpc.system", "rpc_system")
    if rpc_system and "grpc" in rpc_system.lower():
        return True

    # 检查gRPC状态码
    grpc_status = pick_tag(span_tags, "rpc.grpc.status_code", "rpc_grpc_status_code")
    if grpc_status:
        return True

    return False


def is_db_span(span_kind, span_tags):
    """判断是否为数据库Span"""
    if span_kind != "client":
        return False

    # 检查数据库相关标签
    db_tags = ["db.system", "db.name", "db.statement", "db.operation", "db.type"]
    return any(pick_tag(span_tags, tag) for tag in db_tags)


def is_mq_span(span_kind, span_tags):
    """判断是否为消息队列Span"""
    if span_kind not in ["producer", "consumer", "client"]:
        return False

    # 检查消息队列相关标签
    mq_tags = ["messaging.system", "messaging.destination", "messaging.message_payload_size_bytes"]
    return any(pick_tag(span_tags, tag) for tag in mq_tags)


# ---------- 单个 parquet 文件统计 ----------
def process_parquet_file(pf: pathlib.Path, stats: Dict[str, int], trace_ids: Set[str]):
    df = pd.read_parquet(pf)
    # 兼容不同导出字段名
    col_trace = "traceID" if "traceID" in df.columns else ("traceId" if "traceId" in df.columns else "trace_id")

    for _, r in df.iterrows():
        span_tags = ndarray2dict(r.get("tags", {}))
        trace_id = str(r.get(col_trace, ""))
        span_kind = pick_tag(span_tags, "span.kind", "span_kind").lower()

        # 统计Trace和Span数量
        stats["total_spans"] += 1
        if trace_id and trace_id not in trace_ids:
            trace_ids.add(trace_id)
            stats["total_traces"] += 1

        # 精确判断协议类型
        if is_http_span(span_kind, span_tags):
            stats["http_spans"] += 1
            # 检查HTTP错误状态码
            status_code_str = pick_tag(span_tags, "http.status_code", "http_status_code")
            try:
                status_code = int(status_code_str) if status_code_str not in ("", None) else 0
                if status_code >= 400:
                    stats["http_error_spans"] += 1
            except Exception:
                pass

        elif is_grpc_span(span_kind, span_tags):
            stats["grpc_spans"] += 1
            # 检查gRPC错误状态码
            grpc_status_str = pick_tag(span_tags, "rpc.grpc.status_code", "rpc_grpc_status_code")
            try:
                grpc_status = int(grpc_status_str) if grpc_status_str not in ("", None) else 0
                if grpc_status != 0:  # gRPC状态码0表示成功
                    stats["grpc_error_spans"] += 1
            except Exception:
                pass

        elif is_db_span(span_kind, span_tags):
            stats["db_spans"] += 1
            # 检查数据库错误
            db_error = pick_tag(span_tags, "error", "db.error", "error.message")
            if db_error and db_error != "false" and db_error.lower() != "no error":
                stats["db_error_spans"] += 1

        elif is_mq_span(span_kind, span_tags):
            stats["mq_spans"] += 1
            # 检查消息队列错误
            mq_error = pick_tag(span_tags, "error", "messaging.error", "error.message")
            if mq_error and mq_error != "false" and mq_error.lower() != "no error":
                stats["mq_error_spans"] += 1


# ---------- 扫描 aiops25 目录下所有日期并统计 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="aiops25", help="aiops25 根目录")
    ap.add_argument("--pattern", default="2025-??-??", help="日期目录匹配模式（默认 2025-??-??）")
    ap.add_argument("--output", default="statistics_summary.csv", help="统计结果输出文件")
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    output_file = pathlib.Path(args.output)

    # 一层日期目录（如 aiops25/2025-06-06），其下一层同名目录（aiops25/2025-06-06/2025-06-06/trace-parquet）
    date_dirs = sorted(root.glob(args.pattern))

    if not date_dirs:
        print(f"[WARN] 在 {root} 下未找到符合 {args.pattern} 的日期目录")
        return

    # 准备存储所有日期的统计结果
    all_stats = []

    for d in date_dirs:
        inner = d / d.name / "trace-parquet"
        if not inner.exists():
            print(f"[WARN] 跳过：{inner} 不存在")
            continue

        print(f"\n=== 处理日期 {d.name} ===")
        print(f"扫描目录：{inner}")

        # 初始化统计计数器
        stats = {
            "date": d.name,
            "total_traces": 0,
            "total_spans": 0,
            "http_spans": 0,
            "http_error_spans": 0,
            "grpc_spans": 0,
            "grpc_error_spans": 0,
            "db_spans": 0,
            "db_error_spans": 0,
            "mq_spans": 0,
            "mq_error_spans": 0
        }
        trace_ids = set()  # 用于去重统计Trace数量

        # 收集所有 parquet 文件
        pq_files = sorted(list(inner.glob("*.parquet")))
        if not pq_files:
            print(f"[WARN] {inner} 下没有 parquet 文件，跳过")
            continue

        for i, pf in enumerate(pq_files, 1):
            print(f"[{i}/{len(pq_files)}] 处理 {pf.name}")
            try:
                process_parquet_file(pf, stats, trace_ids)
            except Exception as e:
                print(f"[ERROR] 处理 {pf.name} 失败：{e}")

        # 计算百分比
        if stats["total_spans"] > 0:
            stats["http_ratio"] = stats["http_spans"] / stats["total_spans"] * 100
            stats["grpc_ratio"] = stats["grpc_spans"] / stats["total_spans"] * 100
            stats["db_ratio"] = stats["db_spans"] / stats["total_spans"] * 100
            stats["mq_ratio"] = stats["mq_spans"] / stats["total_spans"] * 100

            # 计算错误率
            stats["http_error_ratio"] = stats["http_error_spans"] / max(1, stats["http_spans"]) * 100
            stats["grpc_error_ratio"] = stats["grpc_error_spans"] / max(1, stats["grpc_spans"]) * 100
            stats["db_error_ratio"] = stats["db_error_spans"] / max(1, stats["db_spans"]) * 100
            stats["mq_error_ratio"] = stats["mq_error_spans"] / max(1, stats["mq_spans"]) * 100

            stats["spans_per_trace"] = stats["total_spans"] / stats["total_traces"]
        else:
            stats.update({
                "http_ratio": 0, "grpc_ratio": 0, "db_ratio": 0, "mq_ratio": 0,
                "http_error_ratio": 0, "grpc_error_ratio": 0, "db_error_ratio": 0, "mq_error_ratio": 0,
                "spans_per_trace": 0
            })

        # 添加到总统计
        all_stats.append(stats)

        # 打印当日统计
        print(f"  Trace总数: {stats['total_traces']}")
        print(f"  Span总数: {stats['total_spans']}")
        print(f"  HTTP Span数: {stats['http_spans']} ({stats['http_ratio']:.2f}%)")
        print(f"    HTTP错误数: {stats['http_error_spans']} ({stats['http_error_ratio']:.2f}%)")
        print(f"  gRPC Span数: {stats['grpc_spans']} ({stats['grpc_ratio']:.2f}%)")
        print(f"    gRPC错误数: {stats['grpc_error_spans']} ({stats['grpc_error_ratio']:.2f}%)")
        print(f"  DB Span数: {stats['db_spans']} ({stats['db_ratio']:.2f}%)")
        print(f"    DB错误数: {stats['db_error_spans']} ({stats['db_error_ratio']:.2f}%)")
        print(f"  MQ Span数: {stats['mq_spans']} ({stats['mq_ratio']:.2f}%)")
        print(f"    MQ错误数: {stats['mq_error_spans']} ({stats['mq_error_ratio']:.2f}%)")
        print(f"  平均每个Trace的Span数: {stats['spans_per_trace']:.2f}")

    # 输出汇总统计到CSV文件
    if all_stats:
        df = pd.DataFrame(all_stats)
        # 重新排列列顺序
        cols = ["date", "total_traces", "total_spans", "spans_per_trace",
                "http_spans", "http_ratio", "http_error_spans", "http_error_ratio",
                "grpc_spans", "grpc_ratio", "grpc_error_spans", "grpc_error_ratio",
                "db_spans", "db_ratio", "db_error_spans", "db_error_ratio",
                "mq_spans", "mq_ratio", "mq_error_spans", "mq_error_ratio"]
        df = df[cols]
        df.to_csv(output_file, index=False, float_format="%.2f")
        print(f"\n统计汇总已保存到: {output_file}")

        # 打印总体统计
        total_spans = sum(s["total_spans"] for s in all_stats)
        if total_spans > 0:
            print("\n=== 总体统计 ===")
            print(f"总Span数: {total_spans}")
            print(f"HTTP占比: {sum(s['http_spans'] for s in all_stats) / total_spans * 100:.2f}%")
            print(f"gRPC占比: {sum(s['grpc_spans'] for s in all_stats) / total_spans * 100:.2f}%")
            print(f"DB占比: {sum(s['db_spans'] for s in all_stats) / total_spans * 100:.2f}%")
            print(f"MQ占比: {sum(s['mq_spans'] for s in all_stats) / total_spans * 100:.2f}%")
    else:
        print("没有找到任何可处理的数据")


if __name__ == "__main__":
    main()