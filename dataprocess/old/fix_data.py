#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用文件路径列表代替通配符，修复 EndTimeMs（duration 微秒→毫秒）。
用法：
    python fix_endtime_path.py Data/2025-06-06.csv
    python fix_endtime_path.py Data/2025-06-06.csv Data/2025-06-07.csv --suffix fix
"""
import argparse, pathlib, sys
import pandas as pd


def fix_csv(csv_file: pathlib.Path, suffix: str = ""):
    print(f"🔧 修复 {csv_file}")
    df = pd.read_csv(csv_file)

    # 1. 差值缩回 1000 倍
    delta_ms = (df["EndTimeMs"] - df["StartTimeMs"]) / 1000.0
    df["EndTimeMs"] = df["StartTimeMs"] + delta_ms

    # 2. 重新 0-1 归一化
    trace_range = df.groupby("TraceID").agg(
        start_min=("StartTimeMs", "min"),
        end_max=("EndTimeMs", "max"),
    )
    df = df.merge(trace_range, left_on="TraceID", right_index=True)
    dur = (df["end_max"] - df["start_min"]).clip(lower=1e-6)
    df["Normalized_StartTime"] = (df["StartTimeMs"] - df["start_min"]) / dur
    df["Normalized_EndTime"] = (df["EndTimeMs"] - df["start_min"]) / dur
    df = df.drop(columns=["start_min", "end_max"])

    # 3. 输出
    out_file = csv_file.with_suffix(f".{suffix}.csv") if suffix else csv_file
    df.to_csv(out_file, index=False, float_format="%.8f")
    print(f"✅ 已写入 {out_file}")


def main():
    ap = argparse.ArgumentParser(description="Fix EndTimeMs by file paths.")
    ap.add_argument("paths", nargs="+", help="CSV 文件路径（可多个）")
    ap.add_argument("--suffix", default="", help="非空时另存 *.{suffix}.csv，否则原地覆盖")
    args = ap.parse_args()

    for p in args.paths:
        fix_csv(pathlib.Path(p), args.suffix)


if __name__ == "__main__":
    main()