# classify_service_traces.py
# 用法: python classify_service_traces.py --in combined_services.csv --out-dir ./  (默认当前目录)

import argparse
import pandas as pd
import numpy as np
from collections import Counter, defaultdict

# ---------- 映射（按 Trace 视角：结构 / 时延） ----------
# 优先用 fault_type；缺失再按 fault_category 兜底
STAGE2_BY_TYPE = {
    # Structure
    "dns error": "Structure",
    "target port misconfig": "Structure",
    "code error": "Structure",
    "pod failure": "Structure",
    "pod kill": "Structure",
    "network loss": "Structure",
    "network corrupt": "Structure",
    "jvm exception": "Structure",
    # Latency
    "cpu stress": "Latency",
    "memory stress": "Latency",
    "jvm cpu": "Latency",
    "jvm gc": "Latency",
    "jvm latency": "Latency",
    "network delay": "Latency",
}

STAGE2_BY_CAT = {
    "dns fault": "Structure",
    "misconfiguration": "Structure",
    "erroneous change": "Structure",
    "pod fault": "Structure",
    "stress test": "Latency",
    "jvm fault": "Latency",
    "network attack": "Latency",  # 具体按 type 更准；这里兜底给 Latency
    "io fault": "Latency",        # service 层常表现为慢，兜底放 Latency
}

# Fine 直接用 fault_type（做一些别名合并/标准化）
# 若要合并更多别名，往这里加 mapping 即可
FINE_NORMALIZE = {
    "network corruption": "network corrupt",
    "cpu-stress": "cpu stress",
    "mem stress": "memory stress",
}

def norm_text(x: str) -> str:
    return (x or "").strip().lower()

def pick_stage2(ft: str, fc: str) -> str:
    ft = norm_text(ft)
    fc = norm_text(fc)
    # 别名整理
    ft = FINE_NORMALIZE.get(ft, ft)
    s2 = STAGE2_BY_TYPE.get(ft)
    if not s2:
        s2 = STAGE2_BY_CAT.get(fc, "Latency")
    return s2

def pick_fine(ft: str, fc: str) -> str:
    ft = norm_text(ft)
    fc = norm_text(fc)
    ft = FINE_NORMALIZE.get(ft, ft)
    if ft:
        return ft
    # 没有 fault_type 就用 category 兜底
    return fc if fc else "unknown"

# ---------- 选 Trace “主服务” ----------
def root_service_of_trace(df_trace: pd.DataFrame) -> str:
    # 1) 根 span（ParentID 为空）
    roots = df_trace[df_trace["ParentID"].isna() | (df_trace["ParentID"]=="")]
    if not roots.empty:
        sv = roots["ServiceName"].mode()
        if len(sv): return str(sv.iloc[0])
    # 2) span 数最多
    cnt = df_trace.groupby("ServiceName")["SpanId"].count().sort_values(ascending=False)
    if not cnt.empty: return str(cnt.index[0])
    # 3) 总时长最长
    dur = (df_trace["EndTimeMs"] - df_trace["StartTimeMs"]).groupby(df_trace["ServiceName"]).sum().sort_values(ascending=False)
    if not dur.empty: return str(dur.index[0])
    # 兜底
    sv = df_trace["ServiceName"].dropna().astype(str)
    return sv.iloc[0] if len(sv) else ""

# ---------- 主流程 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", required=True, help="combined_services.csv（含 fault_type / fault_category 列）")
    ap.add_argument("--out-dir", default=".", help="输出目录")
    ap.add_argument("--chunksize", type=int, default=0, help=">0 则分块处理（建议 0~100万之间）")
    args = ap.parse_args()

    in_csv = args.in_csv
    out_dir = args.out_dir.rstrip("/")

    # 读全量（6万多 span 可直接读内存；若更大可加 chunksize 流式）
    df = pd.read_csv(in_csv)
    need_cols = [
        "TraceID","SpanId","ParentID","NodeName","ServiceName","PodName","URL",
        "HttpStatusCode","StatusCode","SpanKind","StartTimeMs","EndTimeMs",
        "Normalized_StartTime","Normalized_EndTime","fault_category","fault_type"
    ]
    miss = set(need_cols) - set(df.columns)
    if miss:
        raise RuntimeError(f"CSV 缺少列：{miss}")

    # 规范文本
    df["fault_type"] = df["fault_type"].astype(str).str.strip()
    df["fault_category"] = df["fault_category"].astype(str).str.strip()

    # Stage1
    # 目前这批都是 service 故障的 span → 全标 Fault；若后续混入 normal，可改为根据是否落在任何注入窗内再置 Fault
    df["stage1"] = "Fault"

    # Stage2 / Fine（按 span 先打标签）
    df["stage2"] = [pick_stage2(ft, fc) for ft, fc in zip(df["fault_type"], df["fault_category"])]
    df["fine"]   = [pick_fine(ft, fc)   for ft, fc in zip(df["fault_type"], df["fault_category"])]

    # —— 先做 trace 级“主服务” & “主 fine” 决策 —— #
    trace_groups = df.groupby("TraceID", sort=False)

    # 准备承接 trace 级结果
    t_stage1, t_stage2, t_fine, t_service = {}, {}, {}, {}
    # 1) 主服务（root_service）
    for tid, g in trace_groups:
        t_service[tid] = root_service_of_trace(g)

    # 2) 主 fine：多数投票（平手→最早出现的 span 的 fine）
    for tid, g in trace_groups:
        fines = g["fine"].astype(str).tolist()
        cnt = Counter(fines)
        top = cnt.most_common()
        if len(top) == 1 or (len(top) > 1 and top[0][1] > top[1][1]):
            best_fine = top[0][0]
        else:
            # 平手：取最早 StartTime 的那个 fine
            g2 = g.sort_values(["StartTimeMs","EndTimeMs","SpanId"])
            best_fine = str(g2.iloc[0]["fine"])
        t_fine[tid] = best_fine
        # stage2 按该 fine 的类型/类目再映射一次（保持一致）
        ft_any = str(g.loc[g["fine"]==best_fine, "fault_type"].iloc[0]) if (g["fine"]==best_fine).any() else ""
        fc_any = str(g.loc[g["fine"]==best_fine, "fault_category"].iloc[0]) if (g["fine"]==best_fine).any() else ""
        t_stage2[tid] = pick_stage2(ft_any, fc_any)
        t_stage1[tid] = "Fault"

    # —— 回填到 span 级并生成 superfine —— #
    df["trace_root_service"] = df["TraceID"].map(t_service)
    df["fine_trace"] = df["TraceID"].map(t_fine)
    df["stage2_trace"] = df["TraceID"].map(t_stage2)
    # Superfine = fine_trace @ root_service
    df["superfine"] = df.apply(lambda r: f"{r['fine_trace']}@{r['trace_root_service']}".lower(), axis=1)

    # 输出 1：带标签的 span 级 CSV
    spans_out = f"{out_dir}/combined_services_labeled_spans.csv"
    out_cols = [
        "TraceID","SpanId","ParentID","NodeName","ServiceName","PodName","URL",
        "HttpStatusCode","StatusCode","SpanKind","StartTimeMs","EndTimeMs",
        "Normalized_StartTime","Normalized_EndTime",
        "fault_category","fault_type","stage1","stage2_trace","fine_trace","superfine"
    ]
    df[out_cols].to_csv(spans_out, index=False, encoding="utf-8", float_format="%.6f")

    # 输出 2：trace 级标签表
    rows = []
    for tid, g in df.groupby("TraceID", sort=False):
        rs = t_service.get(tid, "")
        f  = t_fine.get(tid, "unknown")
        s2 = t_stage2.get(tid, "Latency")
        sf = f"{f}@{rs}".lower()
        rows.append({
            "TraceID": tid,
            "stage1": "Fault",
            "stage2": s2,
            "fine": f,
            "superfine": sf,
            "root_service": rs,
            "span_count": len(g),
            "span_dur_sum": float((g["EndTimeMs"]-g["StartTimeMs"]).sum())
        })
    trace_out = pd.DataFrame(rows)
    traces_csv = f"{out_dir}/combined_services_trace_labels.csv"
    trace_out.to_csv(traces_csv, index=False, encoding="utf-8", float_format="%.6f")

    # 统计
    tot_traces = len(trace_out)
    print(f"[STAT] traces={tot_traces}")
    for col in ["stage2", "fine"]:
        cnt = trace_out[col].value_counts()
        print(f"[STAT] {col}:")
        for k, v in cnt.items():
            print(f"  - {k:16s}  {v:7d}  ({v/tot_traces:.2%})")

    print(f"[OK] spans → {spans_out}")
    print(f"[OK] traces → {traces_csv}")

if __name__ == "__main__":
    main()
