import pandas as pd
from typing import Dict, Tuple
from collections import Counter
import sys
import os

def count_orphan_by_fault_type(csv_path: str,
                               fault_type_col: str = "fault_type",
                               root_indicators: frozenset = frozenset({"-1", "0", ""})) -> Tuple[int, int, float, Dict[str, int]]:
    """
    读取 CSV（含 TraceID/SpanId/ParentID/fault_type），统计总 Trace 数、存在孤儿 span 的 Trace 数及
    各 fault_type 下孤儿 Trace 的数量。

    参数
    ----
    csv_path : str
        输入 csv 文件路径
    fault_type_col : str, optional
        fault_type 列名，默认 "fault_type"
    root_indicators : frozenset, optional
        被认为是根的标志集合，默认 {"-1", "0", ""}

    返回
    ----
    total_traces : int
        总 Trace 条数
    orphan_traces : int
        存在孤儿 span 的 Trace 条数
    orphan_rate : float
        孤儿 Trace 占比（0~1）
    orphan_by_type : Dict[str, int]
        各 fault_type 下孤儿 Trace 计数
    """
    # 读取所需列，缺失 fault_type 用空字符串填充
    df = pd.read_csv(csv_path, dtype=str)
    for col in ["TraceID", "SpanId", "ParentID", fault_type_col]:
        if col not in df.columns:
            raise ValueError(f"列 {col} 不存在于 CSV")
    df = df[["TraceID", "SpanId", "ParentID", fault_type_col]].fillna({fault_type_col: ""})

    total_traces = 0
    orphan_traces = 0
    orphan_by_type: Dict[str, int] = {}

    for tid, g in df.groupby("TraceID"):
        total_traces += 1
        span_pool = set(g["SpanId"].astype(str))
        has_orphan = False
        for _, row in g.iterrows():
            pid = str(row["ParentID"])
            if pid not in root_indicators and pid not in span_pool:
                has_orphan = True
                break
        # 取该 Trace 的 fault_type（众数，空字符串兜底）
        fault_type = g[fault_type_col].mode().iloc[0] if not g[fault_type_col].mode().empty else ""
        if has_orphan:
            orphan_traces += 1
            orphan_by_type[fault_type] = orphan_by_type.get(fault_type, 0) + 1

    orphan_rate = orphan_traces / total_traces if total_traces else 0.0
    return total_traces, orphan_traces, orphan_rate, orphan_by_type

def count_max_trace_nodes(csv_path, trace_id_col):
    """
    统计CSV文件中各Trace的节点数，并返回最大节点数

    参数:
        csv_path (str): CSV文件路径
        trace_id_col (str): Trace ID所在的列名（如'trace_id'）
    """
    # 检查文件是否存在
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"文件不存在: {csv_path}")

    # 读取CSV文件（根据文件大小自动选择合适的引擎）
    try:
        df = pd.read_csv(csv_path, engine='pyarrow' if os.path.getsize(csv_path) > 100 * 1024 * 1024 else 'c')
    except Exception as e:
        raise RuntimeError(f"读取CSV失败: {str(e)}")

    # 检查Trace ID列是否存在
    if trace_id_col not in df.columns:
        raise ValueError(f"CSV中未找到指定的Trace ID列: {trace_id_col}")

    # 按Trace ID分组，统计每个Trace的节点数（即分组内的记录数）
    trace_node_counts = df.groupby(trace_id_col).size()

    # 获取最大节点数
    max_nodes = trace_node_counts.max()
    return max_nodes


# 示例调用
if __name__ == "__main__":
    # tot, orphan, rate, by_type = count_orphan_by_fault_type("E:\ZJU\AIOps\Projects\TraDNN\TraDiag/trace_svnd_all\dataset\Data/Node_fault.csv")
    # print(f"Total traces: {tot}")
    # print(f"Orphan traces: {orphan} ({rate:.2%})")
    # print("Orphan count by fault_type:")
    # for ft, cnt in by_type.items():
    #     print(f"  {ft or 'NULL'}: {cnt}")

    # 判断HttpStatusCode和StatusCode情况
    # csv_path = "E:\ZJU\AIOps\Projects\TraDNN\TraDiag/trace_svnd_all\dataset\Data/Node_fault.csv"  # 改成自己的文件
    # col_status = "StatusCode"
    # col_http = "HttpStatusCode"
    #
    # counter_http = Counter(
    #     pd.read_csv(csv_path, usecols=[col_http], dtype=str)[col_http]
    #     .dropna()
    #     .astype(str)
    #     .str.strip()
    # )
    #
    # counter_status = Counter(
    #     pd.read_csv(csv_path, usecols=[col_status], dtype=str)[col_status]
    #     .dropna()
    #     .astype(str)
    #     .str.strip()
    # )
    #
    # print("HttpStatusCode 计数：")
    # for code, cnt in counter_http.most_common():
    #     print(f"{code:>10} : {cnt:,}")
    #
    # print("StatusCode 计数：")
    # for code, cnt in counter_status.most_common():
    #     print(f"{code:>10} : {cnt:,}")
    #
    # import pandas as pd
    # import argparse
    # import os
    #
    #
    # def count_max_trace_nodes(csv_path, trace_id_col):
    #     """
    #     统计CSV文件中各Trace的节点数，并返回最大节点数
    #
    #     参数:
    #         csv_path (str): CSV文件路径
    #         trace_id_col (str): Trace ID所在的列名（如'trace_id'）
    #     """
    #     # 检查文件是否存在
    #     if not os.path.exists(csv_path):
    #         raise FileNotFoundError(f"文件不存在: {csv_path}")
    #
    #     # 读取CSV文件（根据文件大小自动选择合适的引擎）
    #     try:
    #         df = pd.read_csv(csv_path, engine='pyarrow' if os.path.getsize(csv_path) > 100 * 1024 * 1024 else 'c')
    #     except Exception as e:
    #         raise RuntimeError(f"读取CSV失败: {str(e)}")
    #
    #     # 检查Trace ID列是否存在
    #     if trace_id_col not in df.columns:
    #         raise ValueError(f"CSV中未找到指定的Trace ID列: {trace_id_col}")
    #
    #     # 按Trace ID分组，统计每个Trace的节点数（即分组内的记录数）
    #     trace_node_counts = df.groupby(trace_id_col).size()
    #
    #     # 获取最大节点数
    #     max_nodes = trace_node_counts.max()
    #     return max_nodes

    csv_file = "E:\ZJU\AIOps\Projects\TraDNN\TraDiag/trace_svnd_all\dataset\Data/Normal0607.csv"
    trace_id_col = "TraceID"

    try:
        max_nodes = count_max_trace_nodes(csv_file, trace_id_col)
        print(f"CSV文件中所有Trace的最大节点数为: {max_nodes}")
    except Exception as e:
        print(f"错误: {str(e)}", file=sys.stderr)
        exit(1)