#!/usr/bin/env python3
import pandas as pd
import networkx as nx
import argparse
import os
import sys
import glob
from pathlib import Path
from datetime import datetime, timezone, timedelta
import matplotlib.pyplot as plt

def detect_cycles(csv_path):
    """
    逐 Trace 建图，一旦检测到环即输出 TraceID
    """
    # 只读必要列，节省内存
    df = pd.read_csv(csv_path, usecols=["TraceID", "SpanId", "ParentID"])
    df = df.dropna(subset=["TraceID", "SpanId"])        # 缺失行丢弃
    total_traces = df["TraceID"].nunique()
    cycle_traces = 0

    for trace_id, group in df.groupby("TraceID"):
        G = nx.DiGraph()
        for _, row in group.iterrows():
            child = str(row["SpanId"])
            parent = str(row["ParentID"]) if pd.notna(row["ParentID"]) else None
            G.add_node(child)                     # 保证孤立节点也入图
            if parent and parent != child:        # 自环也记
                G.add_edge(parent, child)

        # 检测简单环（长度>=1）
        cycles = list(nx.simple_cycles(G))
        if cycles:
            cycle_traces += 1
            print(trace_id)          # 标准输出，可重定向到文件
            # 如需调试：打印环
            # print(f"{trace_id}  cycles={cycles}")

    # 可选：统计信息到 stderr（不影响重定向）
    print(f"[Done] total={total_traces}  with_cycles={cycle_traces}", file=sys.stderr)

TARGET_DIRS = {'node', 'pod', 'service'}
def merge_pair(sub_dir_a: Path, sub_dir_b: Path, out_sub_dir: Path):
    """合并单个子目录下的所有 csv"""
    out_sub_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_sub_dir / 'merged.csv'

    # 收集两侧 csv
    files_a = list(sub_dir_a.glob('*.csv')) if sub_dir_a.exists() else []
    files_b = list(sub_dir_b.glob('*.csv')) if sub_dir_b.exists() else []
    all_files = files_a + files_b

    if not all_files:          # 两个目录都没有 csv
        print(f'[WARN] 未找到 csv：{sub_dir_a}  ||  {sub_dir_b} ，跳过')
        return

    total_ids_before = 0
    df_list = []
    for f in all_files:
        df = pd.read_csv(f)
        uniq = df['TraceID'].nunique()
        total_ids_before += uniq
        print(f'[INFO] {f.name}  TraceID 唯一数：{uniq}')
        df_list.append(df)
    merged = pd.concat(df_list, axis=0, ignore_index=True, sort=False)
    merged.to_csv(out_file, index=False, encoding='utf-8')
    print(f'[INFO] 写出 {len(merged)} 行 -> {out_file}')
    print(f'[INFO] 合并后总行数：{len(merged)}，合并后 TraceID 唯一数：{merged["TraceID"].nunique()}')

def iso_to_ms(s):
    if not s: return 0
    s = str(s).strip()
    if s.endswith("Z"):
        dt = datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    return int(datetime.fromisoformat(s).timestamp() * 1000)

def ms_to_iso(ms: int) -> str:
    """毫秒时间戳 -> UTC 的 ISO-8601 字符串（精确到秒）"""
    if ms <= 0:
        return ""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

def main():
    # 检测是否有环
    # ap = argparse.ArgumentParser()
    # ap.add_argument("--csv", default="E:\ZJU\AIOps\Projects\TraDNN\TraDiag\Data\SplitTrace\service\ALL_spans.csv", help="trace csv file (must contain TraceID,SpanId,ParentID)")
    # args = ap.parse_args()
    # if not os.path.isfile(args.csv):
    #     sys.exit(f"File not found: {args.csv}")
    # detect_cycles(args.csv)

    # # csv合并
    # ap = argparse.ArgumentParser(description='合并双目录 node/pod/service csv')
    # ap.add_argument('--a', default="E:\ZJU\AIOps\Projects\TraDNN\dataset\SplitTrace0614", help='第一个根目录')
    # ap.add_argument('--b', default="E:\ZJU\AIOps\Projects\TraDNN\dataset\SplitTrace1729", help='第二个根目录')
    # ap.add_argument('--o', default="E:\ZJU\AIOps\Projects\TraDNN\dataset\SplitTrace", help='输出根目录')
    # args = ap.parse_args()
    #
    # root_a = Path(args.a).expanduser().resolve()
    # root_b = Path(args.b).expanduser().resolve()
    # root_out = Path(args.o).expanduser().resolve()
    #
    # if not root_a.is_dir() or not root_b.is_dir():
    #     raise SystemExit('请确保 --a 和 --b 都是有效目录')
    #
    # for sub in TARGET_DIRS:
    #     merge_pair(root_a / sub, root_b / sub, root_out / sub)
    #
    # print('[OK] 全部合并完成')

    # # 查看gdt命中情况
    # mask = (st > 1749178985000) & (et < 1749180485000) & (df['PodName'] == "shippingservice-0")       2101
    # mask = (st > 1749189791000) & (et < 1749190811000) & (df['PodName'] == "shippingservice-0")       175
    # mask = (st > 1749215009000) & (et < 1749216509000) & (df['PodName'] == "productcatalogservice-2") 2667
    # 2025-06-06T08:03:11Z	2025-06-06T08:20:11Z
    # st = iso_to_ms("2025-06-06T08:03:11Z")
    # et = iso_to_ms("2025-06-06T08:20:11Z")
    # print(st)
    # print(et)
    # df = pd.read_csv('E:/ZJU/AIOps/Projects/TraDNN/dataset/aiops25/processed/phaseone/2025-06-06.csv')
    # st = pd.to_numeric(df['StartTimeMs'], errors='coerce')
    # et = pd.to_numeric(df['EndTimeMs'], errors='coerce')
    #
    # wins_st = iso_to_ms("2025-06-06T00:10:14Z")
    # wins_et = iso_to_ms("2025-06-06T00:29:14Z")
    # col = 'ServiceName'
    # instance_name = 'currencyservice'
    # print(wins_st)
    # print(wins_et)
    # # mask1 = (st > 1749160216000) & (et < 1749160876000)
    # # mask2 = (df['ServiceName'] == "frontend")
    # mask = (st > wins_st) & (et < wins_et) & (df[col] == instance_name)
    # # sub2 = df.loc[mask2]
    # sub = df.loc[mask]
    # trace_cnt = sub['TraceID'].nunique()
    # # print(sub)  # 看结果
    # # print(len(sub1))  # 看数量
    # # print(len(sub2))  # 看数量
    # print(len(sub))
    # print(trace_cnt)
    # print(sub)

    # 查看特定故障
    # df = pd.read_csv(r'E:\ZJU\AIOps\Projects\TraDNN\dataset\SplitTrace\node\merged.csv')
    #
    # # 原地转换
    # df['StartTime_ISO'] = df['StartTimeMs'].apply(ms_to_iso)
    # df['EndTime_ISO'] = df['EndTimeMs'].apply(ms_to_iso)
    #
    # # 写回同一文件（或另存）
    # df.to_csv(r'E:\ZJU\AIOps\Projects\TraDNN\dataset\SplitTrace\node\merged.csv',
    #           index=False, encoding='utf-8')

    # # 判断每个故障有多少Trace
    # # 1. 读文件
    # df = pd.read_csv('E:\ZJU\AIOps\Projects\TraDNN\dataset\SplitTrace/pod\merged.csv')
    # # 2. 基本校验
    # if {'TraceID', 'fault_type'} - set(df.columns):
    #     missing = {'TraceID', 'fault_type'} - set(df.columns)
    #     sys.exit(f"CSV缺少列: {missing}")
    # # 3. 按 fault_type 分组，统计每组 TraceID 的唯一值数量
    # counts = df.groupby('fault_type')['TraceID'].nunique().reset_index(name='trace_count')
    # # 4. 输出结果
    # print("fault_type,trace_count")
    # for _, row in counts.iterrows():
    #     print(f"{row['fault_type']},{row['trace_count']}")

    # 查看trace数量
    # df = pd.read_csv('E:\ZJU\AIOps\Projects\TraDNN\dataset\SplitTrace/node\merged.csv')
    # trace_count = df['TraceID'].nunique()
    # print(f"文件中共有 {trace_count} 条 Trace。")

    # # 画图
    # # 1. 数据
    # df = pd.DataFrame({
    #     'class': ['network delay', 'code error', 'network loss', 'cpu stress',
    #               'network corrupt', 'memory stress', 'dns error'],
    #     'TP': [3720, 2167, 1486, 992, 822, 205, 148],
    #     'Support': [4004, 2443, 2150, 1151, 987, 246, 159],
    #     'Precision': [0.8765, 0.8409, 0.8877, 0.9143, 0.7117, 0.7824, 0.9867],
    #     'Recall': [0.9291, 0.8870, 0.6912, 0.8619, 0.8328, 0.8333, 0.9308],
    #     'F1': [0.9020, 0.8633, 0.7772, 0.8873, 0.7675, 0.8071, 0.9579]
    # })
    #
    # # 2. 百分比格式化
    # df[['Precision', 'Recall', 'F1']] = df[['Precision', 'Recall', 'F1']].applymap('{:.2%}'.format)
    #
    # # 3. 绘制表格
    # fig, ax = plt.subplots(figsize=(10, 3))
    # ax.axis('off')
    # table = ax.table(cellText=df.values,
    #                  colLabels=df.columns,
    #                  cellLoc='center',
    #                  loc='center',
    #                  colWidths=[0.25, 0.08, 0.08, 0.08, 0.08, 0.08])
    # table.auto_set_font_size(False)
    # table.set_fontsize(11)
    # table.scale(1, 1.8)
    # plt.title('Per-Class Metrics', fontsize=14, pad=20)
    # plt.tight_layout()
    # plt.savefig('pretty_table.png', dpi=300, bbox_inches='tight')
    # plt.show()

    # def ms_to_iso(ms: int) -> str:
    #     """毫秒时间戳 -> UTC 的 ISO-8601 字符串（精确到秒）"""
    #     if ms <= 0:
    #         return ""
    #     return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    print(ms_to_iso(1749250986398))

if __name__ == "__main__":
    main()