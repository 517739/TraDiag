import os
import pandas as pd
import pyarrow.parquet as pq
from datetime import datetime, timezone

def iso_to_ms(s):
    if not s: return 0
    s = str(s).strip()
    if s.endswith("Z"):
        dt = datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    return int(datetime.fromisoformat(s).timestamp() * 1000)

# 定义需要的列名和相关的指标
required_columns = ['time','kpi_key', 'kubernetes_node']

# 特定需要的文件指标
metrics = [
    'infra_node_node_cpu_usage_rate',
    'infra_node_node_disk_read_time_seconds_total',
    'infra_node_node_disk_write_time_seconds_total',
    'infra_node_node_filesystem_usage_rate',
    'infra_node_node_memory_usage_rate',
    'infra_node_node_network_receive_bytes_total',
    'infra_node_node_network_transmit_bytes_total'
]

# 文件目录
base_dir = 'E:\ZJU\AIOps\Projects\TraDNN\dataset/aiops25/row'  # 您的根目录，实际情况下需要调整为您的实际路径
out_dir = 'E:\ZJU\AIOps\Projects\TraDNN\dataset/aiops25/infra'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# 合并所有读取的文件
merged_data = []

# 遍历目录
for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)

    full_df = None
    # 确保是文件夹
    if os.path.isdir(folder_path) and folder_name.startswith("2025"):
        infra_node_path = os.path.join(folder_path, 'metric-parquet', 'infra', 'infra_node')

        # 如果该目录存在
        if os.path.exists(infra_node_path):
            # 遍历infra_node目录下的所有文件
            for file_name in os.listdir(infra_node_path):
                if file_name.endswith('.parquet'):
                    prefix = 'infra_node_'
                    suffix = '_2025'
                    start_idx = len(prefix)
                    end_idx = file_name.find(suffix)

                    metric_path = file_name[:end_idx]  # 获取指标名，如：node_cpu_usage_rate
                    metric_name = file_name[start_idx:end_idx]

                    # 只处理需要的指标
                    if metric_path in metrics:
                        # 读取Parquet文件
                        file_path = os.path.join(infra_node_path, file_name)
                        table = pq.read_table(file_path)
                        df = table.to_pandas()

                        # 将 UTC 时间转换为毫秒
                        df['time'] = df['time'].apply(iso_to_ms)

                        # 关键：对filesystem_usage_rate指标过滤device
                        if metric_name == 'node_filesystem_usage_rate':
                            # 检查是否存在device列
                            if 'device' not in df.columns:
                                print(f"警告：文件 {file_name} 缺少device列，跳过处理")
                                continue
                            # 过滤出目标device
                            target_device = '/dev/mapper/aiops-aiops'
                            df = df[df['device'] == target_device].copy()
                            # 过滤后数据为空则跳过
                            if df.empty:
                                print(f"警告：文件 {file_name} 中无device={target_device}的数据，跳过处理")
                                continue

                        # 提取需要的列：仅保留time、Kubernetes_node和kpi_api（根据实际列名调整）
                        required_cols = ['time', 'kubernetes_node', metric_name]
                        if not set(required_cols).issubset(df.columns):
                            print(f"警告：文件 {file_name} 缺少必要列，跳过处理")
                            continue

                        df = df[required_cols]

                        # 合并到完整数据表
                        if full_df is None:
                            # 第一次处理：直接赋值为基础表
                            full_df = df
                        else:
                            # 后续处理：按time和Kubernetes_node合并，保留所有数据（外连接）
                            full_df = full_df.merge(
                                df,
                                on=['time', 'kubernetes_node'],
                                how='outer'  # 保留所有时间和节点的组合，空缺值为NaN
                            )

    # 生成最终完整表（如果有数据）
    if full_df is not None and not full_df.empty:
        # 1. 从metrics列表中提取指标名（去掉前缀'infra_node_'，与DataFrame中的列名对应）
        # 例如：'infra_node_node_cpu_usage_rate' → 'node_cpu_usage_rate'
        ordered_metric_names = [m.replace('infra_node_', '') for m in metrics]

        # 2. 定义固定的基础列（放在最前面）
        base_columns = ['time', 'timeMs', 'kubernetes_node']

        # 3. 过滤出在full_df中实际存在的指标列（避免因文件缺失导致的列不存在报错）
        existing_metrics = [metric for metric in ordered_metric_names if metric in full_df.columns]

        # 4. 组合最终列顺序：基础列 + 按metrics顺序的指标列
        final_column_order = base_columns + existing_metrics

        # 5. 按最终顺序重排列DataFrame的列
        full_df = full_df[final_column_order]

        # 6. 保存为CSV
        print(f"数据行数：{len(full_df)}")
        output_csv = os.path.join(out_dir, folder_name + '.csv')
        full_df.to_csv(output_csv, index=False)
        print(f"完整数据表已生成（列顺序与metrics一致）：{output_csv}")
    else:
        print("未找到符合条件的文件或数据，未生成完整表")
