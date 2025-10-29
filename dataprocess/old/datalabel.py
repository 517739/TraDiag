import pandas as pd
import ast
from datetime import timezone

def load_and_preprocess_data(trace_file, groundtruth_file):
    """加载并预处理trace数据和故障注入数据"""
    # 加载trace数据
    df_traces = pd.read_csv(trace_file)
    
    # 确保时间戳为整数类型
    # df_traces['StartTime'] = pd.to_numeric(df_traces['StartTimeMs'], errors='coerce')
    # df_traces['EndTime'] = pd.to_numeric(df_traces['EndTimeMs'], errors='coerce')
    
    # 加载故障注入数据
    df_groundtruth = pd.read_csv(groundtruth_file)
    
    # 处理 instance 列，将字符串形式的列表转换为实际列表
    def parse_instance(instance_str):
        if pd.isna(instance_str) or instance_str == '':
            return []
        try:
            # 尝试使用 ast.literal_eval 安全地解析列表
            return ast.literal_eval(instance_str)
        except (ValueError, SyntaxError):
            # 如果不是列表，就当作单个字符串处理
            return [instance_str]
    
    df_groundtruth['parsed_instance'] = df_groundtruth['instance'].apply(parse_instance)

    def iso_to_ms(s):
        if pd.isna(s) or s == '':
            return 0
        return int(pd.to_datetime(s).replace(tzinfo=timezone.utc).timestamp() * 1000)

    df_groundtruth['start_time_ms'] = df_groundtruth['start_time'].apply(iso_to_ms)
    df_groundtruth['end_time_ms']   = df_groundtruth['end_time'].apply(iso_to_ms)
    
    return df_traces, df_groundtruth

def is_time_overlap(start1, end1, start2, end2):
    """判断两个时间区间是否重叠"""
    return max(start1, start2) <= min(end1, end2)

def determine_trace_anomaly_and_root_cause(trace_group, groundtruth_df):
    """
    判断一个trace group是否异常，并确定其根因
    
    Args:
        trace_group (pd.DataFrame): 属于同一个TraceID的Span集合
        groundtruth_df (pd.DataFrame): 故障注入数据
    
    Returns:
        tuple: (is_anomaly, root_cause)
    """
    trace_start = trace_group['StartTimeMs'].min()
    trace_end = trace_group['EndTimeMs'].max()
    
    # 遍历所有故障注入记录
    for _, fault in groundtruth_df.iterrows():
        fault_start = fault['start_time_ms']
        fault_end = fault['end_time_ms']
        
        # 检查时间窗口是否重叠
        if not is_time_overlap(trace_start, trace_end, fault_start, fault_end):
            continue
            
        # Case A & C: 服务或实例故障 / 多实例故障
        if pd.notna(fault['service']) and fault['service'] != '':
            # 检查trace中是否涉及该服务
            if (trace_group['ServiceName'] == fault['service']).any():
                # 确定根因
                instances = fault['parsed_instance']
                if instances:
                    # 多实例情况，取第一个作为根因
                    if isinstance(instances, list) and len(instances) > 1:
                        root_cause = f"{fault['service']}_multi_instance"
                    else:
                        # 单实例情况
                        instance_name = instances[0] if isinstance(instances, list) else instances
                        root_cause = f"{fault['service']}.{instance_name}"
                else:
                    # 仅有服务名
                    root_cause = fault['service']
                return True, root_cause
                
        # Case B: 网络故障 (source -> destination)
        elif pd.notna(fault['source']) and pd.notna(fault['destination']) and \
             fault['source'] != '' and fault['destination'] != '':
            source = fault['source']
            destination = fault['destination']
            
            # 查找 source 作为 client 的 Span
            source_client_spans = trace_group[
                (trace_group['ServiceName'] == source) & 
                (trace_group['SpanKind'] == 'client')
            ]
            
            # 检查这些 client Span 的子 Span 是否是 destination
            for _, client_span in source_client_spans.iterrows():
                child_spans = trace_group[trace_group['ParentID'] == client_span['SpanId']]
                if (child_spans['ServiceName'] == destination).any():
                    root_cause = f"{source}_{destination}_network"
                    return True, root_cause
    
    # 如果没有匹配到任何故障
    return False, None

def label_traces(df_traces, df_groundtruth):
    """为所有trace进行标注"""
    # 初始化新列
    df_traces['Anomaly'] = False
    df_traces['RootCause'] = None
    
    # 按 TraceID 分组处理
    grouped = df_traces.groupby('TraceID')
    
    results = {}
    for trace_id, group in grouped:
        is_anomaly, root_cause = determine_trace_anomaly_and_root_cause(group, df_groundtruth)
        results[trace_id] = {
            'Anomaly': is_anomaly,
            'RootCause': root_cause
        }
    
    # 将结果广播到所有Span
    df_traces['Anomaly'] = df_traces['TraceID'].map(lambda x: results[x]['Anomaly'])
    df_traces['RootCause'] = df_traces['TraceID'].map(lambda x: results[x]['RootCause'])
    
    return df_traces

def main():
    trace_file = '../aiops25/processed/2025-06-06.csv'
    groundtruth_file = '../Data/groundtruth.csv'
    output_file = '../Data/Labeled'
    
    print("Loading and preprocessing data...")
    df_traces, df_groundtruth = load_and_preprocess_data(trace_file, groundtruth_file)
    
    print("Labeling traces...")
    df_labeled = label_traces(df_traces, df_groundtruth)
    
    print(f"Saving results to {output_file}...")
    df_labeled.to_csv(output_file, index=False)
    print("Done!")

if __name__ == "__main__":
    main()



