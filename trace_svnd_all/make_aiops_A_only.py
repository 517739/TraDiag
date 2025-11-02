#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_aiops_v5.py - 独立阶段A数据处理
- 训练集和验证集：100%正常数据
- 测试集：90%正常数据 + 10%异常数据（服务异常:节点异常=1:1）
- 所有参数可通过命令行控制
"""
import argparse, os, json, random, hashlib, time
from typing import Optional, List, Dict, Tuple
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
from tqdm import tqdm

# ======= 默认路径/超参 =======
NORMAL_DIR = 'E:\ZJU\AIOps\Projects\TraDNN\dataset\SplitTrace06-07/2025-06-07_normal_traces.csv'
SERVICE_DIR = 'E:\ZJU\AIOps\Projects\TraDNN\dataset\SplitTrace06-07/2025-06-07_service.csv'
NODE_DIR    = 'E:\ZJU\AIOps\Projects\TraDNN\dataset\SplitTrace06-07/2025-06-07_node.csv'
OUT_DIR     = 'dataset/aiops_06-07_2e5'

# 新的默认参数
TOTAL_TRACES_DEFAULT = 60000        # 总trace数量
TRAIN_RATIO = 0.7                    # 训练集比例
VAL_RATIO = 0.1                      # 验证集比例  
TEST_RATIO = 0.2                     # 测试集比例
TEST_NORMAL_RATIO = 0.9              # 测试集中正常数据比例
TEST_FAULT_RATIO = 0.1               # 测试集中异常数据比例
TEST_SVC_NODE_RATIO = 0.5            # 测试异常中服务异常比例

SEED  = 2025
MIN_TRACE_SPANS = 2                  # 丢弃单span trace

# 允许的细类（7+3）
SERVICE_FAULTS = {
    "code error","dns error","cpu stress","memory stress",
    "network corrupt","network delay","network loss",
}
NODE_FAULTS = {
    "node cpu stress","node memory stress","node disk fill",
}
IGNORE_SERVICE = {"pod failure","pod kill","target port misconfig"}

# 常见别名归一
SYN = {
    "network-loss":"network loss","network_loss":"network loss",
    "network-delay":"network delay","network delay(s)":"network delay",
    "dns-error":"dns error","code-error":"code error",
    "cpu-stress":"cpu stress","memory-stress":"memory stress",
    "node cpu":"node cpu stress","node memory":"node memory stress","node disk":"node disk fill",
}

# ======= 工具函数 =======
# 过滤
def filter_short_traces(df: pd.DataFrame, trace_col: str, min_spans: int) -> pd.DataFrame:
    """过滤掉span数量少于min_spans的trace"""
    if df.empty:
        return df

    print(f"    过滤前: {df[trace_col].nunique()} traces")

    # 计算每个trace的span数量
    trace_span_counts = df.groupby(trace_col).size()

    # 找出span数量足够的trace
    valid_traces = trace_span_counts[trace_span_counts >= min_spans].index

    # 过滤数据
    filtered_df = df[df[trace_col].isin(valid_traces)].copy()

    print(f"    过滤后: {filtered_df[trace_col].nunique()} traces")
    print(f"    过滤掉 {len(trace_span_counts) - len(valid_traces)} 个单span trace")

    # 统计span数量分布
    span_dist = trace_span_counts.value_counts().sort_index()
    print(f"    Span数量分布: {dict(span_dist.head(10))}")  # 显示前10个

    return filtered_df

def md5mod(s: str, mod=100) -> int:
    h = hashlib.md5(s.strip().encode("utf-8")).hexdigest()
    return int(h[:8], 16) % mod

def url_template(u: str) -> str:
    if not isinstance(u, str): return "NA"
    core = u.split("?")[0].split("#")[0]
    core = core.replace("//","/").rstrip("/")
    return core or "NA"

def norm_fault(x: Optional[str]) -> Optional[str]:
    if x is None or pd.isna(x): 
        return None
    s = str(x).strip().lower()
    # 处理空字符串和"nan"字符串
    if s in ["", "nan", "none", "null"]:
        return None
    return SYN.get(s, s)

def make_api_key(svc: str, url_tmpl: str) -> str:
    return f"{str(svc)}||{str(url_tmpl)}"

# ======= IO =======
def load_csv(path: str) -> pd.DataFrame:
    print(f"    加载文件: {path}")
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        print(f"    使用python引擎读取: {e}")
        df = pd.read_csv(path, engine="python")
    
    # 检查必要的列
    required_cols = ["TraceID", "SpanId", "ParentID", "ServiceName", "StartTimeMs", "EndTimeMs"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"    警告: 缺少列 {missing_cols}")
    
    # 确保有fault_type列
    if "fault_type" not in df.columns:
        df["fault_type"] = np.nan
    
    # 数据类型转换
    for col in ["StartTimeMs", "EndTimeMs", "HttpStatusCode", "StatusCode"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # 计算延迟
    if "StartTimeMs" in df.columns and "EndTimeMs" in df.columns:
        df["lat_ms"] = (df["EndTimeMs"] - df["StartTimeMs"]).astype(float).clip(lower=0)
    else:
        df["lat_ms"] = 0.0
    
    # 处理故障类型
    df["fault_type"] = df["fault_type"].apply(norm_fault)
    df["url_tmpl"] = df["URL"].astype(str).apply(url_template) if "URL" in df.columns else "NA"
    df["_node"] = df.get("NodeName", "").fillna("").astype(str) if "NodeName" in df.columns else ""
    
    print(f"    加载完成: {df['TraceID'].nunique() if 'TraceID' in df.columns else 0} traces")
    return df

def dump_jsonl(path: str, items: List[dict], desc: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in tqdm(items, total=len(items), desc=desc, ncols=100):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ======= 轻量"滚词" =======
def roll_vocabs(df: pd.DataFrame,
                cols: Dict[str,str],
                api_vocab: Dict[str,int],
                status_vocab: Dict[int,int],
                node_vocab: Dict[str,int],
                service_vocab: Dict[str,int],
                min_trace_size: int = MIN_TRACE_SPANS,
                desc: str = "roll vocabs"):
    if df.empty:
        return
        
    trace_col, span_col = cols["trace"], cols["span"]
    svc_col= cols["svc"]
    nunique = int(df[trace_col].astype(str).nunique())
    pbar = tqdm(total=nunique, desc=desc, ncols=100)
    for tid, g in df.groupby(trace_col, sort=False):
        if len(g) < min_trace_size:
            pbar.update(1); continue
        # API / status / node / service
        svc = g[svc_col].astype(str).fillna("NA").tolist()
        url = g["url_tmpl"].astype(str).tolist()
        for s, u in zip(svc, url):
            key = make_api_key(s, u)
            if key not in api_vocab:
                api_vocab[key] = len(api_vocab) + 1
        # 状态取 HttpStatusCode
        if "HttpStatusCode" in g.columns:
            stat = pd.to_numeric(g["HttpStatusCode"], errors="coerce").fillna(0).astype(int).tolist()
            for sc in stat:
                if sc not in status_vocab:
                    status_vocab[sc] = len(status_vocab) + 1
        nodes = g["_node"].astype(str).tolist()
        for nd in nodes:
            if nd not in node_vocab:
                node_vocab[nd] = len(node_vocab) + 1
        svcs = g[svc_col].astype(str).fillna("NA").tolist()
        for s in svcs:
            if s not in service_vocab:
                service_vocab[s] = len(service_vocab) + 1
        pbar.update(1)
    pbar.close()

# ======= 记录构建 =======
def build_records(df: pd.DataFrame, cols: Dict[str,str],
                  api_vocab: Dict[str,int], status_vocab: Dict[int,int], node_vocab: Dict[str,int],
                  fixed_c3: Optional[int], fault_type_col: str,
                  freeze_vocab: bool,
                  min_trace_size: int = MIN_TRACE_SPANS,
                  service_vocab: Optional[Dict[str,int]] = None,
                  desc: str = "build records") -> List[dict]:
    if df.empty:
        return []
        
    trace_col = cols["trace"]; span_col=cols["span"]; parent_col=cols["parent"]
    svc_col=cols["svc"]; st_col=cols["start"]; et_col=cols["end"]

    records: List[dict] = []
    nunique = int(df[trace_col].astype(str).nunique())
    pbar = tqdm(total=nunique, desc=desc, ncols=100)

    for tid, g in df.groupby(trace_col, sort=False):
        g = g.sort_values(by=[st_col, et_col, span_col], kind="mergesort").reset_index(drop=True)
        # if len(g) < min_trace_size:
        #     pbar.update(1); continue

        # 词表 & id 映射
        n = len(g)
        api_ids = np.zeros(n, dtype=np.int64)
        stat_ids= np.zeros(n, dtype=np.int64)
        node_ids= np.zeros(n, dtype=np.int64)
        service_ids = np.zeros(n, dtype=np.int64) if service_vocab is not None else None

        for i, row in g.iterrows():
            api_key = make_api_key(row[svc_col], row["url_tmpl"])
            if not freeze_vocab and api_key not in api_vocab:
                api_vocab[api_key] = len(api_vocab) + 1
            api_ids[i] = api_vocab.get(api_key, 0)

            # 状态取 HttpStatusCode；NaN→0
            status = 0
            if "HttpStatusCode" in g.columns and not pd.isna(row["HttpStatusCode"]):
                status = int(row["HttpStatusCode"])
            if not freeze_vocab and status not in status_vocab:
                status_vocab[status] = len(status_vocab) + 1
            stat_ids[i] = status_vocab.get(status, 0)

            node = str(row["_node"])
            if not freeze_vocab and node not in node_vocab:
                node_vocab[node] = len(node_vocab) + 1
            node_ids[i] = node_vocab.get(node, 0)

            if service_vocab is not None:
                svc = str(row[svc_col]) if pd.notna(row[svc_col]) else "NA"
                if not freeze_vocab and svc not in service_vocab:
                    service_vocab[svc] = len(service_vocab) + 1
                service_ids[i] = service_vocab.get(svc, 0)

        # parent 索引
        id_to_idx = {str(sid): i for i, sid in enumerate(g[span_col].astype(str).tolist())}
        parent_idx = []
        for pid in g[parent_col].astype(str).tolist():
            j = id_to_idx.get(pid, None)
            parent_idx.append(-1 if (j is None or pid in ["", "nan", "NaN"]) else j)

        # 儿子表
        children = [[] for _ in range(n)]
        for c, p in enumerate(parent_idx):
            if p >= 0:
                children[p].append(c)

        # 根集合（允许多根）
        roots = [i for i,p in enumerate(parent_idx) if p < 0]
        if not roots:
            roots = [0]  # 默认第一个为根

        # 全覆盖 DFS
        order: List[int] = []
        visited = [False]*n
        for r in sorted(roots, key=lambda j: (float(g.loc[j, st_col]) if pd.notna(g.loc[j, st_col]) else 0.0)):
            if visited[r]: continue
            stack=[r]
            while stack:
                u=stack.pop()
                if visited[u]: continue
                visited[u]=True
                order.append(u)
                for v in reversed(children[u]):
                    if not visited[v]:
                        stack.append(v)
        if len(order) < n:
            for i in range(n):
                if not visited[i]: order.append(i)

        pos_map = {i:p for p,i in enumerate(order)}
        depth=[0]*n
        for u in order:
            p=parent_idx[u]
            depth[u] = 0 if p<0 else (depth[p]+1)

        # 边（父子）
        edges = [(p,c) for c,p in enumerate(parent_idx) if p>=0]

        # 弱监督 rca: 最大延迟节点
        lat = g["lat_ms"].values.astype(float)
        rca_idx = int(np.argmax(lat)) if n>0 else 0

        # 标签：y_bin / y_c3 / fault_type
        ft = g[fault_type_col].iloc[0] if fault_type_col in g.columns else None
        ft = norm_fault(ft) if isinstance(ft, str) else None
        if ft in IGNORE_SERVICE:
            ft = None

        if isinstance(ft, str) and ft.strip():
            if ft in SERVICE_FAULTS:
                y_bin, y_c3 = 1, 1
            elif ft in NODE_FAULTS:
                y_bin, y_c3 = 1, 2
            else:
                y_bin, y_c3 = 0, 0
        else:
            y_bin, y_c3 = 0, 0

        nodes=[]
        for i in range(n):
            nodes.append({
                "api_id": int(api_ids[i]),
                "status_id": int(stat_ids[i]),
                "node_id": int(node_ids[i]),
                "latency_ms": float(lat[i]) if not np.isnan(lat[i]) else 0.0,
                "start_ms": float(g.loc[i, st_col]) if pd.notna(g.loc[i, st_col]) else 0.0,
                "end_ms": float(g.loc[i, et_col]) if pd.notna(g.loc[i, et_col]) else 0.0,
                "service": (str(g.loc[i, svc_col]) if pd.notna(g.loc[i, svc_col]) else "NA"),
                "url_tmpl": str(g.loc[i, "url_tmpl"]),
                "pos": int(pos_map.get(i, i)),
                "depth": int(depth[i]),
            })

        records.append({
            "trace_id": str(tid),
            "nodes": nodes,
            "edges": edges,
            "dfs_order": order,
            "y_bin": int(y_bin),
            "y_c3": int(y_c3),
            "fault_type": (ft if (ft and ft!="nan") else None),
            "rca_idx": int(rca_idx),
        })

        pbar.update(1)

    pbar.close()
    return records

# ======= 新的数据分配函数 =======
def allocate_traces_by_ratio(total_traces: int, train_ratio: float, val_ratio: float, test_ratio: float,
                           test_normal_ratio: float, test_fault_ratio: float, test_svc_node_ratio: float):
    """根据比例计算各部分的trace数量"""
    # 验证比例总和
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "train+val+test比例总和必须为1"
    assert abs(test_normal_ratio + test_fault_ratio - 1.0) < 1e-6, "测试集正常+异常比例总和必须为1"
    
    # 计算基础数量
    n_train = int(total_traces * train_ratio)
    n_val = int(total_traces * val_ratio)
    n_test = int(total_traces * test_ratio)
    
    # 调整总数以避免因取整导致的误差
    total_allocated = n_train + n_val + n_test
    if total_allocated < total_traces:
        n_test += (total_traces - total_allocated)
    
    # 计算测试集细分
    n_test_normal = int(n_test * test_normal_ratio)
    n_test_fault = n_test - n_test_normal
    n_test_svc = int(n_test_fault * test_svc_node_ratio)
    n_test_node = n_test_fault - n_test_svc
    
    return {
        'train': n_train,
        'val': n_val, 
        'test_total': n_test,
        'test_normal': n_test_normal,
        'test_fault': n_test_fault,
        'test_svc': n_test_svc,
        'test_node': n_test_node
    }

# ======= 新的数据采样函数 =======
def sample_traces_by_type(df: pd.DataFrame, trace_col: str, n_samples: int, fault_type: str = None, seed: int = SEED):
    """根据故障类型采样指定数量的trace"""
    if n_samples <= 0 or df.empty:
        return pd.DataFrame()
    
    # 获取唯一的trace ID
    if fault_type:
        # 采样特定故障类型
        available_traces = df[df['fault_type'] == fault_type][trace_col].unique()
        print(f"    故障类型 '{fault_type}': 找到 {len(available_traces)} 个traces")
    else:
        # 采样正常数据（无故障类型或故障类型为None）
        normal_mask = df['fault_type'].isna() | (df['fault_type'].isnull())
        available_traces = df[normal_mask][trace_col].unique()
        print(f"    正常数据: 找到 {len(available_traces)} 个traces")
    
    if len(available_traces) == 0:
        print(f"    警告: 没有找到{fault_type if fault_type else '正常'}类型的trace")
        return pd.DataFrame()
    
    # 随机采样
    random.seed(seed)
    if len(available_traces) <= n_samples:
        selected_traces = available_traces
        print(f"    注意: 只有{len(available_traces)}个{fault_type if fault_type else '正常'}trace可用，小于请求的{n_samples}个")
    else:
        selected_traces = random.sample(list(available_traces), n_samples)
    
    # 返回对应的数据
    return df[df[trace_col].isin(selected_traces)]

# ======= 主入口 =======
def main():
    ap = argparse.ArgumentParser(description="独立阶段A数据处理 - 训练/验证:100%正常, 测试:90%正常+10%异常")
    ap.add_argument("--normal", default=NORMAL_DIR, help="正常数据CSV路径")
    ap.add_argument("--service_fault", default=SERVICE_DIR, help="服务故障数据CSV路径")
    ap.add_argument("--node_fault", default=NODE_DIR, help="节点故障数据CSV路径")
    ap.add_argument("--outdir", default=OUT_DIR, help="输出目录")
    ap.add_argument("--seed", type=int, default=SEED, help="随机种子")
    
    # 新的比例参数
    ap.add_argument("--total_traces", type=int, default=TOTAL_TRACES_DEFAULT, 
                   help="总trace数量")
    ap.add_argument("--train_ratio", type=float, default=TRAIN_RATIO, 
                   help="训练集比例")
    ap.add_argument("--val_ratio", type=float, default=VAL_RATIO, 
                   help="验证集比例")
    ap.add_argument("--test_ratio", type=float, default=TEST_RATIO, 
                   help="测试集比例")
    ap.add_argument("--test_normal_ratio", type=float, default=TEST_NORMAL_RATIO,
                   help="测试集中正常数据比例")
    ap.add_argument("--test_svc_node_ratio", type=float, default=TEST_SVC_NODE_RATIO,
                   help="测试异常中服务异常比例")
    
    ap.add_argument("--min_trace_spans", type=int, default=MIN_TRACE_SPANS,
                   help="最小trace span数量")

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    t0 = time.time()
    print("=== 独立阶段A数据处理 ===")
    print(f"[1/6] 读取CSV文件...")
    
    # 加载数据
    df_normal = load_csv(args.normal)
    df_service = load_csv(args.service_fault)
    df_node = load_csv(args.node_fault)

    # 立即过滤单span trace
    print(f"[1.5/6] 过滤单span trace (min_spans={args.min_trace_spans})...")
    df_normal = filter_short_traces(df_normal, 'TraceID', args.min_trace_spans)
    df_service = filter_short_traces(df_service, 'TraceID', args.min_trace_spans)
    df_node = filter_short_traces(df_node, 'TraceID', args.min_trace_spans)

    # 归一化故障类型并筛选
    df_service["fault_type"] = df_service["fault_type"].apply(norm_fault)
    df_node["fault_type"] = df_node["fault_type"].apply(norm_fault)
    
    # 只保留我们关注的故障类型
    df_service = df_service[df_service["fault_type"].isin(SERVICE_FAULTS)]
    df_node = df_node[df_node["fault_type"].isin(NODE_FAULTS)]

    print(f"  正常数据 (过滤后): {df_normal['TraceID'].nunique()} traces")
    print(f"  服务故障 (过滤后): {df_service['TraceID'].nunique()} traces")
    print(f"  节点故障 (过滤后): {df_node['TraceID'].nunique()} traces")

    # 计算数据分配
    allocations = allocate_traces_by_ratio(
        args.total_traces, args.train_ratio, args.val_ratio, args.test_ratio,
        args.test_normal_ratio, 1 - args.test_normal_ratio, args.test_svc_node_ratio
    )
    
    print(f"[2/6] 数据分配 (总计: {args.total_traces} traces)...")
    print(f"  训练集: {allocations['train']} traces (100% 正常)")
    print(f"  验证集: {allocations['val']} traces (100% 正常)") 
    print(f"  测试集: {allocations['test_total']} traces")
    print(f"    - 正常: {allocations['test_normal']} traces ({args.test_normal_ratio*100}%)")
    print(f"    - 异常: {allocations['test_fault']} traces ({(1-args.test_normal_ratio)*100}%)")
    print(f"      * 服务异常: {allocations['test_svc']} traces")
    print(f"      * 节点异常: {allocations['test_node']} traces")

    # 采样数据
    print(f"[3/6] 采样数据...")
    
    # 训练集和验证集：100%正常
    df_train = sample_traces_by_type(df_normal, 'TraceID', allocations['train'], fault_type=None, seed=args.seed)
    df_val = sample_traces_by_type(df_normal, 'TraceID', allocations['val'], fault_type=None, seed=args.seed)
    
    # 测试集：正常 + 服务异常 + 节点异常
    df_test_normal = sample_traces_by_type(df_normal, 'TraceID', allocations['test_normal'], fault_type=None, seed=args.seed)

    # 从服务故障中采样服务异常
    df_test_svc = pd.DataFrame()
    if not df_service.empty and allocations['test_svc'] > 0:
        print(f"    服务故障采样: 需要 {allocations['test_svc']} 个traces")

        # 收集所有可用的服务故障trace
        available_svc_traces = {}
        total_svc_traces = 0

        for fault_type in SERVICE_FAULTS:
            traces = df_service[df_service['fault_type'] == fault_type]['TraceID'].unique()
            if len(traces) > 0:
                available_svc_traces[fault_type] = list(traces)
                total_svc_traces += len(traces)
                print(f"      {fault_type}: {len(traces)} 个traces")

        # 如果总可用数小于需求，调整需求
        actual_svc_needed = min(allocations['test_svc'], total_svc_traces)
        if actual_svc_needed < allocations['test_svc']:
            print(f"      警告: 服务故障trace不足，只能采样 {actual_svc_needed} 个")

        # 按比例分配每个故障类型的采样数量
        collected_traces = []
        for fault_type, traces in available_svc_traces.items():
            # 计算这个故障类型应该贡献的比例
            proportion = len(traces) / total_svc_traces
            needed_for_type = int(actual_svc_needed * proportion)

            # 确保至少采样1个，且不超过可用数量
            needed_for_type = max(1, min(needed_for_type, len(traces)))

            print(f"      从 {fault_type} 采样 {needed_for_type} 个traces")

            # 随机采样
            sampled = random.sample(traces, needed_for_type)
            collected_traces.extend(sampled)

        # 如果总数不足，从剩余trace中补充
        if len(collected_traces) < actual_svc_needed:
            remaining_needed = actual_svc_needed - len(collected_traces)
            print(f"      需要补充 {remaining_needed} 个traces")

            # 收集所有未使用的trace
            unused_traces = []
            for fault_type, traces in available_svc_traces.items():
                used_traces = set(collected_traces)
                unused = [t for t in traces if t not in used_traces]
                unused_traces.extend(unused)

            if unused_traces:
                # 随机选择剩余的trace
                if len(unused_traces) <= remaining_needed:
                    collected_traces.extend(unused_traces)
                else:
                    collected_traces.extend(random.sample(unused_traces, remaining_needed))

        # 获取对应的数据
        if collected_traces:
            df_test_svc = df_service[df_service['TraceID'].isin(collected_traces)]
            print(f"      实际采样服务故障: {len(collected_traces)} 个traces")

    # 从节点故障中采样节点异常
    df_test_node = pd.DataFrame()
    if not df_node.empty and allocations['test_node'] > 0:
        print(f"    节点故障采样: 需要 {allocations['test_node']} 个traces")

        # 收集所有可用的节点故障trace
        available_node_traces = {}
        total_node_traces = 0

        for fault_type in NODE_FAULTS:
            traces = df_node[df_node['fault_type'] == fault_type]['TraceID'].unique()
            if len(traces) > 0:
                available_node_traces[fault_type] = list(traces)
                total_node_traces += len(traces)
                print(f"      {fault_type}: {len(traces)} 个traces")

        # 如果总可用数小于需求，调整需求
        actual_node_needed = min(allocations['test_node'], total_node_traces)
        if actual_node_needed < allocations['test_node']:
            print(f"      警告: 节点故障trace不足，只能采样 {actual_node_needed} 个")

        # 按比例分配每个故障类型的采样数量
        collected_traces = []
        for fault_type, traces in available_node_traces.items():
            # 计算这个故障类型应该贡献的比例
            proportion = len(traces) / total_node_traces
            needed_for_type = int(actual_node_needed * proportion)

            # 确保至少采样1个，且不超过可用数量
            needed_for_type = max(1, min(needed_for_type, len(traces)))

            print(f"      从 {fault_type} 采样 {needed_for_type} 个traces")

            # 随机采样
            sampled = random.sample(traces, needed_for_type)
            collected_traces.extend(sampled)

        # 如果总数不足，从剩余trace中补充
        if len(collected_traces) < actual_node_needed:
            remaining_needed = actual_node_needed - len(collected_traces)
            print(f"      需要补充 {remaining_needed} 个traces")

            # 收集所有未使用的trace
            unused_traces = []
            for fault_type, traces in available_node_traces.items():
                used_traces = set(collected_traces)
                unused = [t for t in traces if t not in used_traces]
                unused_traces.extend(unused)

            if unused_traces:
                # 随机选择剩余的trace
                if len(unused_traces) <= remaining_needed:
                    collected_traces.extend(unused_traces)
                else:
                    collected_traces.extend(random.sample(unused_traces, remaining_needed))

        # 获取对应的数据
        if collected_traces:
            df_test_node = df_node[df_node['TraceID'].isin(collected_traces)]
            print(f"      实际采样节点故障: {len(collected_traces)} 个traces")
    
    # 合并测试集
    test_dfs = []
    if not df_test_normal.empty:
        test_dfs.append(df_test_normal)
    if not df_test_svc.empty:
        test_dfs.append(df_test_svc)
    if not df_test_node.empty:
        test_dfs.append(df_test_node)
        
    if test_dfs:
        df_test = pd.concat(test_dfs, ignore_index=True)
    else:
        df_test = pd.DataFrame()
    
    print(f"  实际采样结果:")
    print(f"    训练集: {df_train['TraceID'].nunique() if not df_train.empty else 0} traces")
    print(f"    验证集: {df_val['TraceID'].nunique() if not df_val.empty else 0} traces") 
    print(f"    测试集: {df_test['TraceID'].nunique() if not df_test.empty else 0} traces")
    if not df_test.empty:
        test_normal_count = len(df_test[df_test['fault_type'].isna()]['TraceID'].unique()) if 'fault_type' in df_test.columns else 0
        test_svc_count = len(df_test[df_test['fault_type'].isin(SERVICE_FAULTS)]['TraceID'].unique()) if 'fault_type' in df_test.columns else 0
        test_node_count = len(df_test[df_test['fault_type'].isin(NODE_FAULTS)]['TraceID'].unique()) if 'fault_type' in df_test.columns else 0
        print(f"      - 正常: {test_normal_count} traces")
        print(f"      - 服务异常: {test_svc_count} traces")
        print(f"      - 节点异常: {test_node_count} traces")

    # 构建词表（仅使用训练集）
    print(f"[4/6] 用训练集构建词表...")
    cols = {"trace":"TraceID","span":"SpanId","parent":"ParentID",
            "svc":"ServiceName","url":"URL","start":"StartTimeMs","end":"EndTimeMs"}
    
    api_vocab: Dict[str,int] = {}
    status_vocab: Dict[int,int] = {}
    node_vocab: Dict[str,int] = {}
    service_vocab: Dict[str,int] = {}
    
    if not df_train.empty:
        roll_vocabs(df_train, cols, api_vocab, status_vocab, node_vocab, service_vocab,
                    min_trace_size=args.min_trace_spans,
                    desc="[4/6] 构建词表")
        print(f"  词表大小: api={len(api_vocab)} status={len(status_vocab)} node={len(node_vocab)} service={len(service_vocab)}")
    else:
        print("  警告: 训练集为空，无法构建词表")

    # 构建记录
    print(f"[5/6] 构建JSONL记录...")
    
    rec_train = build_records(df_train, cols, api_vocab, status_vocab, node_vocab, fixed_c3=None,
                              fault_type_col="fault_type", freeze_vocab=True,
                              min_trace_size=args.min_trace_spans, service_vocab=service_vocab,
                              desc="构建训练集") if not df_train.empty else []
    rec_val   = build_records(df_val, cols, api_vocab, status_vocab, node_vocab, fixed_c3=None,
                              fault_type_col="fault_type", freeze_vocab=True,
                              min_trace_size=args.min_trace_spans, service_vocab=service_vocab,
                              desc="构建验证集") if not df_val.empty else []
    rec_test  = build_records(df_test, cols, api_vocab, status_vocab, node_vocab, fixed_c3=None,
                              fault_type_col="fault_type", freeze_vocab=True,
                              min_trace_size=args.min_trace_spans, service_vocab=service_vocab,
                              desc="构建测试集") if not df_test.empty else []

    # 输出文件
    print(f"[6/6] 写入文件...")
    if rec_train:
        dump_jsonl(os.path.join(args.outdir, "A_train_normal.jsonl"), rec_train, desc="训练集")
    if rec_val:
        dump_jsonl(os.path.join(args.outdir, "A_val.jsonl"), rec_val, desc="验证集")
    if rec_test:
        dump_jsonl(os.path.join(args.outdir, "A_test.jsonl"), rec_test, desc="测试集")
        dump_jsonl(os.path.join(args.outdir, "unified_test.jsonl"), rec_test, desc="统一测试集")

    # 写词表
    type_names = sorted(list(SERVICE_FAULTS)) + sorted(list(NODE_FAULTS))
    vocab = {
        "api_vocab_size":     int(len(api_vocab)+1),
        "status_vocab_size":  int(len(status_vocab)+1),
        "node_vocab_size":    int(len(node_vocab)+1),
        "service_vocab_size": int(len(service_vocab)+1),
        "type_names": type_names,
        "ctx_dim": 0,
        "split_seed": int(args.seed),
        "notes": f"阶段A独立处理 - 训练/验证:100%正常, 测试:{args.test_normal_ratio*100}%正常+{(1-args.test_normal_ratio)*100}%异常"
    }
    with open(os.path.join(args.outdir,"vocab.json"),"w",encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    # 统计信息
    print(f"\n=== 处理完成 ===")
    print(f"输出目录: {args.outdir}")
    print(f"数据分布:")
    print(f"  训练集: {len(rec_train)} traces (100% 正常)")
    print(f"  验证集: {len(rec_val)} traces (100% 正常)")
    
    if rec_test:
        test_normal = sum(1 for r in rec_test if r.get("y_bin", 0) == 0)
        test_fault = sum(1 for r in rec_test if r.get("y_bin", 0) == 1)
        test_svc = sum(1 for r in rec_test if r.get("fault_type") in SERVICE_FAULTS)
        test_node = sum(1 for r in rec_test if r.get("fault_type") in NODE_FAULTS)
        
        print(f"  测试集: {len(rec_test)} traces")
        print(f"    - 正常: {test_normal} traces ({test_normal/len(rec_test)*100:.1f}%)")
        print(f"    - 异常: {test_fault} traces ({test_fault/len(rec_test)*100:.1f}%)")
        if test_fault > 0:
            print(f"      * 服务异常: {test_svc} traces ({test_svc/test_fault*100:.1f}%)")
            print(f"      * 节点异常: {test_node} traces ({test_node/test_fault*100:.1f}%)")
    
    # 故障类型统计
    if rec_test:
        fault_counts = Counter([r.get("fault_type") for r in rec_test if r.get('fault_type')])
        if fault_counts:
            print(f"  测试集故障类型分布: {dict(fault_counts)}")
    
    print(f"总耗时: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()