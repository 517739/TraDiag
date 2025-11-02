# -*- coding: utf-8 -*-
"""
TraDiag v3 训练脚本（稳态+高吞吐版）
- Stage A: 无监督 VAE（normal-only 训练）
- 可选：Stage B（监督细类）、Stage C（无监督 RCA）
- 关键：AMP、梯度累积、DataLoader 多进程/预取、节点预算打包、KL夹紧、KL/结构热身、NaN守卫

[PATCH 摘要]
1) 新增按 v3 数据文件读取（Stage A 用 A_*，Stage B 用 B_*），并保留原有评估/保存流程。
2) 每阶段单独将 val/test 规模按 8:1:1 （以 Trace 为单位）相对各自 train 收敛，可叠加上限。
3) DataLoader kwargs 统一用 utils_v4.loader_kwargs(args)，避免 Windows 下 prefetch_factor 报错。
4) [新增] --a_loss 开关（mse/nll），Stage-A 训练 & 评估 & Stage-C 统一口径。
"""

import os, json, argparse, random
from datetime import datetime
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# [保持原有引用]
from model_v5 import TraceUnifiedModelV3
from utils_v5 import *

# [PATCH] NLL 常数项
import math
LOG_TWO_PI = math.log(2.0 * math.pi)

def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

# ---------------- 节点预算 BatchSampler：按“总节点数”打包 ----------------
class NodeBudgetBatchSampler(tud.Sampler):
    """按总节点预算打包 batch，提升显存/吞吐利用率，避免批间规模差异过大。"""
    def __init__(self, base_indices: List[int], node_sizes: List[int], node_budget: int, shuffle: bool = True):
        self.base_indices = list(base_indices)
        self.node_sizes = list(node_sizes)
        self.node_budget = int(node_budget)
        self.shuffle = shuffle
        assert len(self.base_indices) == len(self.node_sizes) and self.node_budget > 0

    def __iter__(self):
        order = np.random.permutation(len(self.base_indices)) if self.shuffle else np.arange(len(self.base_indices))
        cur_nodes = 0
        buf = []
        for j in order:
            idx = self.base_indices[j]
            n   = self.node_sizes[j] if self.node_sizes[j] > 0 else 1
            if buf and (cur_nodes + n > self.node_budget):
                yield buf
                buf = []; cur_nodes = 0
            buf.append(idx); cur_nodes += n
        if buf: yield buf

    def __len__(self):
        total = max(1, sum(max(1, n) for n in self.node_sizes))
        return max(1, total // self.node_budget)

def _subset_head(ds, k: int, seed: int = 42):
    k = max(0, min(k, len(ds)))
    if k == len(ds):
        return ds
    rng = random.Random(seed)          # 固定随机种子
    indices = list(range(len(ds)))
    rng.shuffle(indices)           # 原地打乱
    return Subset(ds, sorted(indices[:k]))  # 取前k（排序保持小序号在前，DGL batch 更快）

# [PATCH] —— 以 train 为基准，给出 8:1:1 的目标 val/test ——
def _enforce_811_target(train_len: int, val_len: int, test_len: int):
    """
    目标：val≈train/8, test≈train/8；不能超过现有 val/test 可用规模。
    注意：这里的 train/val/test 计数均为“Trace”。
    """
    tgt_v = min(val_len,  train_len // 8)
    tgt_t = min(test_len, train_len // 8)
    return tgt_v, tgt_t

# ---------------- 主流程 ----------------
def main():
    p = argparse.ArgumentParser()
    # 基本
    p.add_argument('--data_root', type=str, default='dataset/aiops_v4_1e52e3')
    p.add_argument('--report_dir', type=str, default='dataset/aiops_v4_1e52e3/1029gatgcntl')

    p.add_argument('--device', type=str, default='cuda:3')
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--seed', type=int, default=2025)

    # 加速/规模控制（以 Trace 为单位）
    p.add_argument('--num_workers', type=int, default=16)
    p.add_argument('--pin_memory', type=int, default=1)
    p.add_argument('--prefetch_factor', type=int, default=4)
    p.add_argument('--train_limit', type=int, default=0, help='限制A阶段训练样本数（normal-only子集），0=不限')
    p.add_argument('--val_limit', type=int, default=0, help='限制评估子集（Trace 数量）')
    p.add_argument('--test_limit', type=int, default=0, help='限制评估子集（Trace 数量）')
    p.add_argument('--steps_per_epoch', type=int, default=0, help='每个epoch最多训练多少个batch（0=全量）')
    p.add_argument('--fast', type=int, default=0, help='快速小规模试跑预设（覆盖若干参数）')
    p.add_argument('--node_budget', type=int, default=20000, help='按总节点预算打包batch（优先于 --batch；0=不用）')

    # 阶段与轮数
    p.add_argument('--epochs_a', type=int, default=10)
    p.add_argument('--enable_b', type=int, default=0)
    p.add_argument('--epochs_b', type=int, default=25)
    p.add_argument('--enable_c', type=int, default=1)

    # 模型/优化
    p.add_argument('--emb', type=int, default=64)
    p.add_argument('--gc_hidden', type=int, default=128)
    p.add_argument('--amp', type=int, default=1)
    p.add_argument('--grad_accum', type=int, default=1)
    p.add_argument('--max_nodes', type=int, default=0, help='A阶段训练丢弃节点数>此值的图；0=不限')
    p.add_argument('--lr', type=float, default=1e-3)

    # 模型选择
    p.add_argument('--call_conv', choices=['gcn', 'gat'], default='gat',
                   help='调用图分支的图卷积类型（gcn/gat）')
    p.add_argument('--host_conv', choices=['gcn', 'gat'], default='gcn',
                   help='主机共址分支的图卷积类型（gcn/gat）')
    p.add_argument('--gat_heads', type=int, default=4, help='GAT 多头数（concat=False，无需整除）')
    p.add_argument('--gat_feat_drop', type=float, default=0.20, help='GAT 特征 dropout')
    p.add_argument('--gat_attn_drop', type=float, default=0.10, help='GAT 注意力 dropout')

    # 早停
    p.add_argument('--early_stop_a', type=int, default=1)
    p.add_argument('--early_stop_b', type=int, default=1)
    p.add_argument('--patience_a', type=int, default=5)
    p.add_argument('--patience_b', type=int, default=5)
    p.add_argument('--delta_a', type=float, default=1e-4)
    p.add_argument('--delta_b', type=float, default=1e-4)

    # Stage A 超参
    p.add_argument('--topk', type=float, default=1.0)
    p.add_argument('--alpha_lat', type=float, default=3.0)
    p.add_argument('--beta_stat', type=float, default=0.5)
    p.add_argument('--beta_kl', type=float, default=0.5)
    p.add_argument('--struct_w', type=float, default=0.3)
    p.add_argument('--status_mask_p', type=float, default=0.15)
    # [PATCH] —— 新增 A 阶段损失口径开关（mse / nll）
    p.add_argument('--a_loss', type=str, default='nll', choices=['mse', 'nll'])

    # KL/结构头稳定化
    p.add_argument('--kl_clip', type=float, default=8.0, help='logvar clamp范围，避免exp溢出')
    p.add_argument('--kl_warmup_steps', type=int, default=4000, help='KL权重线性热身步数')
    p.add_argument('--struct_warmup_steps', type=int, default=2000, help='结构头权重线性热身步数')

    # normal-only 控制
    p.add_argument('--normal_max', type=int, default=0)
    p.add_argument('--normal_index', type=str, default='')
    p.add_argument('--normal_csv', type=str, default='')

    # [PATCH] —— v3 文件名 & 8:1:1 收敛控制（均以 Trace 为单位）——
    p.add_argument('--use_v3_splits', type=int, default=1, help='1=使用 A_*/B_* 文件；0=沿用 train/val/test')
    p.add_argument('--split_a_train', type=str, default='A_train_normal')
    p.add_argument('--split_a_val',   type=str, default='A_val')
    p.add_argument('--split_a_test',  type=str, default='A_test')
    p.add_argument('--split_b_train', type=str, default='B_train_fault')
    p.add_argument('--split_b_val',   type=str, default='B_val_fault')
    p.add_argument('--split_b_test',  type=str, default='B_test_fault')

    p.add_argument('--enforce_811_a', type=int, default=1, help='Stage A: 将 val/test 收敛到与 train 约 8:1:1（Trace）')
    p.add_argument('--enforce_811_b', type=int, default=1, help='Stage B: 将 val/test 收敛到与 train 约 8:1:1（Trace）')
    p.add_argument('--a_val_max',  type=int, default=0)
    p.add_argument('--a_test_max', type=int, default=0)
    p.add_argument('--b_val_max',  type=int, default=0)
    p.add_argument('--b_test_max', type=int, default=0)

    # === ADD: 评估模式与阈值选择策略 ===
    p.add_argument("--metric_mode", choices=["legacy", "accf1"], default="accf1",
                    help="legacy = 保留原始打印(0.5阈/argmax)；accf1 = 用验证集选阈值, 在测试集报 Acc/Prec/Rec/F1")
    p.add_argument("--thr_strategy", choices=["val_recall95", "val_maxf1", "youden"], default="val_maxf1",
                    help="阈值选择策略：召回≥0.95下F1最大 / 直接最大F1 / YoudenJ")
    p.add_argument("--print_legacy_also", action="store_true", default=True,
                    help="在 accf1 模式下，是否同时打印 legacy 报表（对齐老口径）")

    # [PATCH] —— Stage C 是否使用 unified_test（若存在）——
    p.add_argument('--use_unified_test', type=int, default=1)

    args = p.parse_args()
    os.makedirs(args.report_dir, exist_ok=True)

    # “快跑”预设（保留原有逻辑）
    if args.fast:
        args.batch = 1
        args.grad_accum = 4
        args.emb = min(args.emb, 48)
        args.gc_hidden = min(args.gc_hidden, 96)
        if args.normal_max == 0: args.normal_max = 30000
        if args.max_nodes   == 0: args.max_nodes   = 1500
        if args.steps_per_epoch == 0: args.steps_per_epoch = 2000
        args.amp = 0 # 用cpu跑的话设置为0
        args.num_workers = max(args.num_workers, 8)
        args.pin_memory = 1
        args.prefetch_factor = max(args.prefetch_factor, 2)
        args.struct_w = 0.0
        log(f"[FAST] 预设: batch={args.batch}, grad_accum={args.grad_accum}, emb={args.emb}, gc_hidden={args.gc_hidden}, "
            f"normal_max={args.normal_max}, max_nodes={args.max_nodes}, steps_per_epoch={args.steps_per_epoch}")

    # 种子
    if args.seed >= 0:
        random.seed(args.seed); np.random.seed(args.seed)
        torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    # 4090 优化
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        try: torch.set_float32_matmul_precision('medium')
        except Exception: pass

    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp==1 and torch.cuda.is_available()))

    # =======================
    #  数据：以 Trace 为单位的样本（v3 文件）
    # =======================
    log("加载 JSONL 数据 …")
    if args.use_v3_splits:
        # [PATCH] —— 直接用 v3 文件名 ——
        ds_a_tr = JSONLDataset(args.data_root, args.split_a_train, cache_size=100000)
        ds_a_va = JSONLDataset(args.data_root, args.split_a_val,   cache_size=50000)
        ds_a_te = JSONLDataset(args.data_root, args.split_a_test,  cache_size=50000)

        # Stage B 仅在 enable_b 时加载
        if args.enable_b:
            ds_b_tr = JSONLDataset(args.data_root, args.split_b_train, cache_size=100000)
            ds_b_va = JSONLDataset(args.data_root, args.split_b_val,   cache_size=50000)
            ds_b_te = JSONLDataset(args.data_root, args.split_b_test,  cache_size=50000)

        # 统一测试（可选）
        ds_u_te = None
        if args.use_unified_test:
            try:
                ds_u_te = JSONLDataset(args.data_root, 'unified_test', cache_size=50000)
            except Exception:
                ds_u_te = None

        log(f"[A][data] train={len(ds_a_tr)}  val={len(ds_a_va)}  test={len(ds_a_te)}")
        # 统计数据规模，写入 result.txt
        stats = summarize_datasets(args.data_root)
        result_path = os.path.join(args.report_dir, "result.txt")
        append_dataset_and_stageA_report(result_path, stats)

        if args.enable_b:
            log(f"[B][data] train={len(ds_b_tr)}  val={len(ds_b_va)}  test={len(ds_b_te)}")
        if ds_u_te is not None:
            log(f"[U][data] unified_test={len(ds_u_te)}")
    else:
        # 原始 train/val/test 入口（保留）
        ds_tr = JSONLDataset(args.data_root, 'train', cache_size=100000)
        ds_va = JSONLDataset(args.data_root, 'val',   cache_size=50000)
        ds_te = JSONLDataset(args.data_root, 'test',  cache_size=50000)
        log(f"train={len(ds_tr)}  val={len(ds_va)}  test={len(ds_te)}")
        # 为了后续代码统一接口，映射到 A 阶段变量（A=全量）
        ds_a_tr, ds_a_va, ds_a_te = ds_tr, ds_va, ds_te
        ds_b_tr = ds_b_va = ds_b_te = None
        ds_u_te = None

    # [PATCH] —— A/B 两阶段，分别按 8:1:1 收敛 val/test 的 Trace 规模 ——
    # 先处理 A
    if args.enforce_811_a:
        tgt_va, tgt_te = _enforce_811_target(len(ds_a_tr), len(ds_a_va), len(ds_a_te))
        # 叠加上限（如果传了）以及原有 limit
        if args.a_val_max:  tgt_va = min(tgt_va, int(args.a_val_max))
        if args.a_test_max: tgt_te = min(tgt_te, int(args.a_test_max))
        if args.val_limit:  tgt_va = min(tgt_va, int(args.val_limit))
        if args.test_limit: tgt_te = min(tgt_te, int(args.test_limit))
        ds_a_va_lmt = _subset_head(ds_a_va, tgt_va, seed=args.seed)
        ds_a_te_lmt = _subset_head(ds_a_te, tgt_te, seed=args.seed + 1)
        log(f"[A][ratio] train={len(ds_a_tr)}  val={len(ds_a_va_lmt)}  test={len(ds_a_te_lmt)}  (target≈8:1:1)")
    else:
        # 保持原逻辑：仅应用 val/test_limit
        va_idx = list(range(len(ds_a_va))); te_idx = list(range(len(ds_a_te)))
        ds_a_va_lmt, ds_a_te_lmt = ds_a_va, ds_a_te
        if args.val_limit and len(va_idx) > args.val_limit:
            ds_a_va_lmt = Subset(ds_a_va, va_idx[:args.val_limit])
            log(f"限制 A-val 集为 {len(ds_a_va_lmt)} (val_limit={args.val_limit})")
        if args.test_limit and len(te_idx) > args.test_limit:
            ds_a_te_lmt = Subset(ds_a_te, te_idx[:args.test_limit])
            log(f"限制 A-test 集为 {len(ds_a_te_lmt)} (test_limit={args.test_limit})")

    # 再处理 B
    if args.enable_b and (ds_b_tr is not None):
        if args.enforce_811_b:
            tgt_va, tgt_te = _enforce_811_target(len(ds_b_tr), len(ds_b_va), len(ds_b_te))
            if args.b_val_max:  tgt_va = min(tgt_va, int(args.b_val_max))
            if args.b_test_max: tgt_te = min(tgt_te, int(args.b_test_max))
            # 也叠加全局 val/test_limit（方便统一开关）
            if args.val_limit:  tgt_va = min(tgt_va, int(args.val_limit))
            if args.test_limit: tgt_te = min(tgt_te, int(args.test_limit))
            ds_b_va_lmt = _subset_head(ds_b_va, tgt_va, seed=args.seed)
            ds_b_te_lmt = _subset_head(ds_b_te, tgt_te, seed=args.seed + 1)
            log(f"[B][ratio] train={len(ds_b_tr)}  val={len(ds_b_va_lmt)}  test={len(ds_b_te_lmt)}  (target≈8:1:1)")
        else:
            ds_b_va_lmt, ds_b_te_lmt = ds_b_va, ds_b_te
    else:
        ds_b_tr = ds_b_va_lmt = ds_b_te_lmt = None

    # =============== 词表/模型部分（原样保留） ===============
    with open(os.path.join(args.data_root, 'vocab.json'), 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    api_sz    = int(vocab.get('api_vocab_size', 5000))
    status_sz = int(vocab.get('status_vocab_size', 10))
    node_sz   = int(vocab.get('node_vocab_size', 1000))
    type_sz   = int(vocab.get('type_names') and len(vocab['type_names']) or 10)
    ctx_dim   = int(vocab.get('ctx_dim', 0))
    type_names= vocab.get('type_names', None)

    model = TraceUnifiedModelV3(api_sz, status_sz, node_sz, type_sz,
                                ctx_dim=ctx_dim, emb=args.emb, gc_hidden=args.gc_hidden).to(dev)
    log(model.__class__.__name__ + " 已构建。")

    # =============== Stage A normal-only 选择（兼容 v3：A_train_normal 本身即 normal-only） ===============
    if args.use_v3_splits:
        normal_idx = list(range(len(ds_a_tr)))  # A_train_normal 本身已 normal-only
        log(f"[A][data] normal-only(train)={len(normal_idx)}")
    else:
        if args.normal_csv and not args.normal_index:
            out_idx = os.path.join(args.report_dir, 'normal_index.json')
            build_normal_index_from_csv(args.normal_csv, ds_a_tr.items, out_idx)
            args.normal_index = out_idx
            log(f"从 Normal.csv 生成 normal_index：{out_idx}")

        if args.normal_index and os.path.isfile(args.normal_index):
            normal_idx = [int(i) for i in json.load(open(args.normal_index, 'r', encoding='utf-8'))]
            normal_idx = [i for i in normal_idx if 0 <= i < len(ds_a_tr)]
            log(f"从 normal_index 读取 normal-only 数量：{len(normal_idx)}")
        else:
            normal_idx = [i for i, r in enumerate(ds_a_tr.items) if int(r.get('y_bin', 0)) == 0]
            log(f"train.jsonl 内部筛选 normal-only 数量：{len(normal_idx)}")

    # 过滤巨图（仅A训练集）
    if args.max_nodes and args.max_nodes > 0:
        get_item = (ds_a_tr.__getitem__ if hasattr(ds_a_tr, '__getitem__') else lambda i: ds_a_tr.items[i])
        normal_idx = [i for i in normal_idx if len(ds_a_tr.items[i]['nodes']) <= args.max_nodes]
        log(f"过滤超大图后 normal-only 数量：{len(normal_idx)} (max_nodes={args.max_nodes})")

    # 裁剪 & 限制（以 Trace 为单位）
    if args.normal_max and len(normal_idx) > args.normal_max:
        random.shuffle(normal_idx); normal_idx = normal_idx[:args.normal_max]
        log(f"裁剪 normal-only 数量至 {len(normal_idx)} (normal_max={args.normal_max})")
    if args.train_limit and len(normal_idx) > args.train_limit:
        random.shuffle(normal_idx); normal_idx = normal_idx[:args.train_limit]
        log(f"限制 normal-only 训练子集为 {len(normal_idx)} (train_limit={args.train_limit})")

    # =============== DataLoader ===============
    common_kwargs = loader_kwargs(args)  # [PATCH] 安全 kwargs

    # # 训练 DataLoader：node_budget 优先（以 Trace 为单位）
    # if args.node_budget and args.node_budget > 0:
    #     train_sizes = [len(ds_a_tr.items[i]['nodes']) for i in normal_idx]
    #     sampler = NodeBudgetBatchSampler(normal_idx, train_sizes, node_budget=args.node_budget, shuffle=True)
    #     tr_loader_a = DataLoader(ds_a_tr, batch_size=None, batch_sampler=sampler, collate_fn=collate_batch, **common_kwargs)
    #     log(f"使用 NodeBudgetBatchSampler (node_budget={args.node_budget})")
    #     tr_len = len(sampler)
    # else:
    #     ds_tr_norm = Subset(ds_a_tr, normal_idx)
    #     tr_loader_a = DataLoader(ds_tr_norm, batch_size=args.batch, shuffle=True, collate_fn=collate_batch, **common_kwargs)
    #     tr_len = len(tr_loader_a)
    # ========= 2. 如果用 NodeBudgetBatchSampler，再剔除冲突键 =========
    if args.node_budget and args.node_budget > 0:
        train_sizes = [len(ds_a_tr.items[i]['nodes']) for i in normal_idx]
        sampler = NodeBudgetBatchSampler(normal_idx, train_sizes, node_budget=args.node_budget, shuffle=True)
        # 关键：去掉与 batch_sampler 互斥的参数
        bs_kwargs = {k: v for k, v in common_kwargs.items()
                    if k not in {'batch_size', 'shuffle', 'sampler', 'drop_last'}}

        tr_loader_a = DataLoader(ds_a_tr,
                                batch_sampler=sampler,
                                collate_fn=collate_batch,
                                **bs_kwargs)
        tr_len = len(sampler)
        log(f"使用 NodeBudgetBatchSampler (node_budget={args.node_budget})")
    else:
        # 普通随机 batch，保持原逻辑
        ds_tr_norm = Subset(ds_a_tr, normal_idx)
        tr_loader_a = DataLoader(ds_tr_norm,
                                batch_size=args.batch,
                                shuffle=True,
                                collate_fn=collate_batch,
                                **common_kwargs)
        tr_len = len(tr_loader_a)

    va_loader_a = DataLoader(ds_a_va_lmt, batch_size=1, shuffle=False, collate_fn=collate_batch, **common_kwargs)
    te_loader_a = DataLoader(ds_a_te_lmt, batch_size=1, shuffle=False, collate_fn=collate_batch, **common_kwargs)

    # =============== μ/σ（保持原实现） ===============
    log("拟合延迟 μ/σ（normal-only） …")
    mu, sd = fit_latency_stats([ds_a_tr.items[i] for i in normal_idx])

    # 类权重（status）
    w_stat = class_weights_from_dataset(ds_a_tr, num_classes=status_sz + 1, normal_only=True).to(dev)

    # 优化器
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    nan_log_path = os.path.join(args.report_dir, "nan_batches.log")

    # ---------------- Stage-A 损失（可切换 mse/nll） ----------------
    def loss_stage_a(out, g):
        # 状态（两种模式一致）
        stat_hat = torch.nan_to_num(out['stat_hat'])
        l_stat = F.cross_entropy(stat_hat, g.ndata['status'].to(stat_hat.device), weight=w_stat)

        # 结构
        src, dst = g.edges()
        if src.numel() > 0:
            logits_p = torch.nan_to_num(out['struct_logits'][src])
            tgt = g.ndata['api'][dst]
            l_struct = F.cross_entropy(logits_p, tgt)
        else:
            l_struct = torch.tensor(0., device=stat_hat.device)

        # KL
        mu_t, lv_t = out['mu'], out['logvar']
        mu_t = torch.nan_to_num(mu_t, nan=0.0, posinf=1e6, neginf=-1e6)
        lv_t = torch.nan_to_num(lv_t, nan=0.0, posinf=args.kl_clip, neginf=-args.kl_clip)
        lv_t = torch.clamp(lv_t, min=-args.kl_clip, max=args.kl_clip)
        l_kl = 0.5 * torch.mean(mu_t.pow(2) + torch.exp(lv_t) - 1.0 - lv_t)

        # —— 延迟项：分支 ——
        if args.a_loss == 'mse':
            # [PATCH] 原口径：log1p 域 z-MSE
            lat_obs = torch.log1p(torch.clamp(torch.nan_to_num(g.ndata['lat_ms'].to(stat_hat.device),
                                                               nan=0.0, posinf=1e12, neginf=0.0), min=0.0))
            lat_hat = torch.log1p(torch.clamp(torch.nan_to_num(out['lat_hat'], nan=0.0, posinf=1e12, neginf=0.0), min=0.0))
            api_np = g.ndata['api'].cpu().numpy()
            mu_v = torch.tensor([mu.get(int(a), float(lat_obs.mean())) for a in api_np], device=lat_obs.device)
            sd_v = torch.tensor([sd.get(int(a), 0.5) for a in api_np], device=lat_obs.device).clamp_min(0.5)
            z     = (lat_obs - mu_v) / sd_v
            z_hat = (lat_hat - mu_v) / sd_v
            l_lat = F.mse_loss(torch.clamp(z_hat, -6, 6), torch.clamp(z, -6, 6))
        else:
            # [PATCH] GTrace 风格：log1p 域 高斯 NLL（异方差，含 0.5·log(2π)）
            lat_obs = torch.log1p(torch.clamp(torch.nan_to_num(g.ndata['lat_ms'].to(stat_hat.device),
                                                               nan=0.0, posinf=1e12, neginf=0.0), min=0.0))
            mu_hat  = out['lat_mu_hat']
            logvar  = out['lat_logvar_hat']
            logvar = torch.clamp(torch.nan_to_num(logvar, nan=0.0, posinf=10.0, neginf=-10.0),
                                 min=torch.log(torch.tensor(0.5**2, device=lat_obs.device)),
                                 max=torch.log(torch.tensor(10.0**2, device=lat_obs.device)))
            l_lat_node = 0.5 * (torch.exp(-logvar) * (lat_obs - mu_hat)**2 + logvar + LOG_TWO_PI)
            l_lat = l_lat_node.mean()

        parts = {'l_lat': l_lat.item(), 'l_stat': l_stat.item(), 'l_struct': l_struct.item(), 'l_kl': l_kl.item()}
        return l_lat, l_stat, l_struct, l_kl, parts

    # 训练 Stage A
    log("Stage A: 无监督VAE（normal-only）训练开始 …")
    best_train_loss = float('inf')
    best_thr = 0.5  # 默认阈值
    total_steps_nominal = args.steps_per_epoch if args.steps_per_epoch > 0 else tr_len

    for ep in range(1, args.epochs_a + 1):
        model.train()
        total_steps = args.steps_per_epoch if args.steps_per_epoch > 0 else tr_len
        pbar = tqdm(tr_loader_a, ncols=100, desc=f"[A][ep{ep:02d}]", total=total_steps)
        loss_sum = {'loss': 0.0, 'l_lat': 0.0, 'l_stat': 0.0, 'l_struct': 0.0, 'l_kl': 0.0}
        steps = 0
        accum = 0
        opt.zero_grad(set_to_none=True)

        for g, y in pbar:
            g = g.to(dev)

            # 热身权重
            global_step = (ep - 1) * total_steps_nominal + steps
            kl_w = args.beta_kl * min(1.0, global_step / max(1, args.kl_warmup_steps))
            struct_w = args.struct_w * min(1.0, global_step / max(1, args.struct_warmup_steps))

            with torch.cuda.amp.autocast(enabled=(args.amp==1)):
                out = model(g, vae_mode=True, status_mask_p=args.status_mask_p)
                l_lat, l_stat, l_struct, l_kl, parts = loss_stage_a(out, g)
                kl_warm = min(1.0, ep / 5.0)
                loss = args.alpha_lat * l_lat + args.beta_stat * l_stat + struct_w * l_struct + kl_warm * l_kl
                loss = loss / max(args.grad_accum, 1)

            # NaN 守卫
            if not torch.isfinite(loss):
                ti = y.get('trace_idx', None)
                trace_idxs = [int(v) for v in (ti.detach().view(-1).cpu().tolist() if isinstance(ti, torch.Tensor) else [-1])]
                with open(nan_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "epoch": ep, "step": steps, "batch_size": len(trace_idxs),
                        "trace_idxs": trace_idxs[:64], "loss_parts": {k: float(v) for k, v in parts.items()}
                    }, ensure_ascii=False) + "\n")
                print(f"[WARN][A] NaN/Inf loss @epoch={ep} step={steps} traces(sample)={trace_idxs[:8]} → skip", flush=True)
                opt.zero_grad(set_to_none=True)
                steps += 1
                if args.steps_per_epoch > 0 and steps >= args.steps_per_epoch: break
                continue

            if args.amp==1:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum += 1
            if accum % max(args.grad_accum, 1) == 0:
                if args.amp==1: scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                if args.amp==1:
                    scaler.step(opt); scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)

            loss_sum['loss'] += float(loss.item()) * max(args.grad_accum, 1)
            for k in ('l_lat','l_stat','l_struct','l_kl'):
                loss_sum[k] += parts[k]
            steps += 1
            if steps % 10 == 0:
                pbar.set_postfix({
                    'loss': f"{loss_sum['loss']/steps:.3f}",
                    'lat': f"{loss_sum['l_lat']/steps:.3f}",
                    'stat': f"{loss_sum['l_stat']/steps:.3f}",
                    'struct': f"{loss_sum['l_struct']/steps:.3f}",
                    'kl': f"{loss_sum['l_kl']/steps:.3f}",
                    'kl_w': f"{kl_w:.2f}",
                    'sw': f"{struct_w:.2f}",
                })

            if args.steps_per_epoch > 0 and steps >= args.steps_per_epoch:
                break

        # 计算平均训练损失
        avg_train_loss = loss_sum['loss'] / steps if steps > 0 else float('inf')

        # 验证（传入 loss_mode，保持与训练口径一致）
        pr, roc, f1r, thr = evaluate_stage_a(
            model, va_loader_a, mu, sd, dev,
            topk=args.topk, alpha_lat=args.alpha_lat, beta_stat=args.beta_stat, recall_floor=0.95,
            loss_mode=args.a_loss
        )
        log(f"[A][ep{ep:02d}] Train Loss={avg_train_loss:.4f}, ROC-AUC={roc:.4f}, PR-AUC={pr:.4f}, F1@R≥95%={f1r:.4f}, thr={thr:.6f}")

        # 早停策略：基于训练损失而不是验证集指标
        improved = (best_train_loss - avg_train_loss) > args.delta_a
        if improved or ep == 1:  # 第一个epoch总是保存
            if improved:
                best_train_loss = avg_train_loss
                best_thr = thr
            torch.save({'model': model.state_dict(), 'thr': best_thr, 'mu': mu, 'sd': sd},
                       os.path.join(args.report_dir, 'stageA_best.pt'))
            log(f"[A] 保存最优 (Train Loss={best_train_loss:.4f}, thr={best_thr:.6f}) -> stageA_best.pt")
            no_improve_a = 0
        else:
            no_improve_a = no_improve_a + 1 if 'no_improve_a' in locals() else 1
            if args.early_stop_a and no_improve_a >= args.patience_a:
                log(f"[A] Early stop at ep{ep:02d} (训练损失无改善 {no_improve_a} ≥ {args.patience_a})")
                break

        torch.cuda.empty_cache()

    # 测试
    if args.metric_mode == "accf1":
        # 使用训练过程中得到的最佳阈值，而不是从验证集重新选择
        thr_fix = best_thr
        log(f"[A][TEST] 使用训练阶段最佳阈值 thr={thr_fix:.6f}")

        # 2) 在测试集上按固定阈值计算 Acc/Prec/Rec/F1 + 混淆矩阵
        yt, st = collect_unsup_labels_and_scores(
            model, te_loader_a, dev, mu, sd,
            args.a_loss, args.topk, args.alpha_lat, args.beta_stat
        )
        m = binary_metrics_from_scores(yt, st, thr_fix)
        log(f"[A][TEST @thr={m['thr']:.6f}] Acc={m['acc']:.4f}  Prec={m['prec']:.4f}  Rec={m['rec']:.4f}  F1={m['f1']:.4f}")
        cm = m["cm"]
        log(f"Confusion: TN={cm['TN']}  FP={cm['FP']}  FN={cm['FN']}  TP={cm['TP']}")
        append_result_txt(os.path.join(args.report_dir, 'result.txt'),
                          f"[Stage A][ACC/F1] thr={m['thr']:.6f} Acc={m['acc']:.4f} Prec={m['prec']:.4f} Rec={m['rec']:.4f} F1={m['f1']:.4f}\n")

        # （可选）同时打印/记录 legacy 指标，便于对齐论文/历史
        if args.print_legacy_also:
            pr, roc, f1r, thr = evaluate_stage_a(
                model, te_loader_a, mu, sd, dev,
                topk=args.topk, alpha_lat=args.alpha_lat, beta_stat=args.beta_stat, recall_floor=0.95,
                loss_mode=args.a_loss
            )
            log(f"[A][TEST][Legacy] ROC-AUC={roc:.4f} PR-AUC={pr:.4f} F1@R≥95%={f1r:.4f} thr={thr:.6f}")
            append_result_txt(os.path.join(args.report_dir, 'result.txt'),
                              f"[Stage A][Legacy] ROC-AUC={roc:.4f} PR-AUC={pr:.4f} F1@R≥95%={f1r:.4f} thr={thr:.6f}\n")
    else:
        # legacy：保持你当前行为不变
        pr, roc, f1r, thr = evaluate_stage_a(
            model, te_loader_a, mu, sd, dev,
            topk=args.topk, alpha_lat=args.alpha_lat, beta_stat=args.beta_stat, recall_floor=0.95,
            loss_mode=args.a_loss
        )
        append_result_txt(os.path.join(args.report_dir, 'result.txt'),
                          f"\n[Stage A] ROC-AUC={roc:.4f} PR-AUC={pr:.4f} F1@R≥95%={f1r:.4f} thr={thr:.6f}\n")
        log(f"[A][TEST] ROC-AUC={roc:.4f} PR-AUC={pr:.4f} F1@R≥95%={f1r:.4f} thr={thr:.6f}")

    # 把 Stage-A 的 TEST 指标追加写入 result.txt（保持原逻辑）
    append_dataset_and_stageA_report(
        result_path, stats,
        a_test_auc=roc, a_test_prauc=pr, a_test_f1=f1r
    )

    # =============== Stage B（保持原来训练逻辑，仅更换数据源并应用 8:1:1） ===============
    if args.enable_b and (ds_b_tr is not None):
        log("Stage B: 监督细类训练 …")
        tr_loader_b = DataLoader(ds_b_tr, batch_size=args.batch, shuffle=True, collate_fn=collate_batch, **common_kwargs)
        va_loader_b = DataLoader(ds_b_va_lmt, batch_size=args.batch, shuffle=False, collate_fn=collate_batch, **common_kwargs)
        te_loader_b = DataLoader(ds_b_te_lmt, batch_size=args.batch, shuffle=False, collate_fn=collate_batch, **common_kwargs)
        opt_b = torch.optim.Adam(model.parameters(), lr=args.lr)
        ce = nn.CrossEntropyLoss(); best_va_acc = -1.0
        for ep in range(1, args.epochs_b + 1):
            model.train(); pbar = tqdm(tr_loader_b, ncols=100, desc=f"[B][ep{ep:02d}]"); loss_sum=0.0; steps=0
            for g, y in pbar:
                g = g.to(dev)
                with torch.cuda.amp.autocast(enabled=(args.amp==1)):
                    out = model(g, vae_mode=False)
                    l_type = ce(out['logits_type'], y['y_type'].to(dev))
                    l_c3   = ce(out['logits_c3'],   y['y_c3'].to(dev))
                    l_bin  = F.binary_cross_entropy_with_logits(out['logit_bin'], y['y_bin'].float().to(dev))
                    loss = l_type + 0.7 * l_c3 + 0.3 * l_bin
                if not torch.isfinite(loss): continue
                opt_b.zero_grad(set_to_none=True)
                if args.amp==1:
                    scaler.scale(loss).backward(); scaler.unscale_(opt_b)
                else:
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                if args.amp==1: scaler.step(opt_b); scaler.update()
                else: opt_b.step()
                steps += 1; loss_sum += float(loss.item())
                if steps % 10 == 0: pbar.set_postfix({'loss': f"{loss_sum/steps:.3f}"})
            acc_va = evaluate_stage_b(model, va_loader_b, dev)
            # log(f"[B][ep{ep:02d}] type-acc(VAL)={acc_va:.4f}")
            # if acc_va > best_va_acc:
            #     best_va_acc = acc_va
            #     torch.save({'model': model.state_dict()}, os.path.join(args.report_dir, 'stageB_best.pt'))
            #     log(f"[B] 保存最优 (val acc={best_va_acc:.4f}) -> stageB_best.pt")

            # 早停
            improved = (acc_va - best_va_acc) > args.delta_b
            if improved:
                best_va_acc = acc_va
                torch.save({'model': model.state_dict()}, os.path.join(args.report_dir, 'stageB_best.pt'))
                log(f"[B] 保存最优 (val acc={best_va_acc:.4f}) -> stageB_best.pt")
                no_improve_b = 0
            else:
                no_improve_b = no_improve_b + 1 if 'no_improve_b' in locals() else 1
                if args.early_stop_b and no_improve_b >= args.patience_b:
                    log(f"[B] Early stop at ep{ep:02d} (no improve {no_improve_b} ≥ {args.patience_b})")
                    break

        acc_te = evaluate_stage_b(model, te_loader_b, dev)
        append_result_txt(os.path.join(args.report_dir, 'result.txt'), f"[Stage B] type-acc(TEST)={acc_te:.4f}\n")
        log(f"[B][TEST] type-acc={acc_te:.4f}")

        # ===== Stage-B 每类指标可视化（保持你现有实现） =====
        y_true_all, y_pred_all = [], []
        names = type_names if (isinstance(type_names, list) and len(type_names) > 0) else [f"type_{i}" for i in range(type_sz)]

        model.eval()
        with torch.no_grad():
            for g, y in te_loader_b:
                g = g.to(dev)
                out = model(g, vae_mode=False)
                preds = torch.argmax(out['logits_type'], dim=-1).detach().cpu().numpy()
                gts = y['y_type'].detach().cpu().numpy()
                mask = gts >= 0
                if mask.any():
                    y_true_all.extend(gts[mask].tolist())
                    y_pred_all.extend(preds[mask].tolist())

        df = compute_class_metrics(
            np.asarray(y_true_all, dtype=np.int32),
            np.asarray(y_pred_all, dtype=np.int32),
            class_names=names
        )
        per_class_csv = os.path.join(args.report_dir, "stageB_per_class_metrics.csv")
        per_class_png = os.path.join(args.report_dir, "stageB_per_class_metrics.png")
        save_class_metrics_table(df, per_class_csv, per_class_png)

        overall_acc = float(np.mean(np.array(y_true_all) == np.array(y_pred_all))) if len(y_true_all) else None
        append_stageB_report(result_path, overall_acc, df, per_class_csv, per_class_png)

        # === ADD: Stage-B 统一按 Acc / Macro-P/R/F1 输出（c3 与 type） ===
        def _macro_prf1(y_true, y_pred, num_classes=None):
            import numpy as _np
            y_true = _np.asarray(y_true, dtype=_np.int64)
            y_pred = _np.asarray(y_pred, dtype=_np.int64)
            if num_classes is None:
                num_classes = int(max(y_true.max(initial=0), y_pred.max(initial=0)) + 1)
            precs, recs, f1s, supports = [], [], [], []
            for c in range(num_classes):
                tp = int(((y_pred == c) & (y_true == c)).sum())
                fp = int(((y_pred == c) & (y_true != c)).sum())
                fn = int(((y_pred != c) & (y_true == c)).sum())
                supp = int((y_true == c).sum())
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec + 1e-12) if (prec + rec) > 0 else 0.0
                precs.append(prec)
                recs.append(rec)
                f1s.append(f1)
                supports.append(supp)
            mac_p = float(sum(precs) / max(1, len(precs)))
            mac_r = float(sum(recs) / max(1, len(recs)))
            mac_f = float(sum(f1s) / max(1, len(f1s)))
            acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
            return acc, mac_p, mac_r, mac_f

        # 1) 3-类（Normal/Service/Node） —— 直接从 logits_c3 argmax
        y_true_c3, y_pred_c3 = [], []
        model.eval()
        with torch.no_grad():
            for g, y in te_loader_b:
                g = g.to(dev)
                out = model(g, vae_mode=False)
                y_true_c3.extend(y['y_c3'].detach().cpu().numpy().tolist())
                y_pred_c3.extend(torch.argmax(out['logits_c3'], dim=-1).cpu().numpy().tolist())

        acc_c3, mp_c3, mr_c3, mf1_c3 = _macro_prf1(y_true_c3, y_pred_c3, num_classes=3)
        log(f"[B][TEST][C3] Acc={acc_c3:.4f}  Macro-P={mp_c3:.4f}  Macro-R={mr_c3:.4f}  Macro-F1={mf1_c3:.4f}")
        append_result_txt(os.path.join(args.report_dir, 'result.txt'),
                          f"[Stage B][C3] Acc={acc_c3:.4f} Macro-P={mp_c3:.4f} Macro-R={mr_c3:.4f} Macro-F1={mf1_c3:.4f}\n")

        # 2) 细类（Type） —— 只统计 y_type>=0 的样本（和你已有的 mask 一致）
        y_true_type, y_pred_type = [], []
        model.eval()
        with torch.no_grad():
            for g, y in te_loader_b:
                g = g.to(dev)
                out = model(g, vae_mode=False)
                preds = torch.argmax(out['logits_type'], dim=-1).cpu().numpy()
                gts = y['y_type'].cpu().numpy()
                m = gts >= 0
                if m.any():
                    y_true_type.extend(gts[m].tolist())
                    y_pred_type.extend(preds[m].tolist())

        if len(y_true_type):
            # type_sz 在上文读取 vocab 时已有
            acc_ty, mp_ty, mr_ty, mf1_ty = _macro_prf1(y_true_type, y_pred_type, num_classes=type_sz)
            log(f"[B][TEST][Type] Acc={acc_ty:.4f}  Macro-P={mp_ty:.4f}  Macro-R={mr_ty:.4f}  Macro-F1={mf1_ty:.4f}")
            append_result_txt(os.path.join(args.report_dir, 'result.txt'),
                              f"[Stage B][Type] Acc={acc_ty:.4f} Macro-P={mp_ty:.4f} Macro-R={mr_ty:.4f} Macro-F1={mf1_ty:.4f}\n")

    # =============== Stage C：RCA 测试集可选 unified_test ===============
    if args.enable_c:
        log("Stage C: 无监督 RCA 评估 …")
        if args.use_v3_splits and args.use_unified_test and ('ds_u_te' in locals()) and (ds_u_te is not None):
            te_loader_c = DataLoader(ds_u_te, batch_size=1, shuffle=False, collate_fn=collate_batch, **common_kwargs)
        else:
            te_loader_c = DataLoader(ds_a_te_lmt, batch_size=1, shuffle=False, collate_fn=collate_batch, **common_kwargs)
        # [PATCH] 传入 loss_mode，Stage-C 节点可疑度与 Stage-A 口径保持一致
        rca = evaluate_stage_c(model, te_loader_c, mu, sd, dev, alpha_lat=args.alpha_lat, beta_stat=args.beta_stat,
                               loss_mode=args.a_loss)
        msg = "[Stage C] RCA(top1={:.4f}, top3={:.4f}, top5={:.4f}, covered={:d})".format(
            rca['top1'], rca['top3'], rca['top5'], rca['covered'])
        append_result_txt(os.path.join(args.report_dir, 'result.txt'), msg); log(msg)

if __name__ == '__main__':
    main()
