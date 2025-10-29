# -*- coding: utf-8 -*-
"""
TraDiag v3 训练脚本（4090适配版）
- Stage A: 无监督 VAE（normal-only 训练）
- Stage B: （可选）监督细类
- Stage C: （可选）无监督 RCA
- 加固：AMP、梯度累积、DataLoader 加速、NaN 守卫、评估非数与单类退化处理
"""

import os, json, argparse, random, math
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from model_v3 import TraceUnifiedModelV3
from utils_v3 import (
    JSONLDataset, collate_batch, fit_latency_stats,
    class_weights_from_dataset, evaluate_stage_a, evaluate_stage_b, evaluate_stage_c,
    append_result_txt, build_normal_index_from_csv
)

def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def main():
    p = argparse.ArgumentParser()
    # 基本
    p.add_argument('--data_root', type=str, default='../../dataset/aiops_svnd_all_1e5_1e2')
    p.add_argument('--report_dir', type=str, default='../../dataset/aiops_svnd_all_1e5_1e2/result_v3')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--seed', type=int, default=0)

    # 阶段与轮数
    p.add_argument('--epochs_a', type=int, default=20)
    p.add_argument('--enable_b', type=int, default=0)
    p.add_argument('--epochs_b', type=int, default=20)
    p.add_argument('--enable_c', type=int, default=0)

    # 模型/优化
    p.add_argument('--emb', type=int, default=64)
    p.add_argument('--gc_hidden', type=int, default=128)
    p.add_argument('--amp', type=int, default=1)
    p.add_argument('--grad_accum', type=int, default=1)
    p.add_argument('--max_nodes', type=int, default=0, help='A阶段训练丢弃节点数>此值的图；0=不限')

    # Stage A 超参
    p.add_argument('--topk', type=float, default=0.2)
    p.add_argument('--alpha_lat', type=float, default=2.0)
    p.add_argument('--beta_stat', type=float, default=1.0)
    p.add_argument('--beta_kl', type=float, default=0.5)
    p.add_argument('--struct_w', type=float, default=0.5)
    p.add_argument('--status_mask_p', type=float, default=0.7)

    # normal-only 控制
    p.add_argument('--normal_max', type=int, default=0)
    p.add_argument('--normal_index', type=str, default='')
    p.add_argument('--normal_csv', type=str, default='')

    # 适配/加速参数
    p.add_argument('--num_workers', type=int, default=1)         # 你是16核CPU，建议 8~14
    p.add_argument('--pin_memory', type=int, default=1)
    p.add_argument('--prefetch_factor', type=int, default=4)
    p.add_argument('--train_limit', type=int, default=0, help='限制A阶段训练样本数（normal-only子集），0=不限')
    p.add_argument('--val_limit', type=int, default=0)
    p.add_argument('--test_limit', type=int, default=0)
    p.add_argument('--steps_per_epoch', type=int, default=0, help='每个epoch最多训练多少个batch（0=全量）')
    p.add_argument('--fast', type=int, default=0, help='快速小规模试跑预设（覆盖若干参数）')

    args = p.parse_args()
    os.makedirs(args.report_dir, exist_ok=True)

    # “快跑”预设：几分钟一轮，适合排错/摸指标
    if args.fast:
        args.batch = 1
        args.grad_accum = 4
        args.emb = min(args.emb, 48)
        args.gc_hidden = min(args.gc_hidden, 96)
        if args.normal_max == 0: args.normal_max = 30000
        if args.max_nodes   == 0: args.max_nodes   = 1500
        if args.steps_per_epoch == 0: args.steps_per_epoch = 2000
        args.amp = 1
        args.num_workers = max(args.num_workers, 8)
        args.pin_memory = 1
        args.prefetch_factor = max(args.prefetch_factor, 2)
        log(f"[FAST] 小规模预设开启: batch={args.batch}, grad_accum={args.grad_accum}, "
            f"emb={args.emb}, gc_hidden={args.gc_hidden}, normal_max={args.normal_max}, "
            f"max_nodes={args.max_nodes}, steps_per_epoch={args.steps_per_epoch}, "
            f"num_workers={args.num_workers}, prefetch={args.prefetch_factor}")

    # 种子
    if args.seed >= 0:
        random.seed(args.seed); np.random.seed(args.seed)
        torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    # 4090优化
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision('medium')
        except Exception:
            pass

    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp==1 and torch.cuda.is_available()))

    # 数据
    log("加载 JSONL 数据 …")
    ds_tr = JSONLDataset(args.data_root, 'train')
    ds_va = JSONLDataset(args.data_root, 'val')
    ds_te = JSONLDataset(args.data_root, 'test')
    log(f"train={len(ds_tr)}  val={len(ds_va)}  test={len(ds_te)}")

    # 标签平衡提示
    pos_va = sum(int(r.get('y_bin', 0))==1 for r in ds_va.items)
    pos_te = sum(int(r.get('y_bin', 0))==1 for r in ds_te.items)
    log(f"[label balance] val pos={pos_va}/{len(ds_va)}  test pos={pos_te}/{len(ds_te)}")

    # 词表
    with open(os.path.join(args.data_root, 'vocab.json'), 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    api_sz    = int(vocab.get('api_vocab_size', 5000))
    status_sz = int(vocab.get('status_vocab_size', 10))
    node_sz   = int(vocab.get('node_vocab_size', 1000))
    type_sz   = int(vocab.get('type_vocab_size', len(vocab.get('type_names', [])) or 10))
    ctx_dim   = int(vocab.get('ctx_dim', 0))

    # 模型
    model = TraceUnifiedModelV3(api_sz, status_sz, node_sz, type_sz,
                                ctx_dim=ctx_dim, emb=args.emb, gc_hidden=args.gc_hidden).to(dev)
    log(model.__class__.__name__ + " 已构建。")

    # Stage A normal-only 构建
    if args.normal_csv and not args.normal_index:
        out_idx = os.path.join(args.report_dir, 'normal_index.json')
        build_normal_index_from_csv(args.normal_csv, ds_tr.items, out_idx)
        args.normal_index = out_idx
        log(f"从 Normal.csv 生成 normal_index：{out_idx}")

    if args.normal_index and os.path.isfile(args.normal_index):
        normal_idx = [int(i) for i in json.load(open(args.normal_index, 'r', encoding='utf-8'))]
        normal_idx = [i for i in normal_idx if 0 <= i < len(ds_tr)]
        log(f"从 normal_index 读取 normal-only 数量：{len(normal_idx)}")
    else:
        normal_idx = [i for i, r in enumerate(ds_tr.items) if int(r.get('y_bin', 0)) == 0]
        log(f"train.jsonl 内部筛选 normal-only 数量：{len(normal_idx)}")

    # 过滤巨图（仅A训练集）
    if args.max_nodes and args.max_nodes > 0:
        kept = []
        for i in normal_idx:
            n_nodes = len(ds_tr.items[i]['nodes'])
            if n_nodes <= args.max_nodes:
                kept.append(i)
        normal_idx = kept
        log(f"过滤超大图后 normal-only 数量：{len(normal_idx)} (max_nodes={args.max_nodes})")

    # 裁剪 normal-only 规模
    if args.normal_max and len(normal_idx) > args.normal_max:
        random.shuffle(normal_idx)
        normal_idx = normal_idx[:args.normal_max]
        log(f"裁剪 normal-only 数量至 {len(normal_idx)} (normal_max={args.normal_max})")

    # 训练子集进一步限制（与 normal_max 叠加，取更小的）
    if args.train_limit and len(normal_idx) > args.train_limit:
        random.shuffle(normal_idx)
        normal_idx = normal_idx[:args.train_limit]
        log(f"限制 normal-only 训练子集为 {len(normal_idx)} (train_limit={args.train_limit})")

    ds_tr_norm = Subset(ds_tr, normal_idx)

    # 限制 val/test 规模
    import torch.utils.data as tud
    va_indices = list(range(len(ds_va)))
    te_indices = list(range(len(ds_te)))
    if args.val_limit and len(va_indices) > args.val_limit:
        random.shuffle(va_indices); va_indices = va_indices[:args.val_limit]
        ds_va_lmt = tud.Subset(ds_va, va_indices)
        log(f"限制 val 集为 {len(ds_va_lmt)} (val_limit={args.val_limit})")
    else:
        ds_va_lmt = ds_va
    if args.test_limit and len(te_indices) > args.test_limit:
        random.shuffle(te_indices); te_indices = te_indices[:args.test_limit]
        ds_te_lmt = tud.Subset(ds_te, te_indices)
        log(f"限制 test 集为 {len(ds_te_lmt)} (test_limit={args.test_limit})")
    else:
        ds_te_lmt = ds_te

    # DataLoader
    common_kwargs = dict(
        num_workers=max(0, args.num_workers),
        pin_memory=bool(args.pin_memory),
        persistent_workers=True if args.num_workers > 0 else False
    )
    if args.num_workers > 0 and args.prefetch_factor:
        common_kwargs['prefetch_factor'] = args.prefetch_factor

    tr_loader_a = DataLoader(ds_tr_norm, batch_size=args.batch, shuffle=True,  collate_fn=collate_batch, **common_kwargs)
    va_loader_a = DataLoader(ds_va_lmt,  batch_size=1,        shuffle=False, collate_fn=collate_batch, **common_kwargs)
    te_loader_a = DataLoader(ds_te_lmt,  batch_size=1,        shuffle=False, collate_fn=collate_batch, **common_kwargs)

    # μ/σ
    log("拟合延迟 μ/σ（normal-only） …")
    mu, sd = fit_latency_stats([ds_tr.items[i] for i in normal_idx])

    # 类权重（status）
    w_stat = class_weights_from_dataset(ds_tr, num_classes=status_sz + 1, normal_only=True).to(dev)

    # 优化器
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    nan_log_path = os.path.join(args.report_dir, "nan_batches.log")

    def loss_stage_a(out, g):
        api = g.ndata['api']
        lat = torch.nan_to_num(g.ndata['lat_ms'].to(out['lat_hat'].device), nan=0.0, posinf=1e6, neginf=-1e6)
        lat_hat = torch.nan_to_num(out['lat_hat'], nan=0.0, posinf=1e6, neginf=-1e6)
        stat_hat = torch.nan_to_num(out['stat_hat'])

        api_np = api.cpu().numpy()
        mu_v = torch.tensor([mu.get(int(a), float(lat.mean())) for a in api_np], device=lat.device)
        sd_v = torch.tensor([sd.get(int(a), 1.0) for a in api_np], device=lat.device)
        sd_v = torch.clamp(sd_v, min=1e-6)

        lat_z = (lat - mu_v) / sd_v
        lat_hat_z = (lat_hat - mu_v) / sd_v
        l_lat = F.mse_loss(lat_hat_z, lat_z)

        l_stat = F.cross_entropy(stat_hat, g.ndata['status'].to(stat_hat.device), weight=w_stat)

        src, dst = g.edges()
        if src.numel() > 0:
            logits_p = torch.nan_to_num(out['struct_logits'][src])
            tgt = g.ndata['api'][dst]
            l_struct = F.cross_entropy(logits_p, tgt)
        else:
            l_struct = torch.tensor(0., device=lat.device)

        mu_t, lv_t = out['mu'], out['logvar']
        l_kl = -0.5 * torch.mean(1 + lv_t - mu_t.pow(2) - lv_t.exp())

        loss = args.alpha_lat * l_lat + args.beta_stat * l_stat + args.struct_w * l_struct + args.beta_kl * l_kl
        parts = {'l_lat': l_lat.item(), 'l_stat': l_stat.item(), 'l_struct': l_struct.item(), 'l_kl': l_kl.item()}
        return loss, parts

    # 训练 Stage A
    log("Stage A: 无监督VAE（normal-only）训练开始 …")
    best_val_prauc, best_thr = -1.0, 0.0

    for ep in range(1, args.epochs_a + 1):
        model.train()
        total_steps = args.steps_per_epoch if args.steps_per_epoch > 0 else len(tr_loader_a)
        pbar = tqdm(tr_loader_a, ncols=100, desc=f"[A][ep{ep:02d}]", total=total_steps)
        loss_sum = {'loss': 0.0, 'l_lat': 0.0, 'l_stat': 0.0, 'l_struct': 0.0, 'l_kl': 0.0}
        steps = 0
        accum = 0
        opt.zero_grad(set_to_none=True)

        for g, y in pbar:
            g = g.to(dev)
            with torch.cuda.amp.autocast(enabled=(args.amp==1)):
                out = model(g, vae_mode=True, status_mask_p=args.status_mask_p)
                loss, parts = loss_stage_a(out, g)
                loss = loss / max(args.grad_accum, 1)

            # NaN 守卫：记录并跳过
            if not torch.isfinite(loss):
                ti = y.get('trace_idx', None)
                if isinstance(ti, torch.Tensor):
                    trace_idxs = [int(v) for v in ti.detach().view(-1).cpu().tolist()]
                else:
                    trace_idxs = [-1]
                with open(nan_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "epoch": ep,
                        "step": steps,
                        "batch_size": len(trace_idxs),
                        "trace_idxs": trace_idxs[:64],
                        "loss_parts": {k: float(v) for k, v in parts.items()},
                    }, ensure_ascii=False) + "\n")
                print(f"[WARN][A] NaN/Inf loss @epoch={ep} step={steps} traces(sample)={trace_idxs[:8]} → skip", flush=True)
                opt.zero_grad(set_to_none=True)
                steps += 1
                if args.steps_per_epoch > 0 and steps >= args.steps_per_epoch:
                    break
                continue

            if args.amp==1:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum += 1
            if accum % max(args.grad_accum, 1) == 0:
                if args.amp==1:
                    scaler.unscale_(opt)
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
                })

            if args.steps_per_epoch > 0 and steps >= args.steps_per_epoch:
                break

        # 验证
        pr, roc, f1r, thr = evaluate_stage_a(
            model, va_loader_a, mu, sd, dev,
            topk=args.topk, alpha_lat=args.alpha_lat, beta_stat=args.beta_stat, recall_floor=0.95
        )
        log(f"[A][ep{ep:02d}] PR-AUC={pr:.4f}  ROC-AUC={roc:.4f}  F1@R≥95%={f1r:.4f}  thr={thr:.6f}")

        if pr > best_val_prauc:
            best_val_prauc, best_thr = pr, thr
            torch.save({'model': model.state_dict(), 'thr': best_thr, 'mu': mu, 'sd': sd},
                       os.path.join(args.report_dir, 'stageA_best.pt'))
            log(f"[A] 保存最优 (PR-AUC={best_val_prauc:.4f}, thr={best_thr:.6f}) -> stageA_best.pt")

        torch.cuda.empty_cache()

    # 测试
    pr, roc, f1r, thr = evaluate_stage_a(
        model, te_loader_a, mu, sd, dev,
        topk=args.topk, alpha_lat=args.alpha_lat, beta_stat=args.beta_stat, recall_floor=0.95
    )
    append_result_txt(os.path.join(args.report_dir, 'result.txt'),
        f"\n[Stage A] PR-AUC={pr:.4f} ROC-AUC={roc:.4f} F1@R≥95%={f1r:.4f} thr={thr:.6f}\n"
    )
    log(f"[A][TEST] PR-AUC={pr:.4f} ROC-AUC={roc:.4f} F1@R≥95%={f1r:.4f} thr={thr:.6f}")

    # Stage B（可选）
    if args.enable_b:
        log("Stage B: 监督细类训练 …")
        fault_idx = [i for i, r in enumerate(ds_tr.items) if int(r.get('y_bin', 0)) == 1]
        ds_tr_fault = Subset(ds_tr, fault_idx)

        common_kwargs_b = dict(**common_kwargs)
        tr_loader_b = DataLoader(ds_tr_fault, batch_size=args.batch, shuffle=True, collate_fn=collate_batch, **common_kwargs_b)
        va_loader_b = DataLoader(ds_va_lmt,   batch_size=args.batch, shuffle=False, collate_fn=collate_batch, **common_kwargs_b)
        te_loader_b = DataLoader(ds_te_lmt,   batch_size=args.batch, shuffle=False, collate_fn=collate_batch, **common_kwargs_b)

        opt_b = torch.optim.Adam(model.parameters(), lr=1e-3)
        ce = nn.CrossEntropyLoss()
        best_va_acc = -1.0

        for ep in range(1, args.epochs_b + 1):
            model.train()
            pbar = tqdm(tr_loader_b, ncols=100, desc=f"[B][ep{ep:02d}]")
            steps = 0; loss_sum = 0.0

            for g, y in pbar:
                g = g.to(dev)
                with torch.cuda.amp.autocast(enabled=(args.amp==1)):
                    out = model(g, vae_mode=False)
                    l_type = ce(out['logits_type'], y['y_type'].to(dev))
                    l_c3   = ce(out['logits_c3'],   y['y_c3'].to(dev))
                    l_bin  = F.binary_cross_entropy_with_logits(out['logit_bin'], y['y_bin'].float().to(dev))
                    loss = l_type + 0.7 * l_c3 + 0.3 * l_bin

                if not torch.isfinite(loss):
                    ti = y.get('trace_idx', None)
                    trace_idxs = [int(v) for v in (ti.detach().view(-1).cpu().tolist() if isinstance(ti, torch.Tensor) else [-1])]
                    print(f"[WARN][B] NaN/Inf loss @epoch={ep} step={steps} traces(sample)={trace_idxs[:8]} → skip", flush=True)
                    continue

                opt_b.zero_grad(set_to_none=True)
                if args.amp==1:
                    scaler.scale(loss).backward(); scaler.unscale_(opt_b)
                else:
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                if args.amp==1:
                    scaler.step(opt_b); scaler.update()
                else:
                    opt_b.step()

                steps += 1; loss_sum += float(loss.item())
                if steps % 10 == 0:
                    pbar.set_postfix({'loss': f"{loss_sum/steps:.3f}"})

            acc_va = evaluate_stage_b(model, va_loader_b, dev)
            log(f"[B][ep{ep:02d}] type-acc(VAL)={acc_va:.4f}")

            if acc_va > best_va_acc:
                best_va_acc = acc_va
                torch.save({'model': model.state_dict()}, os.path.join(args.report_dir, 'stageB_best.pt'))
                log(f"[B] 保存最优 (val acc={best_va_acc:.4f}) -> stageB_best.pt")

        acc_te = evaluate_stage_b(model, te_loader_b, dev)
        append_result_txt(os.path.join(args.report_dir, 'result.txt'),
                          f"[Stage B] type-acc(TEST)={acc_te:.4f}\n")
        log(f"[B][TEST] type-acc={acc_te:.4f}")

    # Stage C（可选）
    if args.enable_c:
        log("Stage C: 无监督 RCA 评估 …")
        te_loader_c = DataLoader(ds_te_lmt, batch_size=1, shuffle=False, collate_fn=collate_batch, **common_kwargs)
        rca = evaluate_stage_c(
            model, te_loader_c, mu, sd, dev,
            alpha_lat=args.alpha_lat, beta_stat=args.beta_stat
        )
        msg = "[Stage C] RCA(top1={:.4f}, top3={:.4f}, top5={:.4f}, covered={:d})".format(
            rca['top1'], rca['top3'], rca['top5'], rca['covered']
        )
        append_result_txt(os.path.join(args.report_dir, 'result.txt'), msg)
        log(msg)

if __name__ == '__main__':
    main()
