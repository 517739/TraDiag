# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv, GATConv


class ChildSumTreeLSTMOp(nn.Module):
    """简化 Child-Sum TreeLSTM，用于父->子聚合（结构/时序表征通道）"""
    def __init__(self, dim: int):
        super().__init__()
        self.ioux = nn.Linear(dim, 3 * dim, bias=False)
        self.iouh = nn.Linear(dim, 3 * dim, bias=False)
        self.fx = nn.Linear(dim, dim, bias=False)
        self.fh = nn.Linear(dim, dim, bias=False)

    def forward(self, g: dgl.DGLGraph, x: torch.Tensor):
        parent = g.ndata['parent'].to(x.device)  # [N]
        depth = g.ndata['depth'].to(x.device)    # [N]
        order = torch.argsort(depth, descending=True)  # 叶到根
        h = torch.zeros_like(x)
        c = torch.zeros_like(x)
        for idx in order.tolist():
            child = (parent == idx).nonzero(as_tuple=False).flatten()
            if child.numel() == 0:
                hc = x[idx:idx + 1]
                cc = torch.tanh(hc)
            else:
                h_sum = h[child].sum(dim=0, keepdim=True)
                iou = self.ioux(x[idx:idx + 1]) + self.iouh(h_sum)
                i, o, u = torch.chunk(iou, 3, dim=-1)
                i, o = torch.sigmoid(i), torch.sigmoid(o)
                u = torch.tanh(u)
                f = torch.sigmoid(self.fx(x[idx:idx + 1]) + self.fh(h[child]))
                c_next = (f * c[child]).sum(dim=0, keepdim=True) + i * u
                h_next = o * torch.tanh(c_next)
                hc, cc = h_next, c_next
            h[idx] = hc
            c[idx] = cc
        return h


class TraceUnifiedModelV3(nn.Module):
    """
    与 v2 骨架一致的统一模型：
      Encoder: Embedding -> merge -> GCN(call) + GCN(host-approx) + TreeLSTM -> fuse(+ctx)
      VAE: 图级 z，节点级解码 (latency/status) + 结构头(父节点预测下一跳API)
      Heads: det/c3/type + node suspicious（RCA辅助）
    """
    def __init__(self,
                 api_vocab: int,
                 status_vocab: int,
                 node_vocab: int,
                 type_classes: int,
                 ctx_dim: int = 0,
                 emb: int = 64,
                 gc_hidden: int = 128,
                 conv_call: str = 'gcn',  # 'gcn' 或 'gat'
                 conv_host: str = 'gcn',  # 'gcn' 或 'gat'
                 gat_heads: int = 4,
                 gat_feat_drop: float = 0.20,
                 gat_attn_drop: float = 0.10):
        super().__init__()
        # Embeddings
        self.api_emb    = nn.Embedding(api_vocab + 1,   emb)
        self.status_emb = nn.Embedding(status_vocab + 1, emb)
        self.node_emb   = nn.Embedding(node_vocab + 1,   emb)
        self.depth_emb  = nn.Embedding(64,               emb)
        self.pos_emb    = nn.Embedding(2048,             emb)
        self.lat_mlp    = nn.Sequential(nn.Linear(1, emb), nn.ReLU(), nn.Linear(emb, emb))

        in_dim = emb * 5 + emb  # api/status/node/depth/pos + lat
        self.merge = nn.Linear(in_dim, gc_hidden)
        self.call_kind = conv_call.lower()
        self.host_kind = conv_host.lower()
        self.gat_heads = int(gat_heads)
        self.gat_feat_drop = float(gat_feat_drop)
        self.gat_attn_drop = float(gat_attn_drop)

        # 门控
        self.branch_gate = nn.Sequential(
            nn.Linear(gc_hidden * 3, gc_hidden), nn.ReLU(), nn.Linear(gc_hidden, 3)
        )

        # ---- 调用图分支：按需构建 ----
        if self.call_kind == 'gat':
            self.call1 = GATConv(in_feats=gc_hidden,
                                 out_feats=gc_hidden // gat_heads,
                                 num_heads=self.gat_heads,
                                 feat_drop=self.gat_feat_drop,
                                 attn_drop=self.gat_attn_drop,
                                 residual=True,
                                 allow_zero_in_degree=True)  # 维度保持为 gc_hidden
            self.call2 = GATConv(in_feats=gc_hidden,  # 这里输入是 [N, H]
                                 out_feats=gc_hidden, # 单头直接输出 H
                                 num_heads=1,
                                 feat_drop=self.gat_feat_drop,
                                 attn_drop=self.gat_attn_drop,
                                 residual=True,
                                 allow_zero_in_degree=True)
        else:
            self.call1 = GraphConv(gc_hidden, gc_hidden, allow_zero_in_degree=True)
            self.call2 = GraphConv(gc_hidden, gc_hidden, allow_zero_in_degree=True)

        # ---- 主机共址分支：按需构建 ----
        if self.host_kind == 'gat':
            self.host1 = GATConv(gc_hidden, gc_hidden,
                                 num_heads=self.gat_heads,
                                 feat_drop=self.gat_feat_drop,
                                 attn_drop=self.gat_attn_drop,
                                 residual=True,
                                 allow_zero_in_degree=True,
                                 concat=False)
            self.host2 = GATConv(gc_hidden, gc_hidden,
                                 num_heads=1,
                                 feat_drop=self.gat_feat_drop,
                                 attn_drop=self.gat_attn_drop,
                                 residual=True,
                                 allow_zero_in_degree=True,
                                 concat=False)
        else:
            self.host1 = GraphConv(gc_hidden, gc_hidden, allow_zero_in_degree=True)
            self.host2 = GraphConv(gc_hidden, gc_hidden, allow_zero_in_degree=True)

        # TreeLSTM
        self.tlstm = ChildSumTreeLSTMOp(gc_hidden)

        # 图级融合
        fuse_in = gc_hidden * 3
        self.ctx_mlp = nn.Sequential(nn.Linear(ctx_dim, gc_hidden), nn.ReLU()) if ctx_dim > 0 else None
        if ctx_dim > 0:
            fuse_in += gc_hidden
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, gc_hidden),
            nn.ReLU(),
            nn.Linear(gc_hidden, gc_hidden)
        )

        # VAE
        self.mu_head     = nn.Linear(gc_hidden, gc_hidden)
        self.logvar_head = nn.Linear(gc_hidden, gc_hidden)
        # self.dec_lat   = nn.Sequential(nn.Linear(gc_hidden * 2, gc_hidden), nn.ReLU(), nn.Linear(gc_hidden, 1))
        # [PATCH] 延迟解码：改为“均值+方差（logvar）”两个头（log 延迟域）
        self.dec_lat_mu = nn.Sequential(nn.Linear(gc_hidden * 2, gc_hidden), nn.ReLU(), nn.Linear(gc_hidden, 1))
        self.dec_lat_logvar = nn.Sequential(nn.Linear(gc_hidden * 2, gc_hidden), nn.ReLU(), nn.Linear(gc_hidden, 1))

        self.dec_stat  = nn.Sequential(nn.Linear(gc_hidden * 2, gc_hidden), nn.ReLU(), nn.Linear(gc_hidden, status_vocab + 1))
        self.dec_struct = nn.Linear(gc_hidden, api_vocab + 1)

        # 监督头（图级/节点级）
        self.head_bin  = nn.Sequential(nn.Linear(gc_hidden, gc_hidden), nn.ReLU(), nn.Linear(gc_hidden, 1))
        self.head_c3   = nn.Sequential(nn.Linear(gc_hidden, gc_hidden), nn.ReLU(), nn.Linear(gc_hidden, 3))
        self.head_type = nn.Sequential(nn.Linear(gc_hidden, gc_hidden), nn.ReLU(), nn.Linear(gc_hidden, type_classes))
        self.node_head = nn.Linear(gc_hidden, 1)

    @staticmethod
    def stable_kl(mu: torch.Tensor, logvar: torch.Tensor, clamp_low: float = -10.0, clamp_high: float = 10.0,
                  reduce: str = 'mean') -> torch.Tensor:
        """数值稳定的 KL 计算：对 logvar 夹紧，避免 exp 溢出；供需要时直接调用。"""
        logvar = torch.clamp(logvar, min=clamp_low, max=clamp_high)
        var = torch.exp(logvar)
        kl = 0.5 * (mu.pow(2) + var - 1.0 - logvar)
        if reduce == 'mean':
            return kl.mean()
        elif reduce == 'sum':
            return kl.sum()
        return kl

    def encode(self, g: dgl.DGLGraph, vae_mode: bool = False, status_mask_p: float = 0.0):
        dev = g.device
        api    = g.ndata['api'].to(dev)
        status = g.ndata['status'].to(dev)
        node   = g.ndata['node'].to(dev)
        depth  = g.ndata['depth'].to(dev)
        pos    = g.ndata['pos'].to(dev)
        lat    = g.ndata['lat_ms'].to(dev).unsqueeze(-1)  # [N,1]

        api_e = self.api_emb(api)
        if vae_mode and status_mask_p > 0:
            mask = (torch.rand_like(status.float()) < status_mask_p).long()
            status_use = status * (1 - mask)  # 被mask的置零（0 视为未用）
        else:
            status_use = status
        status_e = self.status_emb(status_use)
        node_e   = self.node_emb(node)
        depth_e  = self.depth_emb(torch.clamp(depth, 0, 63))
        pos_e    = self.pos_emb(torch.clamp(pos,   0, 2047))
        lat_e    = self.lat_mlp(lat)

        x0 = torch.cat([api_e, status_e, node_e, depth_e, pos_e, lat_e], dim=-1)
        x0 = self.merge(x0)  # [N, H]

        # 调用图（GCN 用 ReLU；GAT 建议 ELU）
        if self.call_kind == 'gat':
            # 第1层：多头输出形如 [N, heads, D]，摊平成 [N, heads*D] 再 ELU
            h_call = self.call1(g, x0)
            if h_call.dim() == 3:
                h_call = h_call.flatten(1)  # [N, heads*D]，通常等于 [N, H]
            h_call = F.elu(h_call)

            # 第2层：单头输出 [N, 1, H]（或 [N, H]）；挤掉 head 维，保持线性不再激活
            h_call = self.call2(g, h_call)
            if h_call.dim() == 3 and h_call.size(1) == 1:
                h_call = h_call[:, 0, :]
        else:
            h_call = F.relu(self.call1(g, x0))
            h_call = F.relu(self.call2(g, h_call))

        # 主机共址
        g_host = self._build_host_graph(g)
        if self.host_kind == 'gat':
            h_host = self.host1(g_host, x0)
            if h_host.dim() == 3:
                h_host = h_host.flatten(1)  # [N, heads*D] → 对齐到 [N, H]
            h_host = F.elu(h_host)

            h_host = self.host2(g_host, h_host)
            if h_host.dim() == 3 and h_host.size(1) == 1:
                h_host = h_host[:, 0, :]
        else:
            h_host = F.relu(self.host1(g_host, x0))
            h_host = F.relu(self.host2(g_host, h_host))

        # # 调用图（GCN 用 ReLU；GAT 建议 ELU）
        # if self.call_kind == 'gat':
        #     h_call = F.elu(self.call1(g, x0))
        #     h_call = self.call2(g, h_call)  # 第二层不再激活，保持与你现有风格一致
        # else:
        #     h_call = F.relu(self.call1(g, x0))
        #     h_call = F.relu(self.call2(g, h_call))
        #
        # # 主机共址
        # if self.host_kind == 'gat':
        #     h_host = F.elu(self.host1(g, x0))
        #     h_host = self.host2(g, h_host)
        # else:
        #     h_host = F.relu(self.host1(g, x0))
        #     h_host = F.relu(self.host2(g, h_host))
        # TreeLSTM
        h_tree = self.tlstm(g, x0)

        # 节点级表示
        # h_node = (h_call + h_host + h_tree) / 3.0  # [N,H]
        gate = torch.softmax(self.branch_gate(torch.cat([h_call, h_host, h_tree], dim=-1)), dim=-1)  # [N,3]
        w0, w1, w2 = gate[:, :1], gate[:, 1:2], gate[:, 2:3]
        h_node = w0 * h_call + w1 * h_host + w2 * h_tree  # [N,H]

        # 图级融合（临时写入再读出）
        with g.local_scope():
            g.ndata['tmp_call'] = h_call
            g.ndata['tmp_host'] = h_host
            g.ndata['tmp_tree'] = h_tree
            parts = [
                dgl.mean_nodes(g, 'tmp_call'),
                dgl.mean_nodes(g, 'tmp_host'),
                dgl.mean_nodes(g, 'tmp_tree'),
            ]
            if 'ctx' in g.ndata and self.ctx_mlp is not None:
                parts.append(self.ctx_mlp(dgl.mean_nodes(g, 'ctx')))
            fused = self.fuse(torch.cat(parts, dim=-1))  # [B,H]
        return h_node, fused

    @staticmethod
    def reparam(mu, logvar):
        eps = torch.randn_like(mu)
        return mu + torch.exp(0.5 * logvar) * eps

    def _expand_graph_feat(self, g: dgl.DGLGraph, feat_g: torch.Tensor):
        """将图级特征按每图节点数展开到节点维度"""
        bn = g.batch_num_nodes().tolist() if hasattr(g, 'batch_num_nodes') else [g.number_of_nodes()]
        chunks = [feat_g[i].unsqueeze(0).expand(bn[i], -1) for i in range(len(bn))]
        return torch.cat(chunks, dim=0)

    def _build_host_graph(self, g: dgl.DGLGraph) -> dgl.DGLGraph:
        """基于 node_id 构造星型主机共址图"""
        if g.number_of_nodes() == 0:
            return g

        if hasattr(g, 'batch_num_nodes'):
            batch_nodes = g.batch_num_nodes().tolist()
        else:
            batch_nodes = [g.number_of_nodes()]

        node_ids = g.ndata['node'].detach().cpu().tolist()
        src, dst = [], []
        offset = 0
        for n_nodes in batch_nodes:
            groups = {}
            for i in range(n_nodes):
                groups.setdefault(node_ids[offset + i], []).append(offset + i)
            for ids in groups.values():
                if len(ids) <= 1:
                    continue
                anchor = ids[0]
                others = ids[1:]
                src.extend([anchor] * len(others))
                dst.extend(others)
                src.extend(others)
                dst.extend([anchor] * len(others))
            offset += n_nodes

        if not src:
            empty = torch.tensor([], dtype=torch.int64)
            return dgl.graph((empty, empty), num_nodes=g.number_of_nodes()).to(g.device)

        device = g.device
        return dgl.graph(
            (torch.tensor(src, dtype=torch.int64, device=device),
             torch.tensor(dst, dtype=torch.int64, device=device)),
            num_nodes=g.number_of_nodes()
        )

    def forward(self, g: dgl.DGLGraph, vae_mode: bool = False, status_mask_p: float = 0.0, need_node_scores: bool = False):
        if not vae_mode and not need_node_scores:
            h_node, fused = self.encode(g, vae_mode=False, status_mask_p=0.0)
            fused_g = fused
            logit_bin = self.head_bin(fused_g).squeeze(-1)
            logits_c3 = self.head_c3(fused_g)
            logits_type = self.head_type(fused_g)
            # 数值净化
            logit_bin = torch.nan_to_num(logit_bin, nan=0.0, posinf=1e6, neginf=0.0)
            logits_c3 = torch.nan_to_num(logits_c3, nan=0.0, posinf=1e6, neginf=0.0)
            logits_type = torch.nan_to_num(logits_type, nan=0.0, posinf=1e6, neginf=0.0)
            return {'logit_bin': logit_bin, 'logits_c3': logits_c3, 'logits_type': logits_type}

        h_node, fused = self.encode(g, vae_mode=vae_mode, status_mask_p=status_mask_p)  # [N,H], [B,H]

        # VAE 参数，并对 logvar 做夹紧，避免下游 exp 溢出
        mu     = self.mu_head(fused)
        logvar = self.logvar_head(fused)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)

        z = self.reparam(mu, logvar)                     # [B,H]
        z_node = self._expand_graph_feat(g, z)           # [N,H]
        dec_in = torch.cat([h_node, z_node], dim=-1)     # [N,2H]

        lat_mu_hat = self.dec_lat_mu(dec_in).squeeze(-1)  # [N]，log(1+lat_ms) 的均值
        lat_logvar_hat = self.dec_lat_logvar(dec_in).squeeze(-1)  # [N]，log σ²

        # 数值稳定：限制 logvar 的范围（等价于对 σ 加上下/上限）
        lat_logvar_hat = torch.clamp(lat_logvar_hat,
                                     min=torch.log(torch.tensor(0.5 ** 2, device=lat_mu_hat.device)),
                                     max=torch.log(torch.tensor(10.0 ** 2, device=lat_mu_hat.device)))

        # 若为兼容旧代码，提供一个“ms 域”的回退输出（从均值反变换得到）
        lat_hat_ms_compat = torch.expm1(torch.clamp(lat_mu_hat, max=15.0))  # 防溢出

        stat_hat = self.dec_stat(dec_in)  # [N, |S|]
        struct_logits = self.dec_struct(h_node)  # [N, |API|]

        # 监督头（图级/节点级）
        fused_g    = fused
        logit_bin  = self.head_bin(fused_g).squeeze(-1)      # [B]
        logits_c3  = self.head_c3(fused_g)                   # [B,3]
        logits_type= self.head_type(fused_g)                 # [B,C]
        node_logit = self.node_head(h_node).squeeze(-1)      # [N]

        # ---- 输出净化，杜绝 NaN/Inf 传出（评估会更稳）----
        lat_hat_ms_compat   = torch.nan_to_num(lat_hat_ms_compat,       nan=0.0, posinf=1e6, neginf=0.0)
        lat_mu_hat          = torch.nan_to_num(lat_mu_hat,    nan=0.0, posinf=1e6, neginf=0.0)
        lat_logvar_hat      = torch.nan_to_num(lat_logvar_hat,nan=0.0, posinf=1e6, neginf=0.0)
        stat_hat            = torch.nan_to_num(stat_hat,      nan=0.0, posinf=1e6, neginf=0.0)
        struct_logits       = torch.nan_to_num(struct_logits, nan=0.0, posinf=1e6, neginf=0.0)
        logit_bin           = torch.nan_to_num(logit_bin,     nan=0.0, posinf=1e6, neginf=0.0)
        logits_c3           = torch.nan_to_num(logits_c3,     nan=0.0, posinf=1e6, neginf=0.0)
        logits_type         = torch.nan_to_num(logits_type,   nan=0.0, posinf=1e6, neginf=0.0)
        node_logit          = torch.nan_to_num(node_logit,    nan=0.0, posinf=1e6, neginf=0.0)

        return {
            'h_node': h_node, 'fused': fused,
            'mu': mu, 'logvar': logvar,
            'lat_mu_hat': lat_mu_hat, 'lat_logvar_hat': lat_logvar_hat,
            # 兼容旧字段（仅供可视化/回退，不参与损失）
            'lat_hat': lat_hat_ms_compat,
            'stat_hat': stat_hat, 'struct_logits': struct_logits,
            'logit_bin': logit_bin, 'logits_c3': logits_c3, 'logits_type': logits_type,
            'node_logit': node_logit
        }
