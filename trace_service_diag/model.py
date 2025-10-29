# model.py
# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLError
from dgl.nn import GraphConv
from utils import enforce_dag_parent

class ChildSumTreeLSTMOp(nn.Module):
    def __init__(self, x_size):
        super().__init__()
        self.W_iouf = nn.Linear(x_size, 4 * x_size, bias=False)
        self.U_iou  = nn.Linear(x_size, 3 * x_size, bias=False)
        self.b_iou  = nn.Parameter(torch.zeros(1, 3 * x_size))
        self.U_f    = nn.Linear(x_size, x_size)

    def message_func(self, edges): return {'h': edges.src['h'], 'c': edges.src['c']}
    def reduce_func(self, nodes):
        h_tild = torch.sum(nodes.mailbox['h'], 1)
        f = torch.sigmoid(nodes.data['f'].unsqueeze(1) + self.U_f(nodes.mailbox['h']))
        c = torch.sum(f * nodes.mailbox['c'], 1)
        return {'sum': self.U_iou(h_tild), 'c': c}
    def apply_node_func(self, nodes):
        iou = nodes.data['iou'] + nodes.data['sum'] + self.b_iou
        i, o, u = torch.chunk(iou, 3, -1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        c = i * u + nodes.data['c']
        h = o * torch.tanh(c)
        return {'h': h, 'c': c}

class TreeLSTMReadout(nn.Module):
    _loop_count = 0  # [INTG] 统计“loop detected”次数

    def __init__(self, x_size, out_size, ignore_loops=False, debug_dump_path: str = None):
        super().__init__()
        self.cell = ChildSumTreeLSTMOp(x_size)
        self.linear = nn.Linear(x_size, out_size)
        self.ignore_loops = ignore_loops   # [INTG] True=忽略环，用零向量兜底；False=抛错
        self.debug_dump_path = debug_dump_path  # [INTG] 把问题 trace 统计写入此路径（jsonl）

    def forward(self, g: dgl.DGLGraph, x: torch.Tensor, trace_id: int = None):
        # 反向 + 去自环
        g_rev = dgl.reverse(dgl.remove_self_loop(g))
        # [INTG] 在 CPU 上去重边，再搬回原设备（与你代码一致）  :contentReference[oaicite:3]{index=3}
        g_cpu = g_rev.cpu()
        g_cpu = dgl.to_simple(g_cpu, return_counts=None)
        g_rev = g_cpu.to(g_rev.device)

        g_rev.ndata['iou'], g_rev.ndata['f'] = torch.split(
            self.cell.W_iouf(x), [3 * x.size(-1), x.size(-1)], dim=-1)
        g_rev.ndata['sum'] = torch.zeros_like(g_rev.ndata['iou'])
        g_rev.ndata['h']   = torch.zeros_like(x)
        g_rev.ndata['c']   = torch.zeros_like(x)

        try:
            dgl.prop_nodes_topo(g_rev, self.cell.message_func, self.cell.reduce_func,
                                apply_node_func=self.cell.apply_node_func)
            h = g_rev.ndata.pop('h')
        except DGLError as e:
            # [INTG] 捕获环路，打印 & 计数 & 落盘  :contentReference[oaicite:4]{index=4}
            if "loop detected" in str(e):
                TreeLSTMReadout._loop_count += 1
                print(f"[TreeLSTM-Loop#{TreeLSTMReadout._loop_count}] trace_id={trace_id} nodes={g.num_nodes()}")
                if self.ignore_loops:
                    h = torch.zeros_like(x)  # 兜底
                else:
                    raise
            else:
                raise

        return self.linear(F.relu(h))

    @classmethod
    def get_loop_count(cls): return cls._loop_count

class TraceClassifier(nn.Module):
    """GCN + TreeLSTM，图读出拼接 → 分类头"""
    def __init__(self, api_vocab, status_vocab, num_classes,
                 emb=32, gc_hidden=64, tlstm_out=64, cls_hidden=64,
                 ignore_loops=False, debug_dump_path: str = None):
        super().__init__()
        self.api_emb    = nn.Embedding(api_vocab + 1, emb)
        self.status_emb = nn.Embedding(status_vocab + 1, emb)
        self.depth_emb  = nn.Embedding(64, emb)
        self.pos_emb    = nn.Embedding(512, emb)
        self.lat_mlp = nn.Sequential(nn.Linear(1, emb), nn.ReLU(), nn.Linear(emb, emb))
        in_dim = emb*4
        self.merge = nn.Linear(in_dim + emb, gc_hidden)
        self.gcn1 = GraphConv(gc_hidden, gc_hidden, allow_zero_in_degree=True)
        self.gcn2 = GraphConv(gc_hidden, gc_hidden, allow_zero_in_degree=True)
        self.treelstm = TreeLSTMReadout(gc_hidden, tlstm_out,
                                        ignore_loops=ignore_loops,
                                        debug_dump_path=debug_dump_path)
        self.readout = nn.Sequential(nn.Linear(gc_hidden + tlstm_out, cls_hidden),
                                     nn.ReLU(), nn.Linear(cls_hidden, num_classes))

    def forward(self, g: dgl.DGLGraph):
        api    = self.api_emb(g.ndata["api_id"])
        status = self.status_emb(g.ndata["status_id"])
        depth  = self.depth_emb(torch.clamp(g.ndata["depth"], 0, 63))
        pos    = self.pos_emb(torch.clamp(g.ndata["pos"],   0, 511))
        lat    = self.lat_mlp(g.ndata["lat"])
        x = torch.cat([api, status, depth, pos, lat], dim=-1)
        x = F.relu(self.merge(x))
        x = F.relu(self.gcn1(g, x))
        x = F.relu(self.gcn2(g, x))
        g.ndata["x"] = x
        mean_x = dgl.mean_nodes(g, "x")

        # TreeLSTM（基于 parent->child 的有向 DAG）
        parent = g.ndata["parent"]
        parent = torch.tensor(enforce_dag_parent(parent.tolist()), device=g.device, dtype=torch.long)

        mask = parent >= 0
        if mask.any():
            src = parent[mask]
            dst = torch.nonzero(mask, as_tuple=False).squeeze(1)
            g_tree = dgl.graph((src, dst), num_nodes=g.num_nodes(), device=g.device)
        else:
            g_tree = dgl.graph(([],[]), num_nodes=g.num_nodes(), device=g.device)

        # [INTG] 从 ndata 读取 trace_id，传入 TreeLSTM 打印/落盘用  :contentReference[oaicite:5]{index=5}
        tid_tensor = g.ndata.get("trace_id")
        tid = tid_tensor[0].item() if tid_tensor is not None else None

        tl = self.treelstm(g_tree, x, trace_id=tid)
        g.ndata["tl"] = tl
        mean_tl = dgl.mean_nodes(g, "tl")
        return self.readout(torch.cat([mean_x, mean_tl], dim=-1))
