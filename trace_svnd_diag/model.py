# model.py  (v0.2-nodectx)
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv, GATConv

# ========== TreeLSTM（简版 Child-Sum Readout） ==========
class ChildSumTreeLSTMOp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W_iouf = nn.Linear(dim, 4*dim, bias=False)
        self.U_iou  = nn.Linear(dim, 3*dim, bias=False)
        self.b_iou  = nn.Parameter(torch.zeros(1, 3*dim))
        self.U_f    = nn.Linear(dim, dim)
    def message(self, edges): return {'h': edges.src['h'], 'c': edges.src['c']}
    def reduce(self, nodes):
        h_sum = torch.sum(nodes.mailbox['h'], 1)
        f = torch.sigmoid(self.U_f(nodes.mailbox['h']))
        c = torch.sum(f * nodes.mailbox['c'], 1)
        return {'sum': self.U_iou(h_sum), 'c': c}
    def apply(self, nodes):
        iou = nodes.data['iou'] + nodes.data['sum'] + self.b_iou
        i, o, u = torch.chunk(iou, 3, dim=-1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        c = i * u + nodes.data['c']
        h = o * torch.tanh(c)
        return {'h': h, 'c': c}

class TreeLSTMReadout(nn.Module):
    def __init__(self, dim, out):
        super().__init__()
        self.cell = ChildSumTreeLSTMOp(dim)
        self.out  = nn.Linear(dim, out)
    def forward(self, g: dgl.DGLGraph, x):
        g = dgl.reverse(dgl.remove_self_loop(g))
        g.ndata['iou'], g.ndata['f'] = torch.split(self.cell.W_iouf(x), [3*x.size(-1), x.size(-1)], dim=-1)
        g.ndata['sum'] = torch.zeros_like(g.ndata['iou'])
        g.ndata['h']   = torch.zeros_like(x)
        g.ndata['c']   = torch.zeros_like(x)
        dgl.prop_nodes_topo(g, self.cell.message, self.cell.reduce, apply_node_func=self.cell.apply)
        h = g.ndata['h']
        return self.out(F.relu(h))

# ========== 主模型：GCN(调用图) + Host-GCN(同节点) + TreeLSTM + Ctx ==========
class TraceClassifier(nn.Module):
    def __init__(self, api_vocab, status_vocab, node_vocab, n_types, emb=32, gc_hidden=64, tlstm_out=64, ctx_dim=7, cls_hidden=128):
        super().__init__()
        # Embeddings
        self.api_emb    = nn.Embedding(api_vocab + 1, emb)
        self.status_emb = nn.Embedding(status_vocab + 1, emb)
        self.node_emb   = nn.Embedding(node_vocab + 1, emb)  # NEW
        self.depth_emb  = nn.Embedding(64, emb)
        self.pos_emb    = nn.Embedding(512, emb)
        self.lat_mlp    = nn.Sequential(nn.Linear(1, emb), nn.ReLU(), nn.Linear(emb, emb))
        in_dim = emb*5 + emb  # api/status/node/depth/pos + lat_mlp
        self.merge = nn.Linear(in_dim, gc_hidden)

        # 调用图 GCN
        self.gcn1 = GraphConv(gc_hidden, gc_hidden, allow_zero_in_degree=True)
        self.gcn2 = GraphConv(gc_hidden, gc_hidden, allow_zero_in_degree=True)
        # Host 共址 GCN（同 node 的 span 用链式连接）
        self.hgc1 = GraphConv(gc_hidden, gc_hidden, allow_zero_in_degree=True)
        self.hgc2 = GraphConv(gc_hidden, gc_hidden, allow_zero_in_degree=True)
        # TreeLSTM
        self.tlstm = TreeLSTMReadout(gc_hidden, tlstm_out)

        # 上下文通道
        self.ctx_mlp = nn.Sequential(nn.Linear(ctx_dim, cls_hidden//2), nn.ReLU(), nn.Dropout(0.1)) if ctx_dim and ctx_dim>0 else None

        # 融合与三头
        fuse_in = gc_hidden + gc_hidden + tlstm_out  # call + host + tree
        if self.ctx_mlp is not None: fuse_in += cls_hidden//2
        self.fuse = nn.Sequential(nn.Linear(fuse_in, cls_hidden), nn.ReLU(), nn.Dropout(0.2))
        self.head_bin  = nn.Linear(cls_hidden, 1)
        self.head_c3   = nn.Linear(cls_hidden, 3)
        self.head_type = nn.Linear(cls_hidden, n_types if n_types>0 else 1)

    def _make_host_graph(self, node_ids: torch.Tensor):
        # node_ids: (N, ) long on GPU
        groups = {}
        for i, nid in enumerate(node_ids.tolist()):
            groups.setdefault(nid, []).append(i)

        src_list, dst_list = [], []
        for _, idxs in groups.items():
            if len(idxs) <= 1:
                continue
            idxs.sort()
            for a, b in zip(idxs[:-1], idxs[1:]):
                # 链式 + 双向
                src_list.extend([a, b])
                dst_list.extend([b, a])

        device = node_ids.device
        N = node_ids.size(0)
        loop = torch.arange(N, device=device, dtype=torch.long)

        if len(src_list) == 0:
            src = loop
            dst = loop
        else:
            src = torch.tensor(src_list, dtype=torch.long, device=device)
            dst = torch.tensor(dst_list, dtype=torch.long, device=device)
            src = torch.cat([src, loop], dim=0)
            dst = torch.cat([dst, loop], dim=0)

        return dgl.graph((src, dst), num_nodes=N, idtype=torch.int64, device=device)

    def forward(self, g: dgl.DGLGraph):
        api    = self.api_emb(g.ndata["api_id"])
        status = self.status_emb(g.ndata["status_id"])
        nodev  = self.node_emb(g.ndata["node_id"])
        depth  = self.depth_emb(torch.clamp(g.ndata["depth"],0,63))
        pos    = self.pos_emb(torch.clamp(g.ndata["pos"],0,511))
        lat    = self.lat_mlp(g.ndata["lat"])
        x = torch.cat([api,status,nodev,depth,pos,lat], dim=-1)
        x = F.relu(self.merge(x))

        # 调用图
        device = g.device
        x = x.to(device).contiguous()

        src, dst = g.edges()
        src = src.to(torch.long);
        dst = dst.to(torch.long)
        N = g.num_nodes()
        loop = torch.arange(N, device=device, dtype=torch.long)

        src2 = torch.cat([src, dst, loop], dim=0)
        dst2 = torch.cat([dst, src, loop], dim=0)

        g_call = dgl.graph((src2, dst2), num_nodes=N, idtype=torch.int64, device=device)

        h_call = F.relu(self.gcn1(g_call, x))
        h_call = self.gcn2(g_call, h_call)

        # Host 共址图
        g_host = self._make_host_graph(g.ndata["node_id"])
        h_host = F.relu(self.hgc1(g_host, x)); h_host = self.hgc2(g_host, h_host)

        # --- TreeLSTM：基于父→子 DAG。注意：batch 后 parent 仍是“局部索引”，必须逐子图构树 ---
        if "parent" in g.ndata:
            # 拆成子图
            sub_graphs = dgl.unbatch(g)
            sizes = [sg.num_nodes() for sg in sub_graphs]
            # x 也按节点数拆分（顺序与 unbatch 保持一致）
            x_splits = torch.split(x, sizes, dim=0)

            tree_list = []
            for sg, xi in zip(sub_graphs, x_splits):
                p = sg.ndata["parent"]
                m = (p >= 0)
                if m.any():
                    s = p[m].to(torch.long)
                    d = torch.nonzero(m, as_tuple=False).squeeze(1).to(torch.long)
                    t = dgl.graph((s, d), num_nodes=sg.num_nodes(),
                                  idtype=torch.int64, device=sg.device)
                else:
                    t = dgl.graph(
                        (torch.tensor([], dtype=torch.long, device=sg.device),
                         torch.tensor([], dtype=torch.long, device=sg.device)),
                        num_nodes=sg.num_nodes(), idtype=torch.int64, device=sg.device
                    )
                tree_list.append(t)

            g_tree = dgl.batch(tree_list)
        else:
            g_tree = dgl.graph(
                (torch.tensor([], dtype=torch.long, device=g.device),
                 torch.tensor([], dtype=torch.long, device=g.device)),
                num_nodes=g.num_nodes(), idtype=torch.int64, device=g.device
            )

        h_tl = self.tlstm(g_tree, x)

        # 节点级 → 图级 readout（均值）
        g.ndata["call"]=h_call; g.ndata["host"]=h_host; g.ndata["tl"]=h_tl
        mean_call = dgl.mean_nodes(g,"call"); mean_host = dgl.mean_nodes(g,"host"); mean_tl = dgl.mean_nodes(g,"tl")

        parts = [mean_call, mean_host, mean_tl]
        if "ctx" in g.ndata:
            ctx = dgl.mean_nodes(g, "ctx")  # 每节点同向量 → mean 等于原向量
            if self.ctx_mlp is not None:
                parts.append(self.ctx_mlp(ctx))

        fused = self.fuse(torch.cat(parts, dim=-1))

        return {
            "logit_bin":  self.head_bin(fused).squeeze(-1),
            "logits_c3":  self.head_c3(fused),
            "logits_type":self.head_type(fused),
        }
