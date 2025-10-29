# import torch, time
# print("Torch:", torch.__version__, "CUDA avail:", torch.cuda.is_available(), "torch.cuda:", torch.version.cuda)
#
# import dgl
# import dgl.nn.pytorch as dglnn
# print("DGL:", dgl.__version__)
#
# dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# g = dgl.rand_graph(2000, 20000)   # 小图
# x = torch.randn(2000, 64)
# conv = dglnn.GraphConv(64, 64)
#
# # CPU 计时
# t0 = time.time()
# y = conv(g, x)
# t_cpu = time.time() - t0
#
# # GPU 计时（如果 DGL 没有 CUDA 支持，这里要么很慢、要么报错、要么偷偷拽回 CPU）
# g = g.to(dev); x = x.to(dev); conv = conv.to(dev)
# torch.cuda.synchronize() if dev.type=="cuda" else None
# t0 = time.time()
# y = conv(g, x)
# torch.cuda.synchronize() if dev.type=="cuda" else None
# t_gpu = time.time() - t0
#
# print(f"GraphConv CPU time: {t_cpu:.4f}s")
# print(f"GraphConv {dev.type.upper()} time: {t_gpu:.4f}s")
# print("g.device:", g.device, "x.device:", x.device, "conv.weight.device:", next(conv.parameters()).device)

from collections import Counter
import json

def scan(jsonl):
    c_bin = Counter(); c_type = Counter()
    with open(jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            c_bin[r.get('y_bin',0)] += 1
            c_type[r.get('fault_type',-1)] += 1
    print('y_bin:', c_bin)
    print('y_type top:', c_type.most_common(10))

if __name__ == '__main__':
    scan('dataset/aiops_svnd_all_10000_20/train.jsonl'); scan('dataset/aiops_svnd_all_10000_20/val.jsonl'); scan('dataset/aiops_svnd_all_10000_20/test.jsonl')
