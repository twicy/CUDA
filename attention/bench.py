import math

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

attn_v0 = load(name='attn_v0', sources=['main.cpp', 'flash_attn_v0.cu'])

# configuration
d_model = 1024

batch_size = 16
n_head = 8
seq_len = 1024
d_k = d_model // n_head

# self_attention
q = torch.randn(batch_size, n_head, seq_len, d_k).cuda()
k = torch.randn(batch_size, n_head, seq_len, d_k).cuda()
v = torch.randn(batch_size, n_head, seq_len, d_k).cuda()

def attn(q, k, v):
    attn = q @ k.transpose(-2, -1) / (d_k ** 0.5)
    # batch_size, n_head, seq_len, seq_len
    attn = F.softmax(attn)
    return attn @ v

print('=== Baseline (naiive attention) ===')
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    manual_result = attn(q, k, v)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print('=== attn_v0 ===')
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    minimal_result = attn_v0.forward(q, k, v)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print('attn values sanity check:', torch.allclose(minimal_result, manual_result, rtol=0, atol=1e-02))