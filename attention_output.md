10.7.9.232: my_rank 7/16 my_device 7/8
10.7.9.232: 
10.7.9.232: seq length: N = 60000
10.7.9.232: hidden dim: d = 16384
10.7.9.232: num heads: h = 128
10.7.9.232: num GPUs: P = 16
10.7.9.232: head per GPU: 8 tokens per GPU: 3750
10.7.9.232: type: torch.bfloat16
10.7.9.232: 
10.7.9.232: Ulysses attention
10.7.9.232: input [N/P, d]: torch.Size([3750, 16384]), elements: 61440000, size: 0.12 GB
10.7.9.232: Q [h, d, d/h]: torch.Size([128, 16384, 128]), elements: 268435456, size: 536.87 MB
10.7.9.232: K [h, d, d/h]: torch.Size([128, 16384, 128]), elements: 268435456, size: 536.87 MB
10.7.9.232: V [h, d, d/h]: torch.Size([128, 16384, 128]), elements: 268435456, size: 536.87 MB
10.7.9.232: proj [h, d/h, d]: torch.Size([128, 128, 16384]), elements: 268435456, size: 536.87 MB
10.7.9.232: Peak memory allocation: 2.27 GB
10.7.9.232: compute q, k, v
10.7.9.232: q = input x Q, k = input x K, v = input x V
10.7.9.232: flops: 96.64 TFLOPs
10.7.9.232: q [h, N/P, d/h]: torch.Size([128, 3750, 128]), elements: 61440000, size 122.88 MB
10.7.9.232: k [h, N/P, d/h]: torch.Size([128, 3750, 128]), elements: 61440000, size 122.88 MB
10.7.9.232: v [h, N/P, d/h]: torch.Size([128, 3750, 128]), elements: 61440000, size 122.88 MB
10.7.9.232: Peak memory allocation: 2.68 GB
10.7.9.232: all-to-all q, k, v
10.7.9.232: q_ [P, h/P, N/P, d/h]: torch.Size([16, 8, 3750, 128]), elements: 61440000, size 122.88 MB
10.7.9.232: k_ [P, h/P, N/P, d/h]: torch.Size([16, 8, 3750, 128]), elements: 61440000, size 122.88 MB
10.7.9.232: v_ [P, h/P, N/P, d/h]: torch.Size([16, 8, 3750, 128]), elements: 61440000, size 122.88 MB
10.7.9.232: Peak memory allocation: 3.05 GB
10.7.9.232: NCCL version 2.21.5+cuda12.5
10.7.9.232: reshape q_, k_, v_
10.7.9.232: q_ [h/P, N, d/h]: torch.Size([8, 60000, 128]), elements: 61440000, size 122.88 MB
10.7.9.232: k_ [h/P, N, d/h]: torch.Size([8, 60000, 128]), elements: 61440000, size 122.88 MB
10.7.9.232: v_ [h/P, N, d/h]: torch.Size([8, 60000, 128]), elements: 61440000, size 122.88 MB
10.7.9.232: Peak memory allocation: 3.17 GB
10.7.9.232: compute attention
10.7.9.232: A = q x k_t
10.7.9.232: flops: 117.96 TFLOPs
10.7.9.232: A [h/P, N, N]: torch.Size([8, 60000, 60000]), elements: 28800000000, size 57.60 GB
10.7.9.232: Peak memory allocation: 60.65 GB
10.7.9.232: compute c
10.7.9.232: c = A x v
10.7.9.232: flops: 117.96 TFLOPs
10.7.9.232: c [h/P, N, d/h]: torch.Size([8, 60000, 128]), elements: 61440000, size 122.88 MB
10.7.9.232: Peak memory allocation: 60.77 GB
10.7.9.232: transpose c
10.7.9.232: c [N, h/P, d/h]: torch.Size([8, 60000, 128]), elements: 61440000, size 122.88 MB
10.7.9.232: Peak memory allocation: 60.77 GB
10.7.9.232: all-to-all c
10.7.9.232: c_ [P, N/P, h/P, d/h]: torch.Size([16, 3750, 8, 128]), elements: 61440000, size 122.88 MB
10.7.9.232: Peak memory allocation: 60.90 GB
10.7.9.232: transpose & reshape c_ and reshape projection
10.7.9.232: c_ [N/P, d]: torch.Size([3750, 16384]), elements: 61440000, size 122.88 MB
10.7.9.232: proj [d, d]: torch.Size([16384, 16384]), elements: 268435456, size 536.87 MB
10.7.9.232: Peak memory allocation: 61.02 GB
10.7.9.232: compute output
10.7.9.232: output = c x proj
10.7.9.232: flops: 32.21 TFLOPs
10.7.9.232: output [N/P, d]: torch.Size([3750, 16384]), elements: 61440000, size 0.12 GB
10.7.9.232: Peak memory allocation: 61.02 GB
