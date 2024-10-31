import torch
import torch.distributed as dist

def MLP_model(seq_length, hidden_dim, inter_size, num_layers, P, input_) -> torch.Tensor:
    # initialize model
    # input [N, d]
    # W1[L, d, d'/P]
    # W2[L, d'/P, d]
    W1 = torch.ones(num_layers, hidden_dim, inter_size//P, device=my_device, dtype=type)
    W2 = torch.ones(num_layers, inter_size//P, hidden_dim, device=my_device, dtype=type)
    inter = torch.empty(seq_length, inter_size//P, device=my_device, dtype=type)

    if my_rank == root_rank:
        print("\nModel parallel")
        print(f"input_ [N, d]: {input_.shape}, elements: {input_.nelement()}, size: {input_.element_size() * input_.nelement() / 1e9:.2f} GB")
        print(f"W1 [L, d, d'/P]: {W1.shape}, elements: {W1.nelement()}, size: {W1.element_size() * W1.nelement() / 1e9:.2f} GB")
        print(f"W2 [L, d'/P, d]: {W2.shape}, elements: {W2.nelement()}, size: {W2.element_size() * W2.nelement() / 1e9:.2f} GB")
        print(f"inter = input x W1")
        print(f"inter [N, d'/P]: {inter.shape}, elements: {inter.nelement()}, size: {inter.element_size() * inter.nelement() / 1e6:.2f} MB")
        torch.cuda.synchronize()
        print(f"Current memory allocation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Peak memory allocation: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # MLP loop
    for i in range(num_layers):
        inter = torch.matmul(input_, W1[i])
        inter = torch.nn.functional.gelu(inter)
        input_ = torch.matmul(inter, W2[i])
        dist.all_reduce(input_)

    return input_

def MLP_2D(seq_length, hidden_dim, inter_dim, num_layers, TP, DP, input_, group_TP) -> torch.Tensor:
    # initialize model
    # input_ [N/DP, d]
    # W1[L, d, d'/TP]
    # W2[L, d'/TP, d]
    # inter [N/DP, d'/TP]
    W1 = torch.ones(num_layers, hidden_dim, inter_dim//TP, device=my_device, dtype=type)
    W2 = torch.ones(num_layers, inter_dim//TP, hidden_dim, device=my_device, dtype=type)
    inter = torch.empty(seq_length//DP, inter_dim//TP, device=my_device, dtype=type)

    if my_rank == root_rank:
        print("\n2D Model parallel")
        print(f"input_ [N/DP, d]: {input_.shape}, elements: {input_.nelement()}, size: {input_.element_size() * input_.nelement() / 1e9:.2f} GB")
        print(f"W1 [L, d, d'/TP]: {W1.shape}, elements: {W1.nelement()}, size: {W1.element_size() * W1.nelement() / 1e9:.2f} GB")
        print(f"W2 [L, d'/TP, d]: {W2.shape}, elements: {W2.nelement()}, size: {W2.element_size() * W2.nelement() / 1e9:.2f} GB")
        print(f"inter = input x W1")
        print(f"flops: {num_layers * seq_length * hidden_dim * inter_dim / 1e12:.2f} TFLOPs")
        print(f"inter [N/DP, d'/TP]: {inter.shape}, elements: {inter.nelement()}, size: {inter.element_size() * inter.nelement() / 1e6:.2f} MB")
        print(f"activation f(inter)")
        print(f"output = inter x W2")
        print(f"flops: {num_layers * seq_length * hidden_dim * inter_dim / 1e12:.2f} TFLOPs")
        torch.cuda.synchronize()
        print(f"Current memory allocation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Peak memory allocation: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        print(f"Group TP: {group_TP}")
        print(f"input_ size: {input_.element_size() * input_.nelement() / 1e6:.2f} MB")

    # MLP loop
    for i in range(num_layers):
        inter = torch.matmul(input_, W1[i])
        inter = torch.nn.functional.gelu(inter)
        input_ = torch.matmul(inter, W2[i])
        dist.all_reduce(input_, group=group_TP)

    return input_

def ulysses_attention(seq_length, hidden_dim, num_heads, P) -> torch.Tensor:
    # initialize input and model
    # input [N/P, d]
    # Q, K, V [h, d, d/h]
    # O [h, d/h, d]
    input = torch.randn(seq_length//P, hidden_dim, device=my_device, dtype=type)
    Q = torch.ones(num_heads, hidden_dim, hidden_dim//num_heads, device=my_device, dtype=type)
    K = torch.ones_like(Q)
    V  = torch.ones_like(Q)
    O = torch.ones(num_heads, hidden_dim//num_heads, hidden_dim, device=my_device, dtype=type)
    if my_rank == root_rank:
        print("\nUlysses attention")
        print(f"input [N/P, d]: {input.shape}, elements: {input.nelement()}, size: {input.element_size() * input.nelement() / 1e6:.2f} MB")
        print(f"Q [h, d, d/h]: {Q.shape}, elements: {Q.nelement()}, size: {Q.element_size() * Q.nelement() / 1e6:.2f} MB")
        print(f"K [h, d, d/h]: {K.shape}, elements: {K.nelement()}, size: {K.element_size() * K.nelement() / 1e6:.2f} MB")
        print(f"V [h, d, d/h]: {V.shape}, elements: {V.nelement()}, size: {V.element_size() * V.nelement() / 1e6:.2f} MB")
        print(f"O [h, d/h, d]: {O.shape}, elements: {O.nelement()}, size: {O.element_size() * O.nelement() / 1e6:.2f} MB")
        torch.cuda.synchronize()
        print(f"Peak memory allocation: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    # compute q, k, v
    q = torch.matmul(input, Q) # [h, N/P, d/h]
    k = torch.matmul(input, K) # [h, N/P, d/h]
    v = torch.matmul(input, V) # [h, N/P, d/h]
    if my_rank == root_rank:
        print("compute q, k, v")
        print(f"q = input x Q, k = input x K, v = input x V")
        print(f"flops: {3 * 2 * seq_length * hidden_dim * hidden_dim / 1e12:.2f} TFLOPs")
        print(f"q [h, N/P, d/h]: {q.shape}, elements: {q.nelement()}, size {q.element_size() * q.nelement() / 1e6:.2f} MB")
        print(f"k [h, N/P, d/h]: {k.shape}, elements: {k.nelement()}, size {k.element_size() * k.nelement() / 1e6:.2f} MB")
        print(f"v [h, N/P, d/h]: {v.shape}, elements: {v.nelement()}, size {v.element_size() * v.nelement() / 1e6:.2f} MB")
        torch.cuda.synchronize()
        print(f"Peak memory allocation: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    # all-to-all q, k, v
    q_ = torch.empty(P, num_heads//P, seq_length//P, hidden_dim//num_heads, device=my_device, dtype=type)
    k_ = torch.empty_like(q_)
    v_ = torch.empty_like(q_)
    dist.all_to_all_single(q_, q)
    dist.all_to_all_single(k_, k)
    dist.all_to_all_single(v_, v)
    if my_rank == root_rank:
        print("q_, k_, v_ = all-to-all q, k, v")
        print(f"q_ [P, h/P, N/P, d/h]: {q_.shape}, elements: {q_.nelement()}, size {q_.element_size() * q_.nelement() / 1e6:.2f} MB")
        print(f"k_ [P, h/P, N/P, d/h]: {k_.shape}, elements: {k_.nelement()}, size {k_.element_size() * k_.nelement() / 1e6:.2f} MB")
        print(f"v_ [P, h/P, N/P, d/h]: {v_.shape}, elements: {v_.nelement()}, size {v_.element_size() * v_.nelement() / 1e6:.2f} MB")
        torch.cuda.synchronize()
        print(f"Peak memory allocation: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    q_ = torch.transpose(q_, 0, 1)
    k_ = torch.transpose(k_, 0, 1)
    v_ = torch.transpose(v_, 0, 1)
    q_ = torch.reshape(q_, (num_heads//P, seq_length, hidden_dim//num_heads))
    k_ = torch.reshape(k_, (num_heads//P, seq_length, hidden_dim//num_heads))
    v_ = torch.reshape(v_, (num_heads//P, seq_length, hidden_dim//num_heads))
    if my_rank == root_rank:
        print("transpose(0, 1) q_, k_, v_ & reshape")
        print(f"q_ [h/P, N, d/h]: {q_.shape}, elements: {q_.nelement()}, size {q_.element_size() * q_.nelement() / 1e6:.2f} MB")
        print(f"k_ [h/P, N, d/h]: {k_.shape}, elements: {k_.nelement()}, size {k_.element_size() * k_.nelement() / 1e6:.2f} MB")
        print(f"v_ [h/P, N, d/h]: {v_.shape}, elements: {v_.nelement()}, size {v_.element_size() * v_.nelement() / 1e6:.2f} MB")
        print(f"is_contiguous: {q_.is_contiguous()}")
        torch.cuda.synchronize()
        print(f"Peak memory allocation: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    # compute attention
    A = torch.matmul(q_, torch.transpose(k_, 1, 2))
    if my_rank == root_rank:
        print("compute attention")
        print(f"A = q_ x k_t")
        print(f"flops: {2 * seq_length * seq_length * hidden_dim /1e12:.2f} TFLOPs")
        print(f"A [h/P, N, N]: {A.shape}, elements: {A.nelement()}, size {A.element_size() * A.nelement() / 1e9:.2f} GB")
        print(f"is_contiguous: {A.is_contiguous()}")
        torch.cuda.synchronize()
        print(f"Peak memory allocation: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    # softmax A
    # A = torch.nn.functional.softmax(A, dim=-1) runs out of memory
    # compute c
    c = torch.matmul(A, v_)
    if my_rank == root_rank:
        print("compute c")
        print(f"c = A x v_")
        print(f"flops: {2 * seq_length * seq_length * hidden_dim / 1e12:.2f} TFLOPs")
        print(f"c [h/P, N, d/h]: {c.shape}, elements: {c.nelement()}, size {c.element_size() * c.nelement() / 1e6:.2f} MB")
        print(f"is_contiguous: {c.is_contiguous()}")
        torch.cuda.synchronize()
        print(f"Peak memory allocation: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    # all-to-all c
    c = torch.transpose(c, 0, 1).contiguous()
    if my_rank == root_rank:
        print("transpose(0, 1) c")
        print(f"c [N, h/P, d/h]: {c.shape}, elements: {c.nelement()}, size {c.element_size() * c.nelement() / 1e6:.2f} MB")
        print(f"is_contiguous: {c.is_contiguous()}")
        torch.cuda.synchronize()
        print(f"Peak memory allocation: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    c_ = torch.empty(P, seq_length//P, num_heads//P, hidden_dim//num_heads, device=my_device, dtype=type)
    dist.all_to_all_single(c_, c)
    if my_rank == root_rank:
        print("c_ = all-to-all c")
        print(f"c_ [P, N/P, h/P, d/h]: {c_.shape}, elements: {c_.nelement()}, size {c_.element_size() * c_.nelement() / 1e6:.2f} MB")
        torch.cuda.synchronize()
        print(f"Peak memory allocation: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    c_ = torch.transpose(c_, 0, 1)
    c_ = torch.reshape(c_, (seq_length//P, hidden_dim))
    O = torch.reshape(O, (hidden_dim, hidden_dim))
    if my_rank == root_rank:
        print("transpose(0, 1) c_ & reshape c_ and O")
        print(f"c_ [N/P, d]: {c_.shape}, elements: {c_.nelement()}, size {c_.element_size() * c_.nelement() / 1e6:.2f} MB")
        print(f"is_contiguous: {c_.is_contiguous()}")
        print(f"O [d, d]: {O.shape}, elements: {O.nelement()}, size {O.element_size() * O.nelement() / 1e6:.2f} MB")
        torch.cuda.synchronize()
        print(f"Peak memory allocation: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    # compute output
    output = torch.matmul(c_, O)
    if my_rank == root_rank:
        print("compute output")
        print(f"output = c x O")
        print(f"flops: {2 * seq_length * hidden_dim * hidden_dim / 1e12:.2f} TFLOPs")
        print(f"output [N/P, d]: {output.shape}, elements: {output.nelement()}, size {output.element_size() * output.nelement() / 1e6:.2f} MB")
        torch.cuda.synchronize()
        print(f"Peak memory allocation: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    return output

def ulysses_allgather(seq_length, hidden_dim, num_heads, P) -> torch.Tensor:
    # initialize input and model
    # input [N/P, d]
    # Q, K, V [h, d, d/h]
    # proj [h, d/h, d]
    input = torch.randn(seq_length//P, hidden_dim, device=my_device, dtype=type)
    Q = torch.ones(num_heads, hidden_dim, hidden_dim//num_heads, device=my_device, dtype=type)
    K = torch.ones_like(Q)
    V  = torch.ones_like(Q)
    O = torch.ones(num_heads, hidden_dim//num_heads, hidden_dim, device=my_device, dtype=type)
    if my_rank == root_rank:
        print("\n")
        print("\nAll-gather attention")
        print(f"input [N/P, d]: {input.shape}, elements: {input.nelement()}, size: {input.element_size() * input.nelement() / 1e6:.2f} MB")
        print(f"Q [h, d, d/h]: {Q.shape}, elements: {Q.nelement()}, size: {Q.element_size() * Q.nelement() / 1e6:.2f} MB")
        print(f"K [h, d, d/h]: {K.shape}, elements: {K.nelement()}, size: {K.element_size() * K.nelement() / 1e6:.2f} MB")
        print(f"V [h, d, d/h]: {V.shape}, elements: {V.nelement()}, size: {V.element_size() * V.nelement() / 1e6:.2f} MB")
        print(f"O [h, d/h, d]: {O.shape}, elements: {O.nelement()}, size: {O.element_size() * O.nelement() / 1e6:.2f} MB")
        torch.cuda.synchronize()
        print(f"Peak memory allocation: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    # compute q, k, v
    q = torch.matmul(input, Q) # [h, N/P, d/h]
    k = torch.matmul(input, K) # [h, N/P, d/h]
    v = torch.matmul(input, V) # [h, N/P, d/h]
    if my_rank == root_rank:
        print("compute q, k, v")
        print(f"q = input x Q, k = input x K, v = input x V")
        print(f"flops: {3 * 2 * seq_length * hidden_dim * hidden_dim / 1e12:.2f} TFLOPs")
        print(f"q [h, N/P, d/h]: {q.shape}, elements: {q.nelement()}, size {q.element_size() * q.nelement() / 1e6:.2f} MB")
        print(f"k [h, N/P, d/h]: {k.shape}, elements: {k.nelement()}, size {k.element_size() * k.nelement() / 1e6:.2f} MB")
        print(f"v [h, N/P, d/h]: {v.shape}, elements: {v.nelement()}, size {v.element_size() * v.nelement() / 1e6:.2f} MB")
        torch.cuda.synchronize()
        print(f"Peak memory allocation: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    # all-gather k, v
    k_ = torch.empty(P, num_heads, seq_length//P, hidden_dim//num_heads, device=my_device, dtype=type)
    v_ = torch.empty_like(k_)
    dist.all_gather_into_tensor(k_, k)
    dist.all_gather_into_tensor(v_, v)
    if my_rank == root_rank:
        print("all-gather k, v")
        print(f"k_ [P, h, N/P, d/h]: {k_.shape}, elements: {k_.nelement()}, size {k_.element_size() * k_.nelement() / 1e6:.2f} MB")
        print(f"v_ [P, h, N/P, d/h]: {v_.shape}, elements: {v_.nelement()}, size {v_.element_size() * v_.nelement() / 1e6:.2f} MB")
        torch.cuda.synchronize()
        print(f"Peak memory allocation: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    #transpose k_ and v_
    k_ = torch.transpose(k_, 0, 1)
    v_ = torch.transpose(v_, 0, 1)
    k_ = torch.reshape(k_, (num_heads, seq_length, hidden_dim//num_heads))
    v_ = torch.reshape(v_, (num_heads, seq_length, hidden_dim//num_heads))
    if my_rank == root_rank:
        print("transpose(0, 1) k_, v_ & reshape")
        print(f"k_ [h, N, d/h]: {k_.shape}, elements: {k_.nelement()}, size {k_.element_size() * k_.nelement() / 1e6:.2f} MB")
        print(f"v_ [h, N, d/h]: {v_.shape}, elements: {v_.nelement()}, size {v_.element_size() * v_.nelement() / 1e6:.2f} MB")
        print(f"is_contiguous: {k_.is_contiguous()}")
        torch.cuda.synchronize()
        print(f"Peak memory allocation: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    # compute attention
    # A = torch.matmul(q, torch.transpose(k_, 1, 2))
    A = torch.matmul(q, k_.transpose(1, 2))
    if my_rank == root_rank:
        print("compute attention")
        print(f"A = q x k_t")
        print(f"flops: {2 * seq_length * seq_length * hidden_dim /1e12:.2f} TFLOPs")
        print(f"A [h, N/P, N]: {A.shape}, elements: {A.nelement()}, size {A.element_size() * A.nelement() / 1e9:.2f} GB")
        print(f"is_contiguous: {A.is_contiguous()}")
        torch.cuda.synchronize()
        print(f"Peak memory allocation: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    # softmax A
    # A = torch.nn.functional.softmax(A, dim=-1) runs out of memory
    # compute c
    c = torch.matmul(A, v_)
    if my_rank == root_rank:
        print("compute c")
        print(f"c = A x v_")
        print(f"flops: {2 * seq_length * seq_length * hidden_dim / 1e12:.2f} TFLOPs")
        print(f"c [h, N/P, d/h]: {c.shape}, elements: {c.nelement()}, size {c.element_size() * c.nelement() / 1e6:.2f} MB")
        print(f"is_contiguous: {c.is_contiguous()}")
        torch.cuda.synchronize()
        print(f"Peak memory allocation: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    # transpose c
    c = torch.transpose(c, 0, 1)#.contiguous()
    c = torch.reshape(c, (seq_length//P, hidden_dim))
    if my_rank == root_rank:
        print("transpose(0, 1) & reshape c")
        print(f"c [N/P, d]: {c.shape}, elements: {c.nelement()}, size {c.element_size() * c.nelement() / 1e6:.2f} MB")
        print(f"is_contiguous: {c.is_contiguous()}")
        torch.cuda.synchronize()
        print(f"Peak memory allocation: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    # compute output
    output = torch.matmul(c, torch.reshape(O, (hidden_dim, hidden_dim)))
    if my_rank == root_rank:
        print("compute output")
        print(f"output = c x O")
        print(f"flops: {2 * seq_length * hidden_dim * hidden_dim / 1e12:.2f} TFLOPs")
        print(f"output [N/P, d]: {output.shape}, elements: {output.nelement()}, size {output.element_size() * output.nelement() / 1e6:.2f} MB")
        print(f"is_contiguous: {output.is_contiguous()}")
        torch.cuda.synchronize()
        print(f"Peak memory allocation: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    return None


def ulysses_2D_rowwise(seq_length, hidden_dim, num_heads, type, HP, SP) -> torch.Tensor:
    # initialize input and model
    # input [N/SP, d]
    # Q, K, V [h/HP, d/SP, d/h]
    # proj [h/HP, d/h, d]
    input = torch.randn(seq_length//SP, hidden_dim, device=my_device, dtype=type)
    Q = torch.ones(num_heads//HP, hidden_dim//SP, hidden_dim//num_heads, device=my_device, dtype=type)
    K = torch.ones_like(Q)
    V  = torch.ones_like(Q)
    proj = torch.ones(num_heads//HP, hidden_dim//num_heads, hidden_dim, device=my_device, dtype=type)

    if my_rank == root_rank:
        print("\n2D Ulysses Attention")
        print(f"input: {input.shape}, elements: {input.nelement()}, size: {input.element_size() * input.nelement() / 1e9:.2f} GB")
        print(f"Q shape: {Q.shape}, elements: {Q.nelement()}, size: {Q.element_size() * Q.nelement() / 1e6:.2f} MB")
        print(f"K shape: {K.shape}, elements: {K.nelement()}, size: {K.element_size() * K.nelement() / 1e6:.2f} MB")
        print(f"V shape: {V.shape}, elements: {V.nelement()}, size: {V.element_size() * V.nelement() / 1e6:.2f} MB")
        print(f"proj shape: {proj.shape}, elements: {proj.nelement()}, size: {proj.element_size() * proj.nelement() / 1e6:.2f} MB")

    # Create group communicators
    ranks = [i for i in range(world_size) if i // SP == my_rank // SP]
    # print("myid: " + str(my_rank) + " ranks " + str(ranks) + "\n")
    group_TP = dist.new_group(ranks, use_local_synchronization=True)
    ranks = [i for i in range(world_size) if i // HP == my_rank // HP]
    group_HP = dist.new_group(ranks, use_local_synchronization=True)

    Q_ = torch.empty(SP, num_heads//HP, hidden_dim//SP, hidden_dim//num_heads, device=my_device, dtype=type)
    K_ = torch.empty_like(Q_)
    V_ = torch.empty_like(Q_)
    if my_rank == root_rank:
        print("all-gather Q, K, V")
        print(f"Q_ shape: {Q_.shape}, elements: {Q_.nelement()}, size: {Q_.element_size() * Q_.nelement() / 1e6:.2f} MB")
        print(f"K_ shape: {K_.shape}, elements: {K_.nelement()}, size: {K_.element_size() * K_.nelement() / 1e6:.2f} MB")
        print(f"V_ shape: {V_.shape}, elements: {V_.nelement()}, size: {V_.element_size() * V_.nelement() / 1e6:.2f} MB")
    # all-gather
    dist.all_gather_into_tensor(Q_, Q, group=group_TP)
    dist.all_gather_into_tensor(K_, Q, group=group_TP)
    dist.all_gather_into_tensor(V_, Q, group=group_TP)
    # transpose
    Q_ = torch.reshape(Q_.transpose(0, 1), (num_heads//HP, hidden_dim, hidden_dim//num_heads))
    K_ = torch.reshape(K_.transpose(0, 1), (num_heads//HP, hidden_dim, hidden_dim//num_heads))
    V_ = torch.reshape(V_.transpose(0, 1), (num_heads//HP, hidden_dim, hidden_dim//num_heads))

    if my_rank == root_rank:
        print("reshape Q_, K_, V_")
        print(f"Q_ shape: {Q_.shape}, elements: {Q_.nelement()}, size: {Q_.element_size() * Q_.nelement() / 1e6:.2f} MB")
        print(f"K_ shape: {K_.shape}, elements: {K_.nelement()}, size: {K_.element_size() * K_.nelement() / 1e6:.2f} MB")
        print(f"V_ shape: {V_.shape}, elements: {V_.nelement()}, size: {V_.element_size() * V_.nelement() / 1e6:.2f} MB")
        print(f"Torch memory allocation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # compute q, k, v
    q = torch.matmul(input, Q_)
    k = torch.matmul(input, K_)
    v = torch.matmul(input, V_)

    if my_rank == root_rank:
        print("compute q, k, v")
        print(f"inputxQ=q + inputxK=k + inputxV=v flops: {3 * 2 * seq_length * hidden_dim * hidden_dim / 1e12:.2f} TFLOPs")
        print(f"q shape: {q.shape}, elements: {q.nelement()}, size {q.element_size() * q.nelement() / 1e6:.2f} MB")
        print(f"k shape: {k.shape}, elements: {k.nelement()}, size {k.element_size() * k.nelement() / 1e6:.2f} MB")
        print(f"v shape: {v.shape}, elements: {v.nelement()}, size {v.element_size() * v.nelement() / 1e6:.2f} MB")
        print(f"Torch memory allocation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # all-gather k and v
    k_ = torch.empty(SP, num_heads//HP, seq_length//SP, hidden_dim//num_heads, device=my_device, dtype=type)
    v_ = torch.empty_like(k_)
    if my_rank == root_rank:
        print("all-gather k and v")
        print(f"k_ shape: {k_.shape}, elements: {k_.nelement()}, size {k_.element_size() * k_.nelement() / 1e6:.2f} MB")
        print(f"v_ shape: {v_.shape}, elements: {v_.nelement()}, size {v_.element_size() * v_.nelement() / 1e6:.2f} MB")
        print(f"Torch memory allocation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    dist.all_gather_into_tensor(k_, k, group=group_TP)
    dist.all_gather_into_tensor(v_, v, group=group_TP)

    # transpose k_ and v_
    k_ = torch.reshape(k_.transpose(0, 1), (num_heads//HP, seq_length, hidden_dim//num_heads))
    v_ = torch.reshape(v_.transpose(0, 1), (num_heads//HP, seq_length, hidden_dim//num_heads))
    if my_rank == root_rank:
        print("transpose k_ and v_")
        print(f"k_ shape: {k_.shape}, elements: {k_.nelement()}, size {k_.element_size() * k_.nelement() / 1e6:.2f} MB")
        print(f"v_ shape: {v_.shape}, elements: {v_.nelement()}, size {v_.element_size() * v_.nelement() / 1e6:.2f} MB")
        print(f"Torch memory allocation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # compute attention
    A = torch.matmul(q, k_.transpose(1, 2))
    if my_rank == root_rank:
        print("compute attention")
        print(f"A=qxk' flops: {2 * seq_length * seq_length * hidden_dim /1e12:.2f} TFLOPs")
        # print(A)
        print(f"A shape: {A.shape}, elements: {A.nelement()}, size {A.element_size() * A.nelement() / 1e9:.2f} GB")
        print(f"Torch memory allocation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # calculate softmax
    # A = torch.nn.functional.softmax(A, dim=-1) # softmax along rows of A

    c = torch.matmul(A, v_)
    if my_rank == root_rank:
        # print(c)
        print("compute c")
        print(f"c shape: {c.shape}, elements: {c.nelement()}, size {c.element_size() * c.nelement() / 1e6:.2f} MB")
        print(f"Torch memory allocation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    c = torch.reshape(c.transpose(0, 1), (seq_length//SP, hidden_dim//HP))
    if my_rank == root_rank:
        print("transpose c")
        print(f"c shape: {c.shape}, elements: {c.nelement()}, size {c.element_size() * c.nelement() / 1e6:.2f} MB")
        print(f"Torch memory allocation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    output = torch.matmul(c, torch.reshape(proj, (hidden_dim//HP, hidden_dim)))
    if my_rank == root_rank:
        print(f"output shape: {output.shape}, elements: {output.nelement()}, size {output.element_size() * output.nelement() / 1e9:.2f} GB")
        print(f"Torch memory allocation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # all-reduce output
    dist.all_reduce(output, group=group_HP)

    return output

    layer = torch.matmul(Q, K.transpose(0, 1))
    qk = torch.matmul(torch.matmul(input, layer), input.transpose(0, 1))
    if my_rank == root_rank:
        print(f"hidden layer shape: {layer.shape}, elements: {layer.nelement()}, size: {layer.element_size() * layer.nelement() / 1e9:.2f} GB")
        print(f"Torch memory allocation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    qk = torch.nn.functional.softmax(qk, dim=-1)
    if my_rank == root_rank:
        print(f"qk shape: {qk.shape}, elements: {qk.nelement()}, size {qk.element_size() * qk.nelement() / 1e9:.2f} GB")
        print(f"Torch memory allocation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    # torch.exp(temp, out=temp)
    # summed = torch.sum(temp, dim=1, keepdim=True)
    # temp /= summed

    c_ = torch.matmul(qk, torch.matmul(input, V))
    if my_rank == root_rank:
        print(f"c_ shape: {c_.shape}, elements: {c_.nelement()}, size {c_.element_size() * c_.nelement() / 1e9:.2f} GB")
        print(f"Torch memory allocation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")



    K_T = K.transpose(0, 1)
    c_ = torch.matmul(torch.nn.functional.softmax(torch.matmul(torch.matmul(input, torch.matmul(Q, K_T)), input.transpose(0, 1)), dim=-1), torch.matmul(input, V))
    c =  torch.matmul(torch.nn.functional.softmax(torch.matmul(torch.matmul(input, Q), torch.matmul(input, K).transpose(0, 1)), dim=-1), torch.matmul(input, V))

    # Compare c and c_
    if my_rank == root_rank:
        print(c)
        print(c_)
        atol = 1e-6
        if torch.allclose(c, c_, atol=atol):
            print(f"c and c_ are equal within {atol} tolerance.\n")
        else:
            print("c and c_ are not equal.")

# main

# initialize
dist.init_process_group(backend='nccl')
my_rank = dist.get_rank()
world_size = dist.get_world_size()
torch.cuda.set_device(my_rank % torch.cuda.device_count())
my_device = torch.cuda.current_device()
root_rank = 7

# model parameters
seq_length = 70000  # N
hidden_dim = 16384  # d
num_heads = 128     # h
inter_size = 53248  # d'
num_layers = 126    # L
type = torch.bfloat16

# parallelization parameters
TP = 8
DP = 2
P = TP * DP
if P != world_size:
    raise ValueError("P must equal world_size")

# report parameters
if my_rank == root_rank:
    print("my_rank " + str(my_rank) + "/" + str(world_size) + " my_device " + str(my_device) + "/" + str(torch.cuda.device_count()) + "\n")
    print("seq length: " + str(seq_length))
    print("hidden dim: " + str(hidden_dim))
    # print("num layers: " + str(num_layers))
    print("num heads: " + str(num_heads))
    print("type: " + str(type))
    print("TP: " + str(TP))
    print("DP: " + str(DP))
    print("P: " + str(P))
    print("head per GPU: " + str(num_heads//P) + " tokens per GPU: " + str(seq_length//P))

torch.cuda.synchronize()
dist.barrier()
att_out = ulysses_attention(seq_length, hidden_dim, num_heads, P)
torch.cuda.synchronize()
torch.cuda.empty_cache()
'''input_ = torch.empty(seq_length, hidden_dim, device=my_device, dtype=type)
dist.all_gather_into_tensor(input_, att_out)
MLP_out = MLP_model(seq_length, hidden_dim, inter_size, num_layers, P, input_)
del input_
torch.cuda.synchronize()
torch.cuda.empty_cache()
'''
# initialize group communicator
ranks = [i for i in range(world_size) if i // TP == my_rank // TP]
if my_rank == root_rank:
    print("TP ranks: " + str(ranks))
group_TP = dist.new_group(ranks, use_local_synchronization=True)
# all-gather input
input_ = torch.empty(seq_length//DP, hidden_dim, device=my_device, dtype=type)
dist.all_gather_into_tensor(input_, att_out, group=group_TP)
MLP_2D_out = MLP_2D(seq_length, hidden_dim, inter_size, num_layers, TP, DP, input_, group_TP)
del input_
torch.cuda.synchronize()
torch.cuda.empty_cache()
