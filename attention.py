import torch
import torch.distributed as dist

# initialize
dist.init_process_group(backend='nccl')
my_rank = dist.get_rank()
world_size = dist.get_world_size()
torch.cuda.set_device(my_rank % torch.cuda.device_count())
my_device = torch.cuda.current_device()
root_rank = 7

# model parameters
seq_length = 60000 # 10000 # 100000
hidden_dim = 16384
num_heads = 128
type = torch.bfloat16
# num_layers = 126

# parallelization parameters
HP = 1 # parallelize among heads (embarrassimgly parallel)
SP = 16 # parallelize among sequence length (communication)

P = HP * SP
assert P == world_size, f"HP x SP must equal world_size, but got HP={HP}, SP={SP}, world_size={world_size}"

# report parameters
if my_rank == root_rank:
    print("my_rank " + str(my_rank) + "/" + str(world_size) + " my_device " + str(my_device) + "/" + str(torch.cuda.device_count()) + "\n")
    print("seq length: " + str(seq_length))
    print("hidden dim: " + str(hidden_dim))
    # print("num layers: " + str(num_layers))
    print("num heads: " + str(num_heads))
    print("HP: " + str(HP) + " head parallelism")
    print("SP: " + str(SP) + " sequence parallelism")
    print("P = HP x SP: " + str(HP * SP))
    print("head per GPU: " + str(num_heads//HP) + "tokens per GPU: " + str(seq_length//SP) + "\n")

def ulysses(seq_length, hidden_dim, num_heads, P) -> torch.Tensor:
    # initialize input and model
    # input [N/P, d]
    # Q, K, V [h, d, d/h]
    # q, k, v [h, N/P, d/h]
    # proj [h, d/h, d]
    input = torch.randn(seq_length//P, hidden_dim, device=my_device, dtype=type)
    Q = torch.ones(num_heads, hidden_dim, hidden_dim//num_heads, device=my_device, dtype=type)
    K = torch.ones_like(Q)
    V  = torch.ones_like(Q)
    proj = torch.ones(num_heads, hidden_dim//num_heads, hidden_dim, device=my_device, dtype=type)
    if my_rank == root_rank:
        print("\nUlysses attention")
        print(f"input [N/P, d]: {input.shape}, elements: {input.nelement()}, size: {input.element_size() * input.nelement() / 1e9:.2f} GB")
        print(f"Q [h, d, d/h]: {Q.shape}, elements: {Q.nelement()}, size: {Q.element_size() * Q.nelement() / 1e6:.2f} MB")
        print(f"K [h, d, d/h]: {K.shape}, elements: {K.nelement()}, size: {K.element_size() * K.nelement() / 1e6:.2f} MB")
        print(f"V [h, d, d/h]: {V.shape}, elements: {V.nelement()}, size: {V.element_size() * V.nelement() / 1e6:.2f} MB")
        print(f"proj [h, d/h, d]: {proj.shape}, elements: {proj.nelement()}, size: {proj.element_size() * proj.nelement() / 1e6:.2f} MB")
    # compute q, k, v
    q = torch.matmul(input, Q) # [h, N/P, d/h]
    k = torch.matmul(input, K) # [h, N/P, d/h]
    v = torch.matmul(input, V) # [h, N/P, d/h]
    if my_rank == root_rank:
        print("compute q, k, v")
        print(f"q=inputxQ + k=inputxK + v=inputxV flops: {3 * 2 * seq_length * hidden_dim * hidden_dim / 1e12:.2f} TFLOPs")
        print(f"q [h, N/P, d/h]: {q.shape}, elements: {q.nelement()}, size {q.element_size() * q.nelement() / 1e6:.2f} MB")
        print(f"k [h, N/P, d/h]: {k.shape}, elements: {k.nelement()}, size {k.element_size() * k.nelement() / 1e6:.2f} MB")
        print(f"v [h, N/P, d/h]: {v.shape}, elements: {v.nelement()}, size {v.element_size() * v.nelement() / 1e6:.2f} MB")
        print(f"Torch memory allocation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    # all-to-all q, k, v
    q_ = torch.empty(num_heads//P, seq_length, hidden_dim//num_heads, device=my_device, dtype=type)
    k_ = torch.empty_like(q_)
    v_ = torch.empty_like(q_)
    if my_rank == root_rank:
        print("all-to-all q, k, v")
        print(f"q_ [h/P, N, d/h]: {q_.shape}, elements: {q_.nelement()}, size {q_.element_size() * q_.nelement() / 1e6:.2f} MB")
        print(f"k_ [h/P, N, d/h]: {k_.shape}, elements: {k_.nelement()}, size {k_.element_size() * k_.nelement() / 1e6:.2f} MB")
        print(f"v_ [h/P, N, d/h]: {v_.shape}, elements: {v_.nelement()}, size {v_.element_size() * v_.nelement() / 1e6:.2f} MB")
        print(f"Torch memory allocation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    dist.all_to_all_single(q_, q)
    dist.all_to_all_single(k_, k)
    dist.all_to_all_single(v_, v)
    # compute attention
    A = torch.matmul(q_, k_.transpose(1, 2))
    if my_rank == root_rank:
        print("compute attention")
        print(f"A=qxk' flops: {2 * seq_length * seq_length * hidden_dim /1e12:.2f} TFLOPs")
        print(f"A [h/P, N, N]: {A.shape}, elements: {A.nelement()}, size {A.element_size() * A.nelement() / 1e9:.2f} GB")
        print(f"Torch memory allocation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    # compute c
    c = torch.matmul(A, v_)
    if my_rank == root_rank:
        print("compute c")
        print(f"c [h/p, N, d/h]: {c.shape}, elements: {c.nelement()}, size {c.element_size() * c.nelement() / 1e6:.2f} MB")
        print(f"Torch memory allocation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    # all-to-all c
    c_ = torch.empty(num_heads, seq_length//P, hidden_dim//num_heads, device=my_device, dtype=type)
    if my_rank == root_rank:
        print("all-to-all c")
        print(f"c_ [h, N/P, d/h]: {c_.shape}, elements: {c_.nelement()}, size {c_.element_size() * c_.nelement() / 1e6:.2f} MB")
        print(f"Torch memory allocation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    dist.all_to_all_single(c_, c)
    # compute output
    output = torch.matmul(c_, proj)
    if my_rank == root_rank:
        print(f"output [N/P, d]: {output.shape}, elements: {output.nelement()}, size {output.element_size() * output.nelement() / 1e9:.2f} GB")
        print(f"Torch memory allocation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return output



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

ulysses(seq_length, hidden_dim, num_heads, P)
# ulysses_2D_rowwise(seq_length, hidden_dim, num_heads, type, HP, SP)




