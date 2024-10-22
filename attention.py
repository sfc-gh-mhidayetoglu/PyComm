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
seq_length = 5000 # 10000 # 100000
hidden_dim = 16384
num_layers = 126
num_heads = 128

# parallelization parameters
HP = 2 # parallelize among heads (embarrassimgly parallel)
SP = 8 # parallelize among sequence length (communication)
assert HP * SP == world_size, f"HP x SP must equal world_size, but got HP={HP}, SP={SP}, world_size={world_size}"

# report parameters
if my_rank == root_rank:
    print("my_rank " + str(my_rank) + "/" + str(world_size) + " my_device " + str(my_device) + "/" + str(torch.cuda.device_count()) + "\n")
    print("seq length: " + str(seq_length))
    print("hidden dim: " + str(hidden_dim))
    print("num layers: " + str(num_layers))
    print("num heads: " + str(num_heads))
    print("HP: " + str(HP) + " head parallelism")
    print("SP: " + str(SP) + " sequence parallelism")
    print("HP x SP: " + str(HP * SP) + " world size\n")
    print("head per GPU: " + str(num_heads//HP) + "tokens per GPU: " + str(seq_length//SP) + "\n")

# initialize input and model
# input [N/SP, d]
input = torch.randn(seq_length // SP, hidden_dim, device=my_device) # [N/SP, d]
Q = torch.ones(num_heads // HP, hidden_dim // SP, hidden_dim // num_heads, device=my_device)
K = torch.ones_like(Q) # [d/SP, h/HP, d/h]
V  = torch.ones_like(Q) # [d/SP, h/HP, d/h]

if my_rank == root_rank:
    # print(input)
    print(f"D (input): {input.shape}, elements: {input.nelement()}, size: {input.element_size() * input.nelement() / 1e8:.2f} GB")
    # print(Q)
    print(f"Q shape: {Q.shape}, elements: {Q.nelement()}, size: {Q.element_size() * Q.nelement() / 1e6:.2f} MB")
    # print(K)
    print(f"K shape: {K.shape}, elements: {K.nelement()}, size: {K.element_size() * K.nelement() / 1e6:.2f} MB")
    # print(V)
    print(f"V shape: {V.shape}, elements: {V.nelement()}, size: {V.element_size() * V.nelement() / 1e6:.2f} MB")

# Create group communicators
ranks = [i for i in range(world_size) if i // SP == my_rank // SP]
print("myid: " + str(my_rank) + " ranks " + str(ranks) + "\n")
group_TP = dist.new_group(ranks, use_local_synchronization=True)

Q_ = torch.empty(num_heads//HP, hidden_dim, hidden_dim//num_heads, device=my_device)
K_ = torch.empty_like(Q_)
V_ = torch.empty_like(Q_)
dist.all_gather_into_tensor(Q_, Q, group=group_TP)
dist.all_gather_into_tensor(K_, Q, group=group_TP)
dist.all_gather_into_tensor(V_, Q, group=group_TP)

if my_rank == root_rank:
    print(f"Q_ shape: {Q_.shape}, elements: {Q_.nelement()}, size: {Q_.element_size() * Q_.nelement() / 1e6:.2f} MB")
    print(f"K_ shape: {K_.shape}, elements: {K_.nelement()}, size: {K_.element_size() * K_.nelement() / 1e6:.2f} MB")
    print(f"V_ shape: {V_.shape}, elements: {V_.nelement()}, size: {V_.element_size() * V_.nelement() / 1e6:.2f} MB")

Q_ = torch.reshape(Q_, (num_heads//HP, hidden_dim, hidden_dim//num_heads))
K_ = torch.reshape(K_, (num_heads//HP, hidden_dim, hidden_dim//num_heads))
V_ = torch.reshape(V_, (num_heads//HP, hidden_dim, hidden_dim//num_heads))

if my_rank == root_rank:
    print(f"Q_ shape: {Q_.shape}, elements: {Q_.nelement()}, size: {Q_.element_size() * Q_.nelement() / 1e6:.2f} MB")
    print(f"K_ shape: {K_.shape}, elements: {K_.nelement()}, size: {K_.element_size() * K_.nelement() / 1e6:.2f} MB")
    print(f"V_ shape: {V_.shape}, elements: {V_.nelement()}, size: {V_.element_size() * V_.nelement() / 1e6:.2f} MB")
    print(f"Torch memory allocation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# compute Q, K, V
q = torch.matmul(input, Q_.reshape(num_heads//HP, hidden_dim, hidden_dim//num_heads))
k = torch.matmul(input, K_.reshape(num_heads//HP, hidden_dim, hidden_dim//num_heads))
v = torch.matmul(input, V_.reshape(num_heads//HP, hidden_dim, hidden_dim//num_heads))

if my_rank == root_rank:
    print(f"DxQ=q + DxK=k + DxV=v flops: {num_heads // HP * 3 * (2 * seq_length // SP * hidden_dim * hidden_dim // num_heads)/1e9:.2f} GFLOPs")
    # print(q)
    print(f"q shape: {q.shape}, elements: {q.nelement()}, size {q.element_size() * q.nelement() / 1e6:.2f} MB")
    # print(k)
    print(f"k shape: {k.shape}, elements: {k.nelement()}, size {k.element_size() * k.nelement() / 1e6:.2f} MB")
    # print(v)
    print(f"v shape: {v.shape}, elements: {v.nelement()}, size {v.element_size() * v.nelement() / 1e6:.2f} MB")
    print(f"Torch memory allocation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

k_ = torch.empty(SP, num_heads//HP, seq_length//SP, hidden_dim//num_heads, device=my_device)
if my_rank == root_rank:
    print(f"k_ shape: {k_.shape}, elements: {k_.nelement()}, size {k_.element_size() * k_.nelement() / 1e6:.2f} MB")
    print(f"Torch memory allocation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
dist.all_gather_into_tensor(k_, k, group=group_TP)

k_ = torch.reshape(k_, (num_heads//HP, seq_length, hidden_dim//num_heads))

if my_rank == root_rank:
    print(f"k_ shape: {k_.shape}, elements: {k_.nelement()}, size {k_.element_size() * k_.nelement() / 1e6:.2f} MB")
    print(f"Torch memory allocation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# compute attention
A = torch.matmul(q, k_.transpose(1, 2))
if my_rank == root_rank:
    print(f"A=qxk' flops: {num_heads // HP * 2 * (seq_length * seq_length * hidden_dim // num_heads)/1e9:.2f} GFLOPs")
    # print(A)
    print(f"A shape: {A.shape}, elements: {A.nelement()}, size {A.element_size() * A.nelement() / 1e9:.2f} GB")
    print(f"Torch memory allocation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

exit()

# calculate softmax
A = torch.nn.functional.softmax(A, dim=-1)
# in-place softmax
# torch.exp(A, out=A)
# summed = torch.sum(A, dim=1, keepdim=True)
# A /= summed
# compute scores
c = torch.matmul(A, v)
if my_rank == root_rank:
    print(A)
    print(f"scores shape: {A.shape}, elements: {A.nelement()}, size {A.element_size() * A.nelement() / 1e9:.2f} GB")
    print(c)
    print(f"c shape: {c.shape}, elements: {c.nelement()}, size {c.element_size() * c.nelement() / 1e9:.2f} GB")
    print(f"Torch memory allocation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


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




