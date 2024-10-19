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
seq_length = 100000 # 100000
hidden_dim = 16384 * 2
num_layers = 126

# parallelization parameters
num_heads = 16

# report parameters
if my_rank == root_rank:
    print("my_rank " + str(my_rank) + "/" + str(world_size) + " my_device " + str(my_device) + "/" + str(torch.cuda.device_count()) + "\n")
    print("seq length: " + str(seq_length))
    print("hidden dim: " + str(hidden_dim))
    print("num layers: " + str(num_layers))
    print("num heads: " + str(num_heads))

# initialize input and model
input = torch.randn(seq_length, hidden_dim, device=my_device)
Q = torch.ones(hidden_dim, hidden_dim // num_heads, device=my_device)
K = torch.ones_like(Q)
V  = torch.ones_like(Q)

layer = torch.matmul(Q, K.transpose(0, 1))

if my_rank == root_rank:
    print(f"Input shape: {input.shape} size {input.element_size() * input.nelement() / 1e9:.2f} GB")
    print(f"Q shape: {Q.shape} size {Q.element_size() * Q.nelement() / 1e9:.2f} GB")
    print(f"K shape: {K.shape} size {K.element_size() * K.nelement() / 1e9:.2f} GB")
    print(f"V shape: {V.shape} size {V.element_size() * V.nelement() / 1e9:.2f} GB")
    print(f"Layer shape: {layer.shape} size {layer.element_size() * layer.nelement() / 1e9:.2f} GB")

# compute Q, K, V
q = torch.matmul(input, Q)
k = torch.matmul(input, K)
v = torch.matmul(input, V)

if my_rank == root_rank:
    print(f"q shape: {q.shape} size {q.element_size() * q.nelement() / 1e9:.2f} GB")
    print(f"k shape: {k.shape} size {k.element_size() * k.nelement() / 1e9:.2f} GB")
    print(f"v shape: {v.shape} size {v.element_size() * v.nelement() / 1e9:.2f} GB")
    print(f"flops: {3 * (2 * seq_length * hidden_dim * hidden_dim // num_heads)/1e9:.2f} GFLOPs")

# compute attention
A = torch.matmul(q, k.transpose(0, 1))
if my_rank == root_rank:
    print(f"attention shape: {A.shape} size {A.element_size() * A.nelement() / 1e9:.2f} GB")
    print(f"Torch memory allocation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"flops: {2 * (seq_length * seq_length * hidden_dim // num_heads)/1e9:.2f} GFLOPs")

# calculate softmax
# A = torch.nn.functional.softmax(A, dim=-1)
# in-place softmax
torch.exp(A, out=A)
summed = torch.sum(A, dim=1, keepdim=True)
A /= summed
# compute scores
c = torch.matmul(A, v)
if my_rank == root_rank:
    print(f"scores shape: {c.shape} size {c.element_size() * c.nelement() / 1e9:.2f} GB")
    print(f"Torch memory allocation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")



