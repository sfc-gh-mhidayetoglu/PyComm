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
seq_length = 100000
hidden_dim = 16384 * 4
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

if my_rank == root_rank:
    print("input shape: " + str(input.shape))
    print("Q shape: " + str(Q.shape))
    print("K shape: " + str(K.shape))
    print("V shape: " + str(V.shape))
    print("Number of bytes in input: " + str(input.element_size() * input.nelement()))
    print("Number of bytes in Q: " + str(Q.element_size() * Q.nelement()))
    print("Number of bytes in K: " + str(K.element_size() * K.nelement()))
    print("Number of bytes in V: " + str(V.element_size() * V.nelement()))

# compute Q, K, V
q = torch.matmul(input, Q)
k = torch.matmul(input, K)
v = torch.matmul(input, V)

if my_rank == root_rank:
    print("Number of bytes in q: " + str(q.element_size() * q.nelement()))
    print("Number of bytes in k: " + str(k.element_size() * k.nelement()))
    print("Number of bytes in v: " + str(v.element_size() * v.nelement()))
    print("q shape: " + str(q.shape))
    print("k shape: " + str(k.shape))
    print("v shape: " + str(v.shape))

# compute attention
attention = torch.matmul(q, k.transpose(0, 1))
if my_rank == root_rank:
    print("Number of bytes in attention: " + str(attention.element_size() * attention.nelement()))
    print("attention shape: " + str(attention.shape))





