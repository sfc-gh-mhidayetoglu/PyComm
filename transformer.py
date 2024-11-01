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
seq_length = 15000  # N
hidden_dim = 16384  # d
num_heads = 128     # h
inter_size = 53248  # d'
num_layers = 63 # 126    # L
vocab_size = 128256  # V
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
    print("inter size: " + str(inter_size))
    print("num heads: " + str(num_heads))
    print("num layers: " + str(num_layers))
    print("vocab size: " + str(vocab_size))
    print("type: " + str(type))
    print("TP: " + str(TP))
    print("DP: " + str(DP))
    print("P: " + str(P))

# initialize group communicator
ranks_TP = [i for i in range(world_size) if i // TP == my_rank // TP]
ranks_DP = [i for i in range(world_size) if i // DP == my_rank // DP]
if my_rank == root_rank:
    print("TP ranks: " + str(ranks_TP))
    print("DP ranks: " + str(ranks_DP))
group_TP = dist.new_group(ranks_TP, use_local_synchronization=True)
group_DP = dist.new_group(ranks_DP, use_local_synchronization=True)

# initialize model
QKV = torch.ones(num_layers, num_heads//TP, 3, hidden_dim, hidden_dim//num_heads, device=my_device, dtype=type)
O = torch.ones(num_layers, hidden_dim//TP, hidden_dim, device=my_device, dtype=type)
W1 = torch.ones(num_layers, hidden_dim, inter_size//TP, device=my_device, dtype=type)
W2 = torch.ones(num_layers, inter_size//TP, hidden_dim, device=my_device, dtype=type)
embedding = torch.randn(seq_length//DP, hidden_dim, device=my_device, dtype=type)
attention = torch.empty(num_heads//TP//DP, seq_length, seq_length, device=my_device, dtype=type)
activation = torch.empty(seq_length//DP, inter_size//TP, device=my_device, dtype=type)

if my_rank == root_rank:
    print(f"QKV [L, h/TP, 3, d, d/h]: {QKV.shape}, elements: {QKV.nelement()}, size: {QKV.element_size() * QKV.nelement() / 1e9:.2f} GB")
    print(f"O [L, d/TP, d]: {O.shape}, elements: {O.nelement()}, size: {O.element_size() * O.nelement() / 1e9:.2f} GB")
    print(f"W1 [L, d, d'/TP]: {W1.shape}, elements: {W1.nelement()}, size: {W1.element_size() * W1.nelement() / 1e9:.2f} GB")
    print(f"W2 [L, d'/TP, d]: {W2.shape}, elements: {W2.nelement()}, size: {W2.element_size() * W2.nelement() / 1e9:.2f} GB")
    print(f"embedding [N/DP, d]: {embedding.shape}, elements: {embedding.nelement()}, size: {embedding.element_size() * embedding.nelement() / 1e6:.2f} MB")
    print(f"attention [h/TP/DP, N, N]: {attention.shape}, elements: {attention.nelement()}, size: {attention.element_size() * attention.nelement() / 1e9:.2f} GB")
    print(f"activation [N/DP, d'/TP]: {activation.shape}, elements: {activation.nelement()}, size: {activation.element_size() * activation.nelement() / 1e6:.2f} MB")
    torch.cuda.synchronize()
    print(f"Current memory allocation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Peak memory allocation: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

def attention_2D(input, QKV, O, attention, group_TP, group_DP) -> torch.Tensor:
    # compute q, k, v
    qkv = torch.matmul(input, QKV)
    # all-to-all within DP
    qkv_ = torch.empty(DP, num_heads//TP//DP, 3, seq_length//DP, hidden_dim//num_heads, device=my_device, dtype=type)
    dist.all_to_all_single(qkv_, qkv, group=group_DP)
    qkv_ = torch.transpose(qkv_, 0, 2).reshape((3, num_heads//TP//DP, seq_length, hidden_dim//num_heads))
    # compute attention
    attention = torch.matmul(qkv_[0], qkv_[1].transpose(-2, -1))
    # compute scores
    attention = torch.nn.functional.softmax(attention, dim=-1)
    c_ = torch.matmul(attention, qkv_[2])
    # all-to-all within DP
    c_ = torch.transpose(c_, 0, 1).contiguous()
    c = torch.empty(DP, seq_length//DP, num_heads//TP//DP, hidden_dim//num_heads, device=my_device, dtype=type)
    dist.all_to_all_single(c, c_, group=group_DP)
    c = c.transpose(0, 1).reshape((seq_length//DP, hidden_dim//TP))
    # compute output
    output = torch.matmul(c, O)
    # all-reduce within TP
    dist.all_reduce(output, group=group_TP)
    return output

def MLP_2D(input, W1, W2, activation, group_TP) -> torch.Tensor:
    # input [N/DP, d]
    # W1[L, d, d'/TP]
    # W2[L, d'/TP, d]
    # activation [N/DP, d'/TP]
    activation = torch.matmul(input, W1)
    activation = torch.nn.functional.gelu(activation)
    output = torch.matmul(activation, W2)
    dist.all_reduce(output, group=group_TP)
    return output


torch.cuda.synchronize()
dist.barrier()

for i in range(num_layers):
    embedding_ = attention_2D(embedding, QKV[i], O[i], attention, group_TP, group_DP)
    embedding = MLP_2D(embedding_, W1[i], W2[i], activation, group_TP)

torch.cuda.synchronize()
dist.barrier()
if my_rank == root_rank:
    print(f"Current memory allocation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Peak memory allocation: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
