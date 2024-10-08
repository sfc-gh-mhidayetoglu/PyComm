import torch
import torch.distributed as dist
import time

# initialize
dist.init_process_group(backend='nccl')
my_rank = dist.get_rank()
world_size = dist.get_world_size()
torch.cuda.set_device(my_rank % torch.cuda.device_count())
my_device = torch.cuda.current_device()
root_rank = 8

# print("my_rank " + str(my_rank) + "/" + str(world_size) + " my_device " + str(my_device) + "/" + str(torch.cuda.device_count()) + "\n")

# model parameters
hidden_dim = 16384
batch_size = 1024
input_size = 5120
num_layers = 118

# parallelization parameters
TP = 8
DP = 2

# report parameters
if my_rank == root_rank:
    print("my_rank " + str(my_rank) + "/" + str(world_size) + " my_device " + str(my_device) + "/" + str(torch.cuda.device_count()) + "\n")
    print("hidden dim: " + str(hidden_dim) + "\n")
    print("batch size: " + str(batch_size) + "\n")
    print("input size: " + str(input_size) + "\n")
    print("num layers: " + str(num_layers) + "\n")

    print("TP: " + str(TP) + "\n")
    print("DP: " + str(DP) + "\n")
    if TP * DP != world_size:
        print("TP * DP != world_size\n")
        exit()

# allocate memory
A = torch.randn(hidden_dim, hidden_dim//TP, dtype=torch.bfloat16, device=my_device) # root layer
list_A = [torch.randn_like(A) for _ in range(num_layers)] # l x (n, n/TP)
B = torch.randn(hidden_dim//TP, batch_size//DP, dtype=torch.bfloat16, device=my_device)     # (n/TP, b/ DP)
C = torch.empty(hidden_dim//TP, batch_size//DP, dtype=torch.bfloat16, device=my_device)     # (n/TP, b/DP)
C_part = torch.empty(hidden_dim, batch_size//DP, dtype=torch.bfloat16, device=my_device) # (n, b/DP)
# TP sharding of C_part
list_C_part = [C_part.narrow(0, i, hidden_dim//TP) for i in range(0, hidden_dim, hidden_dim//TP)] # TP x (n/TP, b/DP)

if my_rank == root_rank:
    print("A " + str(A.size()) + " size " + str(A.element_size() * A.nelement() / 1e6) + " MB\n")
    print("list_A " + str(len(list_A)) + " size " + str(sum([A.element_size() * A.nelement() for A in list_A]) / 1e6) + " MB\n")
    print("B " + str(B.size()) + " size " + str(B.element_size() * B.nelement() / 1e6) + " MB\n")
    print("C " + str(C.size()) + " size " + str(C.element_size() * C.nelement() / 1e6) + " MB\n")
    print("C_part " + str(C_part.size()) + " size " + str(C_part.element_size() * C_part.nelement() / 1e6) + " MB\n")
    print("list_C_part " + str(len(list_C_part)) + " size " + str(sum([C_part.element_size() * C_part.nelement() for C_part in list_C_part]) / 1e6) + " MB\n")

# Create group communicators
ranks = [i for i in range(world_size) if i // TP == my_rank // TP]
# print("myid: " + str(my_rank) + " ranks " + str(ranks) + "\n")
group_TP = dist.new_group(ranks, use_local_synchronization=True)
local_rank = my_rank % TP
# Create cuda events
event_matmul_start = torch.cuda.Event(enable_timing=True)
event_matmul_end = torch.cuda.Event(enable_timing=True)
event_comm_start = torch.cuda.Event(enable_timing=True)
event_comm_end = torch.cuda.Event(enable_timing=True)

for layer in range(num_layers):

    # A is different for each layer
    A = list_A[layer]

    # CRITICAL PART STARTS ***************************************************
    # Synchronize
    torch.cuda.synchronize()
    dist.barrier()
    time_total = time.perf_counter()
    event_matmul_start.record()

    # partial multiplication
    C_part = torch.matmul(A, B)

    # record events
    event_matmul_end.record()
    event_comm_start.record()

    # Reduce partial results into total results in each TP group
    dist.reduce_scatter(C, list_C_part, group=group_TP)

    # Synchronize
    event_comm_end.record()
    torch.cuda.synchronize()
    time_total = time.perf_counter() - time_total # in seconds
    time_comm = event_comm_start.elapsed_time(event_comm_end) # in microseconds
    time_matmul = event_matmul_start.elapsed_time(event_matmul_end) # in microseconds
    dist.barrier()
    # CRITICAL PART ENDS ***************************************************

    # double buffering
    C, B = B, C

    # find max time
    time_max = torch.tensor(time_total, device=my_device)
    dist.all_reduce(time_max, op=dist.ReduceOp.MAX)
    time_max = time_max.item()
    if my_rank == root_rank:
        print("layer %d" % (layer))
        print("matmul %.2f comm %.2f matmul+comm = %.2f overhead %.2f us" % (time_matmul*1e3, time_comm*1e3, (time_matmul+time_comm)*1e3, time_total*1e6-(time_matmul+time_comm)*1e3))
        print("total %.2f max %.2f us " % (time_total * 1e6, time_max * 1e6))

exit()  
if my_rank == root_rank:
    print("initialize input\n")
    _input = torch.randn((hidden_dim, input_size), dtype=torch.bfloat16, device=my_device)
    print("input " + str(_input.size()) + " size " + str(_input.element_size() * _input.nelement() / 1e6) + " MB\n")

