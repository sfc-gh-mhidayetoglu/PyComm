import torch
import torch.distributed as dist
import time
from enum import Enum

# initialize
dist.init_process_group(backend='nccl')
my_rank = dist.get_rank()
world_size = dist.get_world_size()
torch.cuda.set_device(my_rank % torch.cuda.device_count())
my_device = torch.cuda.current_device()
root_rank = 8

# model parameters
hidden_dim = 16384
batch_size = 1024
num_layers = 118
mini_batch = 1

# parallelization parameters
TP = 8
DP = 2

# masurement parameters
num_warmup = num_layers

# report parameters
if my_rank == root_rank:
    print("my_rank " + str(my_rank) + "/" + str(world_size) + " my_device " + str(my_device) + "/" + str(torch.cuda.device_count()) + "\n")
    print("hidden dim: " + str(hidden_dim))
    print("batch size: " + str(batch_size))
    print("num layers: " + str(num_layers))
    print("mini_batch: " + str(mini_batch))
    print("num_warmup: " + str(num_warmup))

    print("TP: " + str(TP))
    print("DP: " + str(DP))
    if TP * DP != dist.get_world_size():
        print("TP * DP != world_size\n")
        exit()

# Create group communicators
ranks = [i for i in range(world_size) if i // TP == my_rank // TP]
# print("myid: " + str(my_rank) + " ranks " + str(ranks) + "\n")
group_TP = dist.new_group(ranks, use_local_synchronization=True)
local_rank = my_rank % TP

# Create cuda events
event_start = torch.cuda.Event(enable_timing=True)
event_end = torch.cuda.Event(enable_timing=True)
event_matmul_start = torch.cuda.Event(enable_timing=True)
event_matmul_end = torch.cuda.Event(enable_timing=True)
event_comm_start = torch.cuda.Event(enable_timing=True)
event_comm_end = torch.cuda.Event(enable_timing=True)

def matmul_colwise(hidden_dim = 16384, batch_size = 1024, num_layers = 118, TP = 8, DP = 2, mini_batch = 1):
    # allocate memory
    A = torch.randn(hidden_dim, hidden_dim//TP, dtype=torch.bfloat16, device=my_device) # root layer (n, n/TP)
    list_A = [torch.randn_like(A) for _ in range(num_layers)] # l x (n, n/TP)
    B = torch.randn(hidden_dim//TP, batch_size//DP, dtype=torch.bfloat16, device=my_device) # (n/TP, b/DP)
    C = torch.empty(hidden_dim//TP, batch_size//DP, dtype=torch.bfloat16, device=my_device) # (n/TP, b/DP)
    C_buff = torch.empty(hidden_dim, batch_size//DP, dtype=torch.bfloat16, device=my_device) # (n, b/DP)
    # report memory usage
    if my_rank == root_rank:
        print("A " + str(A.size()) + " size " + str(A.element_size() * A.nelement() / 1e6) + " MB")
        print("list_A " + str(len(list_A)) + " size " + str(sum([A.element_size() * A.nelement() for A in list_A]) / 1e6) + " MB")
        print("B " + str(B.size()) + " size " + str(B.element_size() * B.nelement() / 1e6) + " MB")
        print("C " + str(C.size()) + " size " + str(C.element_size() * C.nelement() / 1e6) + " MB")
        print("C_buff " + str(C_buff.size()) + " size " + str(C_buff.element_size() * C_buff.nelement() / 1e6) + " MB")
    # warmup iterations
    for iter in range(num_warmup):
        # partial multiplication
        C_buff = torch.matmul(list_A[iter], B)
        # Reduce partial results into total results in each TP group
        dist.reduce_scatter_tensor(C, C_buff, group=group_TP)
        # double buffering
        C, B = B, C
    # synchronize
    torch.cuda.synchronize()
    dist.barrier()
    time_perf = time.perf_counter()
    event_start.record()
    # iterate over layers
    for layer in range(num_layers):
        C_buff = torch.matmul(list_A[layer], B)
        dist.reduce_scatter_tensor(C, C_buff, group=group_TP)
        torch.cuda.synchronize()
        C, B = B, C
    # synchronize
    event_end.record()
    dist.barrier()
    time_perf = time.perf_counter() - time_perf
    time_event = event_start.elapsed_time(event_end)
    # report time
    if my_rank == root_rank:
        print("column-wise total %.2f event %.2f ms" % (time_perf*1e3, time_event))
        print("column-wise per-iteration perf %.2f event %.2f us\n" % (time_perf/num_layers*1e6, time_event/num_layers*1e3))

    
def matmul_rowwise(hidden_dim = 16384, batch_size = 1024, num_layers = 118, TP = 8, DP = 2, mini_batch = 1):
    # allocate memory
    A = torch.randn(hidden_dim//TP, hidden_dim, dtype=torch.bfloat16, device=my_device) # root layer (n/TP, n)
    list_A = [torch.randn_like(A) for _ in range(num_layers)] # l x (n/TP, n)
    B = torch.randn(hidden_dim//TP, batch_size//DP, dtype=torch.bfloat16, device=my_device) # (n/TP, b/DP)
    C = torch.empty(hidden_dim//TP, batch_size//DP, dtype=torch.bfloat16, device=my_device) # (n/TP, b/DP)
    B_buff = torch.empty(hidden_dim, batch_size//DP, dtype=torch.bfloat16, device=my_device) # (n, b/DP)
    # report memory usage
    if my_rank == root_rank:
        print("A " + str(A.size()) + " size " + str(A.element_size() * A.nelement() / 1e6) + " MB")
        print("list_A " + str(len(list_A)) + " size " + str(sum([A.element_size() * A.nelement() for A in list_A]) / 1e6) + " MB")
        print("B " + str(B.size()) + " size " + str(B.element_size() * B.nelement() / 1e6) + " MB")
        print("C " + str(C.size()) + " size " + str(C.element_size() * C.nelement() / 1e6) + " MB")
        print("B_buff " + str(B_buff.size()) + " size " + str(B_buff.element_size() * B_buff.nelement() / 1e6) + " MB")
    # warmup iterations
    for iter in range(num_warmup):
        # gather B
        dist.all_gather_into_tensor(B_buff, B, group=group_TP)
        # partial multiplication
        C = torch.matmul(list_A[iter], B_buff)
        # double buffering
        C, B = B, C
    # synchronize
    torch.cuda.synchronize()
    dist.barrier()
    time_perf = time.perf_counter()
    event_start.record()
    # iterate over layers
    for layer in range(num_layers):
        dist.all_gather_into_tensor(B_buff, B, group=group_TP)
        C = torch.matmul(list_A[layer], B_buff)
        torch.cuda.synchronize()
        C, B = B, C
    # synchronize
    event_end.record()
    dist.barrier()
    time_perf = time.perf_counter() - time_perf
    time_event = event_start.elapsed_time(event_end)
    # report time
    if my_rank == root_rank:
        print("row_wise total %.2f event %.2f ms" % (time_perf*1e3, time_event))
        print("row_wise per-iter perf %.2f event %.2f us\n" % (time_perf/num_layers*1e6, time_event/num_layers*1e3))

# measure row-wise partitioning
matmul_colwise(hidden_dim, batch_size, num_layers, TP, DP)
matmul_rowwise(hidden_dim, batch_size, num_layers, TP, DP)
