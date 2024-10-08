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

def matmul_1D_colwise(hidden_dim = 16384, batch_size = 1024, num_layers = 118, TP = 8, DP = 2):

    # allocate memory
    A = torch.randn(hidden_dim, hidden_dim//TP, dtype=torch.bfloat16, device=my_device) # root layer
    list_A = [torch.randn_like(A) for _ in range(num_layers)] # l x (n, n/TP)
    B = torch.randn(hidden_dim//TP, batch_size//DP, dtype=torch.bfloat16, device=my_device)     # (n/TP, b/ DP)
    C = torch.empty(hidden_dim//TP, batch_size//DP, dtype=torch.bfloat16, device=my_device)     # (n/TP, b/DP)
    C_part = torch.empty(hidden_dim, batch_size//DP, dtype=torch.bfloat16, device=my_device) # (n, b/DP)
    # TP sharding of C_part
    list_C_part = [C_part.narrow(0, i, hidden_dim//TP) for i in range(0, hidden_dim, hidden_dim//TP)] # TP x (n/TP, b/DP)

    # report memory usage
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

    time_comm = []
    time_matmul = []
    time_total = []

    # iterate over layers
    for layer in range(num_layers):

        # A is different for each layer
        A = list_A[layer]

        # CRITICAL PART STARTS ***************************************************
        # Synchronize
        torch.cuda.synchronize()
        dist.barrier()
        time_start = time.perf_counter()
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
        time_end = time.perf_counter()
        dist.barrier()
        # CRITICAL PART ENDS ***************************************************

        # double buffering
        C, B = B, C

        # record time
        time_comm.append(event_comm_start.elapsed_time(event_comm_end))
        time_matmul.append(event_matmul_start.elapsed_time(event_matmul_end))
        time_total.append(time_end - time_start)
   
    # report time
    for layer in range(num_layers):
        matmul = time_matmul[layer] # in microseconds
        comm = time_comm[layer] # in microseconds 
        total = time_total[layer] # in seconds
        max_ = torch.tensor(total, device=my_device) # in seconds
        dist.all_reduce(max_, op=dist.ReduceOp.MAX)
        max_ = max_.item()
        if my_rank == root_rank:
            print("layer %d" % (layer))
            print("matmul %.2f comm %.2f matmul+comm = %.2f overhead %.2f us" % (matmul*1e3, comm*1e3, (matmul+comm)*1e3, total*1e6-(matmul+comm)*1e3))
            print("total %.2f max %.2f us " % (total * 1e6, max_ * 1e6))


# print("my_rank " + str(my_rank) + "/" + str(world_size) + " my_device " + str(my_device) + "/" + str(torch.cuda.device_count()) + "\n")

# model parameters
hidden_dim = 16384
batch_size = 1024
num_layers = 118

# parallelization parameters
TP = 8
DP = 2

# report parameters
if my_rank == root_rank:
    print("my_rank " + str(my_rank) + "/" + str(world_size) + " my_device " + str(my_device) + "/" + str(torch.cuda.device_count()) + "\n")
    print("hidden dim: " + str(hidden_dim) + "\n")
    print("batch size: " + str(batch_size) + "\n")
    print("num layers: " + str(num_layers) + "\n")

    print("TP: " + str(TP) + "\n")
    print("DP: " + str(DP) + "\n")
    if TP * DP != dist.get_world_size():
        print("TP * DP != world_size\n")
        exit()

# measure row-wise partitioning
matmul_1D_colwise(hidden_dim, batch_size, num_layers, TP, DP)
