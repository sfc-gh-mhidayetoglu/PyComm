import torch
import torch.distributed as dist
import time
import math
import numpy as np

# initialize
dist.init_process_group(backend='nccl')
my_rank = dist.get_rank()
world_size = dist.get_world_size()
torch.cuda.set_device(my_rank % torch.cuda.device_count())
my_device = torch.cuda.current_device()
root_rank = 7

# model parameters
hidden_dim = 16384
batch_size = 1024
num_layers = 126
mini_batch = 1

# parallelization parameters
TP = 64
DP = 1

# report parameters
if my_rank == root_rank:
    print("my_rank " + str(my_rank) + "/" + str(world_size) + " my_device " + str(my_device) + "/" + str(torch.cuda.device_count()) + "\n")
    print("hidden dim: " + str(hidden_dim))
    print("batch size: " + str(batch_size))
    print("num layers: " + str(num_layers))
    print("mini_batch: " + str(mini_batch))

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
event_comm2_start = torch.cuda.Event(enable_timing=True)
event_comm2_end = torch.cuda.Event(enable_timing=True)

def matmul_colwise(hidden_dim = 16384, batch_size = 1024, num_layers = 118, TP = 8, DP = 2, mini_batch = None):
    # allocate memory
    A = torch.randn(hidden_dim, hidden_dim//TP, dtype=torch.bfloat16, device=my_device) # root layer (n, n/TP)
    list_A = [torch.ones_like(A) / hidden_dim for _ in range(num_layers)] # l x (n, n/TP)
    B = torch.ones(hidden_dim//TP, batch_size//DP, dtype=torch.bfloat16, device=my_device) # (n/TP, b/DP)
    C = torch.empty(hidden_dim//TP, batch_size//DP, dtype=torch.bfloat16, device=my_device) # (n/TP, b/DP)
    C_buff = torch.empty(hidden_dim, batch_size//DP, dtype=torch.bfloat16, device=my_device) # (n, b/DP)
    # report memory usage
    if my_rank == root_rank:
        print("A " + str(A.size()) + " size " + str(A.element_size() * A.nelement() / 1e6) + " MB")
        print("list_A " + str(len(list_A)) + " size " + str(sum([A.element_size() * A.nelement() for A in list_A]) / 1e6) + " MB")
        print("B " + str(B.size()) + " size " + str(B.element_size() * B.nelement() / 1e6) + " MB")
        print("C " + str(C.size()) + " size " + str(C.element_size() * C.nelement() / 1e6) + " MB")
        print("C_buff " + str(C_buff.size()) + " size " + str(C_buff.element_size() * C_buff.nelement() / 1e6) + " MB")
        print("Torch memory allocation: " + str(torch.cuda.memory_allocated() / 1e6) + " MB")
    if mini_batch is not None:
        # synchronize
        torch.cuda.synchronize()
        dist.barrier()
        time_perf = time.perf_counter()
        event_start.record()
        # iterate over layers
        for layer in range(num_layers):
            torch.matmul(list_A[layer], B, out=C_buff)
            dist.reduce_scatter_tensor(C, C_buff, group=group_TP)
            C, B = B, C
        # synchronize
        event_end.record()
        torch.cuda.synchronize()
        dist.barrier()
        time_perf = time.perf_counter() - time_perf
        time_event = event_start.elapsed_time(event_end)
        # report time
        if my_rank == root_rank:
            print("column-wise total %.2f event %.2f ms" % (time_perf*1e3, time_event))
            print("column-wise per-iteration perf %.2f event %.2f us\n" % (time_perf/num_layers*1e6, time_event/num_layers*1e3))
    else:
        time_comm = []
        time_matmul = []
        time_total = []
        # iterate over layers
        for layer in range(num_layers):
            # Synchronize
            torch.cuda.synchronize()
            dist.barrier()
            time_start = time.perf_counter()
            event_matmul_start.record()
            # partial multiplication
            torch.matmul(list_A[layer], B, out=C_buff)
            # record events
            event_matmul_end.record()
            event_comm_start.record()
            # Reduce partial results into total results in each TP group
            dist.reduce_scatter_tensor(C, C_buff, group=group_TP)
            # Synchronize
            event_comm_end.record()
            torch.cuda.synchronize()
            time_end = time.perf_counter()
            dist.barrier()
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
                print("column-wise layer %d" % (layer), end=" ")
                FLOPs = 2 * A.size(0) * A.size(1) * B.size(1)
                Bytes = C_buff.element_size() * C_buff.nelement()
                print("matmul %.2f (%.2f TFLOPS) comm %.2f (%.2f GB/s) matmul+comm = %.2f overhead %.2f us" % (matmul*1e3, FLOPs / (matmul / 1e3) / 1e12, comm*1e3, Bytes / (comm / 1e3) / 1e9, (matmul+comm)*1e3, total*1e6-(matmul+comm)*1e3), end=" ")   
                print("total %.2f max %.2f us" % (total * 1e6, max_ * 1e6))
    return B

def matmul_rowwise(hidden_dim = 16384, batch_size = 1024, num_layers = 118, TP = 8, DP = 2, mini_batch = None):
    # allocate memory
    A = torch.randn(hidden_dim//TP, hidden_dim, dtype=torch.bfloat16, device=my_device) # root layer (n/TP, n)
    list_A = [torch.ones_like(A) / hidden_dim for _ in range(num_layers)] # l x (n/TP, n)
    B = torch.ones(hidden_dim//TP, batch_size//DP, dtype=torch.bfloat16, device=my_device) # (n/TP, b/DP)
    C = torch.empty(hidden_dim//TP, batch_size//DP, dtype=torch.bfloat16, device=my_device) # (n/TP, b/DP)
    B_buff = torch.empty(hidden_dim, batch_size//DP, dtype=torch.bfloat16, device=my_device) # (n, b/DP)
    # report memory usage
    if my_rank == root_rank:
        print("A " + str(A.size()) + " size " + str(A.element_size() * A.nelement() / 1e6) + " MB")
        print("list_A " + str(len(list_A)) + " size " + str(sum([A.element_size() * A.nelement() for A in list_A]) / 1e6) + " MB")
        print("B " + str(B.size()) + " size " + str(B.element_size() * B.nelement() / 1e6) + " MB")
        print("C " + str(C.size()) + " size " + str(C.element_size() * C.nelement() / 1e6) + " MB")
        print("B_buff " + str(B_buff.size()) + " size " + str(B_buff.element_size() * B_buff.nelement() / 1e6) + " MB")
        print("Torch memory allocation: " + str(torch.cuda.memory_allocated() / 1e6) + " MB")
    if mini_batch is not None:
        # synchronize
        torch.cuda.synchronize()
        dist.barrier()
        time_perf = time.perf_counter()
        event_start.record()
        # iterate over layers
        for layer in range(num_layers):
            dist.all_gather_into_tensor(B_buff, B, group=group_TP)
            torch.matmul(list_A[layer], B_buff, out=C)
            C, B = B, C
        # synchronize
        event_end.record()
        torch.cuda.synchronize()
        dist.barrier()
        time_perf = time.perf_counter() - time_perf
        time_event = event_start.elapsed_time(event_end)
        # report time
        if my_rank == root_rank:
            print("row_wise total %.2f event %.2f ms" % (time_perf*1e3, time_event))
            print("row_wise per-iter perf %.2f event %.2f us\n" % (time_perf/num_layers*1e6, time_event/num_layers*1e3))
    else:
        time_comm = []
        time_matmul = []
        time_total = []
        # iterate over layers
        for layer in range(num_layers):
            # Synchronize
            torch.cuda.synchronize()
            dist.barrier()
            time_start = time.perf_counter()
            event_comm_start.record()
            # gather B
            dist.all_gather_into_tensor(B_buff, B, group=group_TP)
            # record events
            event_comm_end.record()
            event_matmul_start.record()
            # partial multiplication
            torch.matmul(list_A[layer], B_buff, out=C)
            # Synchronize
            event_matmul_end.record()
            torch.cuda.synchronize()
            time_end = time.perf_counter()
            dist.barrier()
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
                print("row-wise layer %d" % (layer), end=" ")
                FLOPs = 2 * A.size(0) * A.size(1) * B_buff.size(1)
                Bytes = B_buff.element_size() * B_buff.nelement()

                print("comm %.2f (%.2f GB/s) matmul %.2f (%.2f TFLOPS) comm+matmul = %.2f overhead %.2f us" % (comm*1e3, Bytes / (comm / 1e3) / 1e9, matmul*1e3, FLOPs / (matmul / 1e3) / 1e12, (comm+matmul)*1e3, total*1e6-(comm+matmul)*1e3), end=" ")
                print("total %.2f max %.2f us" % (total * 1e6, max_ * 1e6))
    return B


def hilbert_curve_index(n, x, y):
    rx, ry, s, d = 0, 0, 1, 0
    while s < n:
        rx = (x & s) > 0
        ry = (y & s) > 0
        d += s * s * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1:
                x = n - 1 - x
                y = n - 1 - y
            x, y = y, x
        s <<= 1
    return d

def matmul_2D(hidden_dim = 16384, batch_size = 1024, num_layers = 126, TP=8, DP = 2, mini_batch = None):
    # allocate memory
    TP_sqrt = math.isqrt(TP)
    A = torch.randn(hidden_dim//TP_sqrt, hidden_dim//TP_sqrt, dtype=torch.bfloat16, device=my_device) # root layer (n/sqrt(TP), n/sqrt(TP))
    list_A = [torch.ones_like(A) / hidden_dim for _ in range(num_layers)] # l x (n/sqrt(TP), n/sqrt(TP))
    B = torch.ones(hidden_dim//TP, batch_size//DP, dtype=torch.bfloat16, device=my_device) # (n/TP, b/DP)
    C = torch.empty(hidden_dim//TP, batch_size//DP, dtype=torch.bfloat16, device=my_device) # (n/TP, b/DP)
    B_buff = torch.empty(hidden_dim//TP_sqrt, batch_size//DP, dtype=torch.bfloat16, device=my_device) # (n/sqrt(TP), b/DP)
    C_buff = torch.empty(hidden_dim//TP_sqrt, batch_size//DP, dtype=torch.bfloat16, device=my_device) # (n/sqrt(TP), b/DP)

    # report memory usage
    if my_rank == root_rank:
        print("A " + str(A.size()) + " size " + str(A.element_size() * A.nelement() / 1e6) + " MB")
        print("list_A " + str(len(list_A)) + " size " + str(sum([A.element_size() * A.nelement() for A in list_A]) / 1e6) + " MB")
        print("B " + str(B.size()) + " size " + str(B.element_size() * B.nelement() / 1e6) + " MB")
        print("C " + str(C.size()) + " size " + str(C.element_size() * C.nelement() / 1e6) + " MB")
        print("B_buff " + str(B_buff.size()) + " size " + str(B_buff.element_size() * B_buff.nelement() / 1e6) + " MB")
        print("C_buff " + str(C_buff.size()) + " size " + str(C_buff.element_size() * C_buff.nelement() / 1e6) + " MB")
        print("Torch memory allocation: " + str(torch.cuda.memory_allocated() / 1e6) + " MB")

    # map_2D = [[None for _ in range(TP_sqrt)] for _ in range(TP_sqrt)]
    # map_2D = [[0, 1, 12, 5], [11, 2, 7, 6], [4, 13, 8, 9], [15, 14, 3, 10]] # arbitrary order
    # map_2D = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]] # row-wise order
    # map_2D = [[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]] # column-wise order
    # Morton (Z-order) curve mapping
    def morton_index(x, y):
        answer = 0
        for i in range(max(x.bit_length(), y.bit_length())):
            answer |= ((x >> i) & 1) << (2 * i + 1) | ((y >> i) & 1) << (2 * i)
        return answer
    map_2D = [[None for _ in range(TP_sqrt)] for _ in range(TP_sqrt)]
    for i in range(TP_sqrt):
        for j in range(TP_sqrt):
            map_2D[i][j] = morton_index(i, j)
    # Map local_rank to a 2D domain
    rank_2D = [None] * TP
    for i in range(TP_sqrt):
        for j in range(TP_sqrt):
            rank_2D[map_2D[i][j]] = (i, j)

    if my_rank == root_rank:
        print("myid " + str(my_rank) + "\nrank_2D " + str(rank_2D) + "\nmap_2D " + str(map_2D))

    sendlist = [map_2D[rank % TP_sqrt][rank // TP_sqrt] for rank in range(TP)]
    recvlist = [rank_2D[rank][1] * TP_sqrt + rank_2D[rank][0] for rank in range(TP)]
    if my_rank == root_rank:
        print("myid " + str(my_rank) + " sendlist " + str(sendlist) + " recvlist " + str(recvlist))

    matrix_1 = [["." for _ in range(TP)] for _ in range(TP)]
    matrix_2 = [["." for _ in range(TP)] for _ in range(TP)]
    for i in range(TP):
        matrix_1[sendlist[i]][i] = 1
        matrix_2[i][recvlist[i]] = 1
    if matrix_1 != matrix_2:
        if my_rank == root_rank:
            print("matrix_1 != matrix_2")
        return
    if my_rank == root_rank:
        print("matrix_1")
        for row in matrix_1:
            print(" ".join(map(str, row)))
    commlist = list()
    for sender in range(TP):
        for recver in range(TP):
            if matrix_1[recver][sender] == 1:
                commlist.append((sender, recver))
    if my_rank == root_rank:
        for comm in commlist:
            print(str(comm[0]) + " -> " + str(comm[1]))


    row_group = [map_2D[rank_2D[local_rank][0]][col] for col in range(TP_sqrt)]
    col_group = [map_2D[row][rank_2D[local_rank][1]] for row in range(TP_sqrt)]
    group_TP_row = dist.new_group(row_group, use_local_synchronization=True)
    group_TP_col = dist.new_group(col_group, use_local_synchronization=True)

    if my_rank == root_rank:
        print("myid " + str(my_rank) + " row_group " + str(row_group) + " col_group " + str(col_group))

    ''' p2p_list = list()
    for sender, recver in commlist:
        if sender == recver:
            continue
        if local_rank == sender:
            p2p_list.append(dist.P2POp(dist.isend, sendbuf, recver, group=group_TP))
        if local_rank == recver:
            p2p_list.append(dist.P2POp(dist.irecv, recvbuf, sender, group=group_TP))

    torch.cuda.synchronize()
    dist.barrier()
    reqs = dist.batch_isend_irecv(p2p_list)
    for req in reqs:
        req.wait()
    torch.cuda.synchronize()
    dist.barrier()
    return'''

    # P2P communication buffers
    B_temp = torch.empty_like(B)
    C_temp = torch.empty_like(C)
    # record P2P communications within TP group
    is_self = False
    my_comm_list = []
    for sender, recver in commlist:
        if sender == recver:
            if local_rank == sender:
                is_self = True
        else:
            if local_rank == sender:
                my_comm_list.append((dist.send, B, recver, group_TP))
            if local_rank == recver:
                my_comm_list.append((dist.recv, B_temp, sender, group_TP))

    torch.cuda.synchronize()
    for layer in range(num_layers):
        dist.barrier()
        time_start = time.perf_counter()
        '''for sender, recver in commlist:
            if sender == recver:
                if local_rank == sender:
                    B_temp = B.clone()
            else:
                if local_rank == sender:
                    dist.send(B, recver, group=group_TP)
                if local_rank == recver:
                    dist.recv(B_temp, sender, group=group_TP)'''
        # replay P2P communications
        if is_self:
            B_temp = B.clone()
        else:
            for comm in my_comm_list:
                comm[0](comm[1], comm[2], group=comm[3])
        # torch.cuda.synchronize()
        # dist.barrier()
        dist.all_gather_into_tensor(B_buff,B_temp, group=group_TP_col)
        torch.matmul(list_A[layer], B_buff, out=C_buff)
        dist.reduce_scatter_tensor(C_temp, C_buff, group=group_TP_row)
        torch.cuda.synchronize()
        dist.barrier()
        time_end = time.perf_counter()
        if my_rank == root_rank:
            print("total %.2f us" % ((time_end - time_start) * 1e6))

    if mini_batch is not None:
        # synchronize
        torch.cuda.synchronize()
        dist.barrier()
        time_perf = time.perf_counter()
        event_start.record()
        # iterate over layers
        for layer in range(num_layers):
            # send_handle = [dist.Work] * TP_sqrt
            # for i in range(TP_sqrt):
            #     handle_list[i] = dist.irecv(B_buff[i*TP_sqrt:(i+1)*TP_sqrt], src=B_col_panel[i], group=group_TP)
            # dist.all_gather_into_tensor(B_buff, B, group=group_TP)
            dist.all_gather_into_tensor(B_buff, B, group=group_TP_col)
            torch.matmul(list_A[layer], B_buff, out=C_buff)
            dist.reduce_scatter_tensor(C, C_buff, group=group_TP_row)
            C, B = B, C
        # synchronize
        event_end.record()
        torch.cuda.synchronize()
        dist.barrier()
        time_perf = time.perf_counter() - time_perf
        time_event = event_start.elapsed_time(event_end)
        # report time
        if my_rank == root_rank:
            print("2D total %.2f event %.2f ms" % (time_perf*1e3, time_event))
            print("2D per-iter perf %.2f event %.2f us\n" % (time_perf/num_layers*1e6, time_event/num_layers*1e3))
    else:
        time_comm = []
        time_matmul = []
        time_comm2 = []
        time_total = []
        # iterate over layers
        for layer in range(num_layers):
            # Synchronize
            torch.cuda.synchronize()
            dist.barrier()
            time_start = time.perf_counter()
            # gather B_buff
            event_comm_start.record()
            dist.all_gather_into_tensor(B_buff, B, group=group_TP_col)
            event_comm_end.record()
            # partial multiplication
            event_matmul_start.record()
            torch.matmul(list_A[layer], B_buff, out=C_buff)
            event_matmul_end.record()
            # scatter C_buff
            event_comm2_start.record()
            dist.reduce_scatter_tensor(C, C_buff, group=group_TP_row)
            event_comm2_end.record()
            # Synchronize
            torch.cuda.synchronize()
            time_end = time.perf_counter()
            dist.barrier()
            # double buffering
            C, B = B, C
            # record time
            time_comm.append(event_comm_start.elapsed_time(event_comm_end))
            time_matmul.append(event_matmul_start.elapsed_time(event_matmul_end))
            time_comm2.append(event_comm2_start.elapsed_time(event_comm2_end))
            time_total.append(time_end - time_start)
        # report time
        for layer in range(num_layers):
            comm = time_comm[layer] # in microseconds 
            matmul = time_matmul[layer] # in microseconds
            comm2 = time_comm2[layer] # in microseconds
            total = time_total[layer] # in seconds
            max_ = torch.tensor(total, device=my_device) # in seconds
            dist.all_reduce(max_, op=dist.ReduceOp.MAX)
            max_ = max_.item()
            if my_rank == root_rank:
                print("2D layer %d" % (layer), end=" ")
                FLOPs = 2 * A.size(0) * A.size(1) * B_buff.size(1)
                Bytes_B = B_buff.element_size() * B_buff.nelement()
                Bytes_C = C_buff.element_size() * C_buff.nelement()
                print("comm %.2f (%.2f GB/s) matmul %.2f (%.2f TFLOPS) comm2 %.2f (%.2f GB/s) comm+matmul+comm2 = %.2f overhead %.2f us" % (comm*1e3, Bytes_B / (comm / 1e3) / 1e9, matmul*1e3, FLOPs / (matmul / 1e3) / 1e12, comm2*1e3, Bytes_C / (comm2 / 1e3) / 1e9, (comm+matmul+comm2)*1e3, total*1e6-(comm+matmul+comm2)*1e3), end=" ")
                print("total %.2f max %.2f us" % (total * 1e6, max_ * 1e6))
    return C

B_2D = matmul_2D(hidden_dim, batch_size, num_layers, TP, DP)
exit()

# measure row-wise partitioning
B_colwise = matmul_colwise(hidden_dim, batch_size, num_layers, TP, DP)
B_colwise = matmul_colwise(hidden_dim, batch_size, num_layers, TP, DP, mini_batch)
B_rowwise = matmul_rowwise(hidden_dim, batch_size, num_layers, TP, DP)
B_rowwise = matmul_rowwise(hidden_dim, batch_size, num_layers, TP, DP, mini_batch)
B_2D = matmul_2D(hidden_dim, batch_size, num_layers, TP, DP)
B_2D = matmul_2D(hidden_dim, batch_size, num_layers, TP, DP, mini_batch)

if B_colwise.eq(torch.ones_like(B_colwise)).all():
    if my_rank == root_rank:
        print("B_colwise correct")
else:
    if my_rank == root_rank:
        print("B_colwise incorrect")
if B_rowwise.eq(torch.ones_like(B_rowwise)).all():
    if my_rank == root_rank:
        print("B_rowwise correct")
else:
    if my_rank == root_rank:
        print("B_rowwise incorrect")
if B_2D.eq(torch.ones_like(B_2D)).all():
    if my_rank == root_rank:
        print("B_2D correct")
else:
    if my_rank == root_rank:
        print("B_2D incorrect")
