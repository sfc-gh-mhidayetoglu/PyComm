import torch
import torch.distributed as dist
import time
import numpy as np

# initialize
dist.init_process_group(backend='nccl')
my_rank = dist.get_rank()
world_size = dist.get_world_size()
torch.cuda.set_device(my_rank % torch.cuda.device_count())
my_device = torch.cuda.current_device()
root_rank = 7

def find_max(time):
    time_max = torch.tensor([time], device=my_device)
    torch.distributed.all_reduce(time_max, op=torch.distributed.ReduceOp.MAX)
    return time_max.item()

# import number of elements
data = np.loadtxt('log_0.txt', usecols=4, dtype=int)
print(data)

count = 8388608
type = torch.bfloat16

buff = torch.empty(count, dtype=type, device=my_device)
bytes = buff.numel() * buff.element_size()
if my_rank == root_rank:
    print("my_rank " + str(my_rank) + "/" + str(world_size) + " my_device " + str(my_device) + "/" + str(torch.cuda.device_count()))
    print(data)
    print(f"Buffer size in MB: {bytes / 1e6}")

event_start = torch.cuda.Event(enable_timing=True)
event_end = torch.cuda.Event(enable_timing=True)
for i in range(0, 3000):
    buff_ = buff.narrow(0, 0, data[i])
    bytes = buff_.numel() * buff_.element_size()
    torch.cuda.synchronize()
    dist.barrier()
    time_start = time.perf_counter()
    event_start.record()
    dist.all_reduce(buff_)
    event_end.record()
    torch.cuda.synchronize()
    time_end = time.perf_counter()
    event_time = event_start.elapsed_time(event_end)
    perf_time = time_end - time_start
    event_max = find_max(event_time)
    time_max = find_max(perf_time)
    if my_rank == root_rank:
        print(f"iter {i} {data[i]} elements perf {perf_time*1e6:.2f} event {event_time*1e3:.2f} perf max {time_max*1e6:.2f} event max {event_max*1e3:.2f} us throughput: {bytes / (event_time * 1e3) / 1e9:.2f} GB/s")

 
