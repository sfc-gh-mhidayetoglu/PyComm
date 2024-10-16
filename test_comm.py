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

# import number of elements
data = np.loadtxt('log_0.txt', usecols=4, dtype=int)

count = 8388608
type = torch.bfloat16

buff = torch.empty(count, dtype=type, device=my_device)



if my_rank == root_rank:
    print("my_rank " + str(my_rank) + "/" + str(world_size) + " my_device " + str(my_device) + "/" + str(torch.cuda.device_count()))
    print(data)
    print(f"Buffer size in MB: {buff.element_size() * buff.numel() / 1e6}")
 
