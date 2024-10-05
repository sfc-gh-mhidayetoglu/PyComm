import torch
import torch.distributed as dist

dist.init_process_group(backend='nccl')

my_rank = dist.get_rank()
world_size = dist.get_world_size()
torch.cuda.set_device(my_rank % torch.cuda.device_count())
my_device = torch.cuda.current_device()

print("my_rank " + str(my_rank) + "/" + str(world_size) + " my_device " + str(my_device) + "/" + str(torch.cuda.device_count()) + "\n")

A = torch.empty(3, 4)
B = torch.empty(3, 4, device=my_device)
C = A.to(my_device)
print(A)
print(B)
print(C)
