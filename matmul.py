import torch
import torch.distributed as dist

A = torch.empty(3, 4)

if torch.cuda.is_available():
    print('We have a GPU!')
else:
    print('Sorry, CPU only.')

print("Hello\n")
print(A)
