import torch

 

print(f"Pytorch: {torch.__version__}")

if torch.cuda.is_available():    
    print(f"Device Count  : {torch.cuda.device_count()}")
    print(f"Device Current: {torch.cuda.current_device()}")

    print(f"Device 0: {torch.cuda.device(0)}")
    print(f"Device 0: {torch.cuda.get_device_name(0)}")

    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

else:

    print("No Device available")