import torch

# Check if CUDA is available
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    # Number of CUDA devices
    print(f"Number of CUDA Devices: {torch.cuda.device_count()}")
    
    # Current CUDA device
    current_device = torch.cuda.current_device()
    print(f"Current CUDA Device index: {current_device}")
    print(f"Current CUDA Device: {torch.cuda.get_device_name(current_device)}")
    
    # Loop through available devices for more detailed info
    for i in range(torch.cuda.device_count()):
        print(f"\nCUDA Device #{i}")
        print(f"Device Name: {torch.cuda.get_device_name(i)}")
        # Memory details
        total_memory = torch.cuda.get_device_properties(i).total_memory
        print(f"Total Memory: {total_memory / (1024 ** 3)} GB")
        # More properties can be accessed with get_device_properties
        props = torch.cuda.get_device_properties(i)
        print(f"Compute Capability: {props.major}.{props.minor}")
        print(f"MultiProcessor Count: {props.multi_processor_count}")
        
        # Current and max memory usage (may need torch.cuda.memory_allocated() and torch.cuda.max_memory_allocated() for dynamic queries)
        print(f"Current Memory Allocated: {torch.cuda.memory_allocated(i) / (1024 ** 3)} GB")
        print(f"Max Memory Allocated: {torch.cuda.max_memory_allocated(i) / (1024 ** 3)} GB")
        torch.cuda.reset_peak_memory_stats(i)  # Reset peak memory metrics

        # CUDA device capabilities
        print(f"CUDA Capability (major, minor): {torch.cuda.get_device_capability(i)}")
else:
    print("CUDA is not available. Ensure that you have a CUDA-compatible GPU and the correct drivers installed.")
