# -*- coding: utf-8 -*-
"""
@author: Alessandro Diana

explanation: PyTorch counterpart of 'check.py'. Prints the installed PyTorch
version and verifies whether a CUDA-capable GPU is visible and usable.
Run this right after installing PyTorch to confirm the CUDA setup is correct.
"""

import torch


def main():
    print("PyTorch version:", torch.__version__)
    print("Built with CUDA:", torch.version.cuda)                  # None if CPU-only build
    print("cuDNN version:", torch.backends.cudnn.version())
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print("Num GPUs available:", n)
        for i in range(n):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"    Total VRAM: {props.total_memory / 1024**3:.2f} GB")
    else:
        print("No CUDA GPU detected: training will run on CPU (much slower).")


if __name__ == "__main__":
    main()
