#!/bin/bash
export CUDA_PATH=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda-13.3/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda-13.3/targets/x86_64-linux/lib:$LIBRARY_PATH
export CPATH=/usr/local/cuda/targets/x86_64-linux/include:$CPATH

echo "Building with CUDA libraries..."
cargo build --release
