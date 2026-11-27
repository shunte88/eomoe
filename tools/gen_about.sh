#!/bin/sh
# Regenerate src/about.txt (`eomoe about`) after any dependency change: sh tools/gen_about.sh
set -e
cd "$(dirname "$0")/.."
{
cat <<'HDR'
eomoe - everyman's own MoE. MIT licensed, (c) 2026 Stue Hunter. https://github.com/shunte88/eomoe

Third-party components:

ggml / llama.cpp | MIT, (c) 2023-2026 The ggml authors
  src/quant.rs ports ggml-quants.c dequantization; the forward pass, tokenizer and chat
  template handling follow llama.cpp's behaviour. https://github.com/ggml-org/llama.cpp
Qwen3-Coder-30B-A3B-Instruct | Apache-2.0, Alibaba Cloud (Qwen team)
  Model weights are not bundled: `eomoe install` downloads the unsloth GGUF quantization.
NVIDIA CUDA | NVIDIA CUDA EULA
  libcuda and NVRTC are loaded from the host at run time, not bundled. The Docker image is
  built on nvidia/cuda images (NVIDIA Deep Learning Container License).

Rust crates linked into the binary (name version | license):
HDR
cargo tree -e normal --prefix none -f '{p} | {l}' | sed 's/ (\*)$//; s/ (proc-macro)//' | grep -v '^eomoe ' | sort -u
} > src/about.txt
