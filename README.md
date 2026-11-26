# eomoe

A single-binary Rust server that runs Qwen3-Coder-30B-A3B (Q4_K_M GGUF, a 30B mixture-of-experts model) on one 8 GB GPU. It speaks the same HTTP APIs as llama-server, so Claude Code, Codex and OpenAI clients work unchanged.

Experts live in RAM. The ones in use are cached in VRAM, and misses are streamed over PCIe while compute runs. eomoe replaces the reference llama.cpp setup (`local-code-llm`) and is faster on every measured path (RTX 5060 8 GB; numbers and gates in [STATUS.md](STATUS.md)):

| | llama.cpp oracle | eomoe |
|---|---|---|
| Prefill, 4K prompt | 393 tok/s | **953 tok/s** |
| Decode, short prompt | 27.2 tok/s | **38.1 tok/s** |
| Decode after a long prompt | 16.0 tok/s (4K) | **25.5 tok/s** (19K) |
| Context | 16K | **32K** (f16 KV) |
| Claude Code fix-a-bug session (19K-token prompt) | did not fit in 16K | **44 s** |

## Quick start

```bash
cargo build --release
./target/release/eomoe install          # model + Claude Code / Codex clients
./target/release/eomoe serve            # leave running (127.0.0.1:8080)
./target/release/eomoe claude           # Claude Code on the local server
./target/release/eomoe codex            # Codex CLI on the local server
./target/release/eomoe check            # streamed smoke test of both APIs
```

`install` reuses an existing reference install (`~/.local/share/local-code-llm`) instead of downloading again. eomoe's own home is `$EOMOE_HOME`, by default `~/.local/share/eomoe`.

## HTTP API

The API is wire-compatible with llama-server (`e6443bd`). The gates in `tests/phase5_golden.rs` diff it against captured oracle traffic (`tests/golden/api.json`).

| Endpoint | API |
|---|---|
| `POST /v1/chat/completions` | OpenAI chat: tools, streaming, `stream_options.include_usage`, `timings` |
| `POST /v1/responses` | OpenAI Responses (Codex) |
| `POST /v1/messages`, `/v1/messages/count_tokens` | Anthropic Messages (Claude Code) |
| `GET /v1/models`, `/health` | |

```python
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")
print(c.chat.completions.create(model="local-code", messages=[{"role": "user", "content": "hi"}]).choices[0].message.content)
```

The prompt prefix is reused across requests (KV cache), so agent follow-up turns only prefill the new tokens.

## Configuration

Settings come from flags, environment variables, or a YAML file (`--config`, or `EOMOE_CONFIG_PATH`). Flags override environment variables, which override the file. See `config.example.yaml` and `eomoe serve --help`.

| Flag | Env | Default |
|---|---|---|
| `--model` | `MODEL_FILE` | `<home>/models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf` |
| `--port` | `PORT` | 8080 |
| `--ctx-size` | `CTX_SIZE` | 32768 |
| `--cache-mib` | `MOE_CACHE_MIB` | all free VRAM |
| `--alias` | | `local-code` |

## How it works

| Module | Role |
|---|---|
| `gguf.rs`, `tokenizer.rs`, `template.rs` | GGUF loader, BPE tokenizer, Jinja chat template. Byte-identical to llama.cpp |
| `qwen3moe.rs`, `quant.rs`, `cpu.rs` | Weight map and routing; the f32 CPU reference forward pass (the correctness oracle) |
| `gpu.rs`, `kernels/eomoe.cu` | CUDA engine. Kernels are plain `extern "C"` functions loaded by name, so an all-Rust kernel module can replace the `.cu` file |
| `expert_cache.rs` | VRAM expert cache: frequency-class LRU and next-layer prefetch |
| `generate.rs`, `toolcall.rs`, `server.rs`, `cli.rs` | Sampling, tool-call parsing, the HTTP API, the CLI |

Main techniques:
- **Expert cache:** experts are cached per layer group in VRAM. Uploads come from pinned RAM on a copy stream that overlaps compute.
- **Decode prefetch:** during decode, each layer runs the next layer's router on its own output to predict that layer's experts (84% recall) and prefetches them.
- **Prefill matrix multiplies:** tensor-core GEMM working directly on the Q4_K/Q6_K integers (MMQ-style), with exact weights and ~f32 activations.
- **Attention:** tensor-core FlashAttention-2 for prefill; split-K attention for long-context decode.

## Development

```bash
cargo test --release --lib
cargo test --release --test phase6_golden -- --ignored --nocapture --test-threads=1   # GPU gates; one model at a time
```

A phase is done only when its gate passes against the oracle and the evidence is in `STATUS.md`. The golden files come from the reference llama.cpp build via `tools/capture_*.py`. Requirements: Rust, the CUDA toolkit (`nvcc`; set `NVCC` to override its path), and an sm_120 GPU with 8 GB of VRAM. For other GPUs, set `EOMOE_CUDA_ARCH` (e.g. `89`).
