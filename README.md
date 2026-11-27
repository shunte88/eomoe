<p align="center">
  <img src="assets/eomoe.webp" alt="eomoe man, the everyman MoE superhero" width="320"/>
</p>

# eomoe - everyman's own MoE: a 30B coding model on an 8 GB GPU, faster than llama.cpp

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
| `POST /v1/completions` | OpenAI text completions: raw prompt (string or token ids), streaming |
| `POST /completion`, `/infill` | llama.cpp native completion and fill-in-the-middle (`input_prefix`, `input_suffix`, `input_extra`), for llama.vim / llama.vscode |
| `POST /api/chat`, `/api/generate`; `GET /api/tags`, `/api/ps`, `/api/version`; `POST /api/show` | Ollama API, for tools that only speak Ollama (`OLLAMA_HOST=127.0.0.1:8080 ollama run local-code`) |
| `GET /v1/models`, `/health` | |
| `GET /` | Minimal web chat page |

`/chat/completions`, `/responses` and `/models` are also served without the `/v1` prefix, and `/v1/responses/input_tokens` counts tokens for Codex.

Structured output: `response_format` with `json_object` or `json_schema` (also Responses `text.format`) constrains sampling to valid JSON. The supported schema subset is `type`, `properties`, `required`, `additionalProperties`, `items`, `enum` and `const`. `logit_bias` (`{"<id>": bias}` or `[[id, bias]]`, `false` bans) is honoured too.

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
| `--host` | | 127.0.0.1 |
| `--port` | `PORT` | 8080 |
| `--ctx-size` | `CTX_SIZE` | 32768 |
| `--cache-mib` | `MOE_CACHE_MIB` | all free VRAM |
| `--parallel` | `PARALLEL` | 4 (requests served at once; they share the KV pool, 1..=16) |
| `--alias` | | `local-code` |
| `--api-key` | `LLAMA_API_KEY` | none (comma-separated keys; requests then need `Authorization: Bearer KEY` or `x-api-key: KEY`) |
| `--api-key-file` | `LLAMA_ARG_API_KEY_FILE` | none (one key per line, `#` comments) |
| `--ssl-key-file`, `--ssl-cert-file` | `LLAMA_ARG_SSL_KEY_FILE`, `LLAMA_ARG_SSL_CERT_FILE` | none (both set: HTTPS) |

The YAML file also takes `logging.level` (`RUST_LOG` overrides it).

Deployment: `docker compose up` reads `./config/config.yaml` and `./models` (it needs nvidia-container-toolkit). `eomoe.service` is a systemd unit.

## Remote access

The server binds 127.0.0.1 by default. Ways to reach it from another machine:

- **SSH tunnel** (nothing to configure): on the laptop run `ssh -N -L 8080:127.0.0.1:8080 gpu-box`, then use eomoe as if it were local.
- **Tailscale:** run `eomoe serve --host <tailscale IP> --api-key KEY`. For HTTPS, `tailscale cert gpu-box.<tailnet>.ts.net` writes a key and a certificate; pass them as `--ssl-key-file` and `--ssl-cert-file`.
- **LAN:** run `--host 0.0.0.0 --api-key KEY` and preferably add TLS. Keys travel in headers, so use HTTPS outside a trusted network.

Auth matches llama-server: `/health`, `/v1/health` and the chat page at `/` stay open, and anything else without a valid key gets llama-server's 401 body.

On the client machine, `check`, `claude` and `codex` take `--url` and `--api-key` (`EOMOE_URL`, `EOMOE_API_KEY`):

```bash
eomoe claude --url https://gpu-box.example.ts.net:8080 --api-key KEY
```

For a self-signed certificate, trust it with `SSL_CERT_FILE=cert.pem` (`check`, `codex`) or `NODE_EXTRA_CA_CERTS=cert.pem` (`claude`).

Open `/` in a browser for a minimal chat page. It streams `/v1/chat/completions`; enter the key there if one is set.

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

## License and support

eomoe's own source is [MIT licensed](LICENSE). Distributed binaries may also include third-party components, see eomoe --about dialog for component information.

MIT already covers commercial use, so building eomoe into a product needs no permission from me. If you want more than the license gives you - support, an integration, or an OEM arrangement - [open an issue](https://github.com/shunte88/eomoe/issues) and say what you need.


If eomoe helps power your projects, a link back is appreciated:

> Built on [eomoe - everyman's own MoE](https://github.com/shunte88/eomoe), © 2026 Stue Hunter. MIT licensed.

Bug reports, hardware test results, and useful pull requests are welcome. You can also [buy 
me a coffee](https://www.buymeacoffee.com/shunte88) or browse [Team Badger merchandise](https://www.zazzle.com/team_badger_t_shirt-235604841593837420).

