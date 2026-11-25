# eomoe status

A phase is **done** only when its gate passes against the reference oracle and the evidence is recorded here.
Oracle: llama.cpp `e6443bd` (`~/.local/share/local-code-llm`), Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.

Architecture decisions:
- CUDA C kernels now, all-Rust kernels (rust-cuda) as the end goal. Every kernel is a plain `extern "C" __global__` with only pointer and plain-value params, loaded from PTX by name, so the kernel source can be swapped without touching the engine.
- First release targets qwen3moe (F32 / Q4_K / Q6_K only), with the architecture code kept separate.

| Phase | Gate | Status |
|---|---|---|
| 0. Baseline | Golden outputs + perf captured from the oracle | **done** 2026-11-25 |
| 1. GGUF loader, tokenizer, chat template | Byte-identical to `tests/golden/{tokenize,template}.json` | **done** 2026-11-25 |
| 2. CPU reference forward pass | `greedy.json` top-1 on every step; mean logprob error ≤ llama.cpp's own CPU-vs-CUDA floor | **done** 2026-11-25 |
| 3. CUDA engine | Matches phase 2 / `greedy.json` | done 2026-11-25 (see below) |
| 4. Expert cache | Hit rate on `ref_trace.mct` ≥ oracle; decode tok/s ≥ oracle | done 2026-11-25 (see below) |
| 5. Server + one-stop CLI | Reference `check` passes; real Claude Code + Codex tool sessions | done 2026-11-25 (see below) |
| 6. Beyond parity | Prefetch, CPU-miss compute, batching, multi-GPU | not started |

## Phase 0 baseline (2026-11-25)

Hardware: RTX 5060 8 GB (sm_120, CUDA 13.2 driver / 13.3 toolkit), Ryzen 9 5900XT, 30 GB RAM (~13 GB held by other apps during the run).

Server flags: `--ctx-size 16384 --parallel 1 --gpu-layers 999 --cpu-moe --moe-cache-mib 4800 --jinja`.
4800 MiB is the largest cache that fits: 5120 fails to allocate the 234 MiB compute buffer. VRAM in use: 7526 / 8151 MiB.

Cache layout: 2 groups × 24 layers (down-projection Q6_K vs Q4_K), 879 slots each = 28.6% of experts (4.6 tokens' worth).

| Metric | Value |
|---|---|
| Decode, short prompt (256 tokens) | **27.2 tok/s** (36.8 ms/token) |
| Prefill, 4096-token prompt | **392.9 tok/s** |
| Decode right after the 4K prompt | 16.0 tok/s |
| Cache hit rate, ubatch ≤ 8 (decode) | **76.17%** (117973 hits / 36917 misses) |
| Cache hit rate, ubatch > 8 (prefill) | 19.45% |
| Uploaded over PCIe, decode | 101853 MiB across the run, ≈ 250 MB/token |

Reading: at about 250 MB of misses per token and about 13 GB/s over PCIe, roughly half of each 37 ms decode step is synchronous expert upload. Overlapping uploads with compute (prefetch) is the main lever.

Golden files (`tests/golden/`, re-capture with `python3 -I tools/capture_golden.py`; add `--no-gen` to refresh only tokenize/template from a `--gpu-layers 0` oracle):
`tokenize.json` (15 cases), `template.json` (4, incl. tool definitions and tool calls), `greedy.json` (4 prompts × 32 steps, top-5 logprobs), `perf.json`, `ref_trace.mct` (llama.cpp MCT1 expert-selection trace of the whole run).

## Phase 1 gate (2026-11-25)

Command: `cargo test --release --test phase1_golden -- --include-ignored` (needs the model; path via `EOMOE_TEST_MODEL`).

| Test | Checks | Result |
|---|---|---|
| `gguf_reads_metadata_and_tensors` | arch, block count, 579 tensors, types/dims, F32 values vs an independent Python reader | pass |
| `tokenize_matches_oracle` | 17 oracle cases (15 `parse_special=true`, 2 `false`): exact ids + decode round trip | 17/17 |
| `decode_matches_oracle_greedy_content` | detokenized greedy tokens == oracle `content`; prompt encoding == oracle `prompt_tokens` (4 prompts) | 4/4 |
| `chat_template_matches_oracle` | 5 `/apply-template` cases, incl. nested tool schema (enum, int/float/bool defaults, type list, `<`, `'`), multi tool calls with non-string args, consecutive tool results | 5/5 byte-identical |

Model open (mmap + metadata + tensor index): ~0.07 s.
Known gaps, untested because no oracle case exists: Python float exponent formatting in `tojson` (`1e-09`); llama.cpp's forced-CONTROL override list (every affected Qwen token is already CONTROL).

## Phase 2 gate (2026-11-25)

Command: `cargo test --release --test phase2_golden -- --include-ignored --nocapture`

| Test | Checks | Result |
|---|---|---|
| `dequant_matches_gguf_py` | Rows dequantized by gguf-py (`tools/capture_dequant.py`): Q4_K (token_embd, expert gate) and Q6_K (output, expert down), rel 1e-6 | pass |
| `forward_matches_oracle_greedy` | Teacher-forced over 4 prompts × 32 steps (128 steps): top-1 == oracle on every non-tie step; mean \|Δlogprob\| over oracle top-5 ≤ 0.2414 | top-1 128/128, mean 0.174 |

**Revised gate (user decision 2026-11-25).** The planned gate allowed only 0.05 logprob error per token. That is unreachable, even for llama.cpp itself:
- llama.cpp quantizes activations inside its quantized matmuls (Q8_K on CPU, q8_1 on CUDA). This model amplifies the rounding: residual-stream sums reach ~950 by layer 2.
- `tools/measure_backend_floor.py` scores llama.cpp's own CPU backend against its CUDA `greedy.json` with the same metric: 1 top-1 mismatch, 434/640 logprobs off by more than 0.05, mean 0.241, max 2.28.
- eomoe CPU in f32 scores 0 top-1 mismatches, 399/640 logprobs off by more than 0.05, mean 0.174, max 2.75. The top-5 set differs on 17 steps, always at the 5th-place boundary.

Further evidence the forward pass is correct:
- **Layer by layer:** a `llama-eval-callback` dump for prompt "The" matches ours in layers 0–2, with identical expert selection.
- **Exact match under emulation:** when our matmuls emulate Q8_K activation quantization, layer 0 `Vcur` matches llama.cpp to 6 digits (−0.190937 vs −0.190936). The emulation code was removed afterwards; f32 lands closer to the CUDA oracle.
- **The gate goes red on a real bug:** dropping the top-8 weight renormalization gives 21/128 top-1 failures and mean 5.06.

Speed (CPU reference, rayon over 32 logical cores, whole sequence in one batch): 15–20 tok/s.

## Phase 3 gate (2026-11-25)

Command: `cargo test --release --test phase3_golden -- --include-ignored --nocapture --test-threads 1` (RTX 5060 8 GB, sm_120; kernels in `kernels/eomoe.cu`, built to a SASS fatbin by `build.rs` and loaded by name in `src/gpu.rs`).

| Test | Result |
|---|---|
| `gpu_matches_cpu_reference`: \|Δlogprob\| over CPU top-5, all teacher-forced steps, tol 0.02 | max 0.00008 |
| `gpu_forward_matches_oracle_greedy`: the Phase 2 floor gate | top-1 fails 0/128, mean 0.1738 (floor 0.2414), max 2.7544, top-5 set differs on 17 steps. Identical to the CPU reference |
| `gpu_incremental_decode_matches_oracle_tokens`: prefill, then 1-token decode through the KV cache | all 4 cases reproduce the `greedy.json` tokens |

VRAM allocated: 1412 MiB (dense weights, f32 KV cache for 256 ctx, 128 expert slots, scratch).

Speed, with no expert cache yet (Phase 4):

| Mode | Decode | Prefill |
|---|---|---|
| Real run: every routed expert re-uploaded from pageable memory, synchronously, every layer (~1.2 GB/token) | 5.8 tok/s | 9–15 tok/s |
| Kernels only, uploads skipped (debug run, output invalid) | 76.5 tok/s | 143 tok/s |

The oracle decodes at 27.2 tok/s, so the kernels are not the bottleneck: expert transfer is. That is Phase 4's job (VRAM expert cache, pinned/async staging).

Profile history (nsys): the first kernels dequantised one element per thread with one block per (row, token). mm_q4_k plus mm_q6_k took 95% of GPU time, which capped kernels-only decode at 27 tok/s. Rewriting them as warp-per-row with vectorised sub-block decode gave the 76.5 tok/s above.

Known ceilings (marked `ponytail:` in the code): matvec re-reads the weights for every token, so prefill is not tiled; attention runs one position per step.

## Phase 4 gate (2026-11-25)

Command: `cargo test --release --test phase4_golden -- --include-ignored --nocapture --test-threads 1`

| Test | Oracle | eomoe |
|---|---|---|
| `cache_hit_rate_on_ref_trace_beats_oracle`: replay `ref_trace.mct` at the oracle's 879 slots per group, ubatch ≤ 8 | 117973 hits / 36917 misses (76.17%) | **117973 / 36917, identical**: `src/expert_cache.rs` is an exact port of llama.cpp's ClassLRU |
| `gpu_decode_beats_oracle_tok_s`: the 4 greedy cases (warm-up, tokens must still equal `greedy.json`), then prompt 1 + 256 greedy tokens, 4800 MiB cache | 27.2 tok/s | **33.7–34.1 tok/s** over 3 runs (+25%), 73.3% hits over the whole run |

Phase 3 gates re-run on the final code: GPU vs CPU 0.00009, floor gate 0/128 top-1 fails with mean 0.1738, incremental decode reproduces `greedy.json`.

Engine (`src/gpu.rs`):
- **Expert cache.** Layers are grouped by expert layout (Q6_K vs Q4_K down), as in the oracle. Each group gets a bank of slots (gate | up | down) and an `ExpertCache`. The budget is split so each group caches the same fraction of its experts (4800 MiB gives 880 slots per group).
- **Large batches.** Ubatches over 32 tokens bypass the policy, as in the oracle. They read the experts already cached and stream misses through 32 staging slots in waves.
- **Pinned experts.** At load, all experts (16.3 GiB) are copied into page-locked RAM in slot layout. A miss is then one async DMA at 14 GB/s (pageable runs at 10 GB/s and blocks the host). If pinning fails, it falls back to pageable copies. Pinning takes ~8.6 s at load.
- **Grouped expert kernels.** `mm_q4_k_ids` / `mm_q6_k_ids` take one weight pointer per gathered row. Each layer's MoE is gather, then [gate|up] (one matvec over the contiguous slot), then swiglu, down, and combine: 5 launches instead of ~28. The routing goes up as one async copy from pinned memory.
- **Attention.** It is now warp-parallel over positions (flash-decoding style). Before this change it took 32% of kernel time.

Measured steps on this decode benchmark (tok/s): pageable sync uploads 16.9, pinned async 27.5, grouped kernels 29.3, warp attention 34.1.

Rejected: a CRF half-life of 32–64 gives +0.24 pp on `ref_trace`, but that would be tuning on the gate's own trace. The oracle's 128 is kept.

Remaining time per token is ~29 ms: ~20 ms of miss uploads (~280 MB/token at 14 GB/s, PCIe 4.0 x8), serial with ~9 ms of kernels. Overlapping them (next-layer expert prefetch) is Phase 6.

## Phase 5 gate (2026-11-25)

`eomoe serve` (`src/server.rs`) speaks llama-server's `/v1/messages`, `/v1/responses`, count_tokens / input_tokens, `/v1/models` and `/health`. Request conversion, SSE event sequences, IDs and errors were ported from llama.cpp `e6443bd` source. The CLI (`src/cli.rs`) has the reference launcher's commands: install, serve, check, codex, claude.

| Gate | Command | Result |
|---|---|---|
| Reference `check` | `PORT=8090 ~/.local/share/local-code-llm/bin/local-code-llm check` against `eomoe serve --port 8090` | **pass** (both endpoints stream and complete); `eomoe check` passes too |
| Oracle API diff | `cargo test --release --test phase5_golden -- --ignored` (replays `tests/golden/api.json`, captured by `tools/capture_api_golden.py`) | **pass**: all 6 cases (messages and responses × text / tool / tool_result, full and stream). Prompt tokens are exact: 24 / 404 / 463 / 32 / 308 / 364. Tool calls are identical (`Bash {"command":"ls -la"}`, `shell {"command":["ls","-l","/tmp/demo"]}`), as are the stop reasons and streamed event sequences. count_tokens 404 = 404; health and models match |
| Real Claude Code 2.1.332 | `eomoe claude --port 8090 -p "calc.py has a bug: add() should return the sum. Fix it by editing the file." --tools Read,Edit,Bash` | **pass**: 3 turns (Read, then Edit, then answer), and `calc.py` now has `return a + b`. 68 s wall |
| Real Codex 0.172.0 | `eomoe codex --port 8090 exec --skip-git-repo-check --sandbox workspace-write "mul.py has a bug: ..."` | **pass**: 5 turns, and `mul.py` now has `return a * b`. 124 s wall |

Server log for the Codex session: the first turn prefilled 6811 tokens in 86 s (79 tok/s). Later turns reused the KV prefix (6848 / 7054 / 7147 / 7241 cached), so each recomputed only 20–130 tokens. Decode ran at 13.6–14.4 tok/s at a ~7K context.

Code review (standards and spec, two parallel reviewers) led to these fixes, all re-gated above:
- A client disconnect now stops generation even while a tool call or stop string is held back. Checked live: `Ok(Cancelled)` after 3 tokens.
- A 30 s `:` SSE keep-alive during long prefills.
- Errors after the stream starts are reported (Anthropic `event: error`, Responses `data: {"error":…}`).
- A UTF-8 split character after an invalid byte is no longer lost.
- The sampling seed changes per nanosecond, not per second.
- A malformed JSON body gets the `{"error":…}` shape.
- Added `/v1/responses/input_tokens`.
- Unit tests for the stop matcher and both request converters.

Engine changes made for serving:
- `Engine::truncate` / `forward_last` support KV prefix reuse (`src/generate.rs`).
- The expert cache defaults to all free VRAM after the KV cache: 2937 MiB at 16K ctx, 539 slots per group.
- The engine lives on one OS thread (the CUDA context is not bound per call).

Known limits, carried to Phase 6:
- **Context.** A stock Claude Code prompt is ~19K tokens, more than the 16K ctx (the reference has the same limit and the same error), so the gate session limited its tools. An f16 KV cache would give 32K ctx in the same VRAM.
- **Prefill** runs at 80–110 tok/s against the oracle's 393 (the matvec is not tiled). It dominates first-turn latency.
- **Decode** at 4–7K context is 14–17 tok/s, against 34 tok/s at a short context. Attention cost and cache misses at a smaller cache.
- `/v1/chat/completions` is not served; neither client uses it.
