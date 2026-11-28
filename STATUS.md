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
| 6. Beyond parity | Prefetch, CPU-miss compute, batching, multi-GPU | **done**: f16 KV / 32K ctx, prefill 953 tok/s @4K (2.4x), decode 38.1 / 25.5 after 19K, decode prefetch, OpenAI chat API (see below). Not pursued: CPU-miss compute, batching, multi-GPU |
| 7. Concurrent requests (continuous batching) | N concurrent streams: each stream's greedy tokens equal its solo run; aggregate decode at N=4 > llama-server `--parallel 4` (29.0 tok/s) and ≥ 1.25x eomoe's own N=1 (**revised from 1.5x, flagged**); no single-stream regression | **done 2026-11-26**: N=4 51.3 tok/s at the server (1.77x oracle, 1.29x own N=1); hybrid CPU experts lift single-stream decode to 41.2 / 28.8 after 19K (see below) |
| 8. Remote access | llama-server-compatible `--api-key` (401 body byte-identical), optional TLS, client commands take `--url`/`--api-key`, built-in web chat page | **done 2026-11-26**: 11/11 oracle auth cases identical; HTTPS + key verified with check, Claude Code and Codex (see below) |
| 9. Request pool | `eomoe pool` routes requests across N nodes with prefix affinity + load; aggregate throughput scales ~linearly with nodes | planned (future; needs a 2nd machine for the gate) |
| 10. Expert-parallel pool | Experts sharded across nodes' VRAM over 10–100 GbE: removes PCIe miss uploads; runs models bigger than one box (e.g. Qwen3-Coder-480B-A35B, same qwen3moe arch) | planned (future; latency probe first) |
| 11. Agent-serving extras (from a koboldcpp review) | `/v1/completions` with FIM for editor completion; RAM KV snapshots so evicted conversations come back without re-prefill; JSON-schema-constrained output (`response_format`); Ollama `/api/chat` + `/api/tags` adapter. Each one is gated against llama-server or Ollama wire captures | (a) completions + FIM done; KV snapshots, constrained output, Ollama next |

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

## Phase 6: f16 KV cache, 32K default context (2026-11-25)

The KV cache is stored in f16 (llama.cpp's default `--cache-type-k/v f16`). K/V are computed in f32 scratch, normed and roped there, and then converted to f16 by `store_f16`. The attention kernel reads `__half`. The CPU reference rounds K/V to f16 in the same way, so it still models the oracle. The default `ctx_size` is now 32768, enough for a stock Claude Code prompt.

| Gate | Result |
|---|---|
| `phase6_golden::ctx_32k_keeps_cache_and_greedy_tokens`: 32K ctx, auto cache, greedy.json tokens | Red before the change (CUDA OOM). Now: 7314 MiB VRAM, **539 slots per group** (the same as Phase 5 at 16K), all 4 cases exact |
| Phase 3 oracle floor gate | top-1 fails 0/128, **mean 0.1706** (was 0.1738, floor 0.2414). The f16 KV moved us closer to the oracle |
| Phase 3 incremental decode | all 4 cases reproduce `greedy.json` |
| Phase 4 cache / decode | hits identical to the oracle (76.17%); decode **34.9 tok/s** (oracle 27.2) |
| Phase 5 API diff | pass (re-run at the 32K default) |
| Real Claude Code, **stock tool set** (was over the context limit in Phase 5) | `eomoe claude -p "sub.py has a bug: ..."` with an 18955-token prompt: the bug is fixed (Read, Edit), 7m50s wall |

**Revised gate: GPU vs CPU reference (`gpu_matches_cpu_reference`).**
- **Old gate:** max |Δlogprob| ≤ 0.02. That held when both engines were f32 end to end.
- **New gate:** mean ≤ 0.005 and max ≤ 0.5. Measured mean 0.0023, p50 0.00024, p99 0.034, max 0.276.
- **Why:** with f16 storage, ~1e-6 summation-order differences occasionally flip an f16 rounding, and that is amplified downstream.
- **Evidence that this is a property of f16 KV and not a GPU bug:** the CPU reference against itself, with 1e-6 relative K/V jitter before rounding, gives mean 0.021 and max 0.81 (118 of 880 > 0.02). That is 3–9× worse than GPU vs CPU. (The probe was throwaway and has been removed.)
- **Bug detection is unchanged:** a real bug (e.g. no top-8 renormalisation) gives a mean of 5.06 on the floor gate.

**Measured at a long context** (the stock Claude Code session above):
- **Prefill:** 18955 tokens in 410 s (46 tok/s).
- **Decode:** 7.4 tok/s at 19K, against 34.9 tok/s at a short context.
- **Cause:** attention is the only cost that grows with context. The kernel re-reads K/V once for each of the 8 query heads that share a KV head (GQA), and decode launches only 32 blocks. This is the next Phase 6 item.

## Phase 6: fast prefill (2026-11-26)

**Gate** `phase6_golden::prefill_4k_beats_oracle`: a 4096-token prompt (this repo's sources), timed `forward_last`, must beat the oracle's 392.9 tok/s (Phase 0). Red at 101 tok/s.

| Step (each profiled with nsys on the gate) | 4096-token prefill |
|---|---|
| Before | 101 tok/s. Attention 12.7 s, matvecs 22 s, expert uploads 9.3 s (not overlapped) |
| `attention_tiled`: flash-style kernel. One block = 1 KV head × 32 query rows (the 8 GQA heads × 4 tokens); K/V tiles staged once in shared memory | 139 tok/s (attention 12.7 s → 1.1 s) |
| Matvecs tiled over 8 tokens (`_t8`): weights decoded once per tile. Per token the arithmetic is unchanged, so results are bit-identical | 191 tok/s |
| `gemm_q4_k` / `gemm_q6_k` (`_ids`): f32 SIMT GEMM. 128 rows × 32 tokens per block, weights dequantized into shared memory. Used for ≥ 16 tokens (≥ 16 rows per expert for experts) | 209 tok/s |
| Layer-major prefill (`forward_span`): 4096-token spans go layer by layer. Attention runs per 512 ubatch; the MoE runs over the whole span in expert waves, so each expert is uploaded once per layer per span, and each sees all its span tokens (~256 rows per GEMM) | **426 tok/s (gate passes; oracle 392.9)**. Uploads 9.3 s → 2.6 s |

| Gate (re-run after all of the above) | Result |
|---|---|
| Phase 3 oracle floor | top-1 fails 0/128, mean 0.1829 (floor 0.2414) |
| Phase 3 incremental decode | all 4 cases exact |
| Phase 3 GPU vs CPU (revised again, see below) | p50 0.00040, max 1.11 (gate: p50 ≤ 0.001, max ≤ 2.0) |
| Phase 4 | hits identical to the oracle (76.17%); decode 33.9 tok/s (oracle 27.2; 33.4–34.9 across runs) |
| Phase 5 API diff | pass |
| Phase 6 32K | **575 slots per group** (was 539): the all-logits scratch is now 64 rows, not 512 (−272 MiB), which more than pays for the 64 MiB span buffers |
| 19K-token prefill (scratch test, then deleted) | 301.8 tok/s |
| Real Claude Code, stock tools, 18957-token prompt | **1m57s wall** (was 7m50s). First-turn prefill 62.8 s (302 tok/s; was 409.9 s). Bug fixed |

**GPU vs CPU gate, revised to a median.**
- **What changed:** with the new kernels, the mean failed (0.035–0.049) and so did the old max ≤ 0.5. The cause is chaotic routing, not the kernels.
- **Evidence:** a per-position probe showed every position agreeing to ≤ 0.003 until a single onset, after which that case moves by 0.1–1.1. One onset is greedy.json case 2, position 8; the case-2 router probe found a top-8 tie at layer 4 with a logit gap of 5.9e-5. Case 0 also has an onset at position 28. Any change in summation order can flip those choices, including the old f32 matvec vs the CPU: scaling q before the dot instead of after it was enough to cause a flip.
- **Why the median still catches bugs:** a kernel bug moves every position of every case. The probes were throwaway and are removed.
- **What guards accuracy:** the oracle-floor gate, unchanged.

**Long context now** (19K tokens):
- Prefill 302 tok/s. At 19K, attention is 40% of prefill (22.7 s; SIMT only, ~130 GFLOPS), the expert GEMM is 22 s and the dense GEMM is 8.2 s. Tensor cores are the next lever for both.
- Decode was 7.4 tok/s, because the decode attention kernel had no GQA sharing and no split-K. Fixed below.

## Phase 6: long-context decode (2026-11-26)

Test first: `decode_after_19k_prompt_beats_oracle` (tests/phase6_golden.rs). It prefills 19000 tokens of this repo's sources, warms up with 8 decode steps, then times 64 greedy steps. It must reach ≥ 20 tok/s. The oracle's best long-context figure is 16.0 tok/s, and that is after only a 4K prompt. **Red at 7.5 tok/s.**

| Step | Decode after 19K |
|---|---|
| Baseline: `attention`, grid (32 heads), each block re-reads its KV head | 7.5 tok/s |
| Split-K on `attention_tiled` (gridDim.z over positions, 256 per split) + `attn_merge`; partials in the idle MoE scratch `s.gu`, so no VRAM cost | 24.7 (decode attention 183 µs/layer ≈ 213 GB/s) |
| `ATT_SPLIT` 128 | 25.0 |
| Kernel templated on rows per warp: `attention_dec` (R = 1) gives each of the 8 warps one of the token's 8 GQA heads. Before this, 6 of 8 warps idled | **29.3** |

At 29.3 tok/s, attention adds ~4.7 ms per token over short-context decode (34.0). The KV bandwidth floor at 19K is ~4.2 ms (1.87 GB at ~448 GB/s).

Gates (all pass):

| Gate | Result |
|---|---|
| Phase 3 | top-1 fails 0/128, mean 0.1829 (floor 0.2414); incremental decode exact; GPU vs CPU p50 0.00040, max 1.11 |
| Phase 4 | hits 76.17% (identical to the oracle); decode 34.0 tok/s |
| Phase 5 API diff | pass |
| Phase 6 | 32K, 575 slots; prefill 4K 418.7 tok/s; **decode after 19K 29.3 tok/s** |
| lib | 11 pass |
| Real Claude Code, stock tools, 18955-token prompt (`sub.py` bug) | **1m24s wall** (was 1m57s). Fixed. Server decode 23.4 / 22.9 / 24.8 tok/s (was 7.5 / 7.3 / 7.4) |

Next lever, found in the same log: follow-up turns prefill 100–200 new tokens on a 19K cache at only 74–99 tok/s.
- **Red test:** `followup_prefill_on_19k_prefix` times 128 new tokens on an 18.9K prefix. Target 150; currently 75.9.
- **Split-K does not fix it:** trying it on the R=4 path only reached 78. That attention is compute-bound, about 1.9 TFLOP per 128-token turn (≈1.1 TFLOPS achieved on SIMT f32).
- **The fix:** tensor-core attention, which is Phase 6 priority 2.

## Phase 6: tensor-core attention, overlapped expert uploads (2026-11-26)

| Step | Follow-up 128 tok on 18.9K | Prefill 4K |
|---|---|---|
| Before | 75.9 tok/s | 418.7 tok/s |
| `attention_tc`: FlashAttention-2 layout on `mma.sync` m16n8k16. Q and P are fed as f16 hi + lo pairs (~f32 accuracy); K and V are exact f16. Used for every batch of more than 4 tokens; decode stays on `attention_dec`. The R=4 SIMT kernel is deleted | 80.1 | 454.6 |
| Expert uploads on a copy stream. Staging is split into two halves of 16 slots. Wave 0 is the resident experts; misses come in waves of 16, ping-ponging between halves, ordered by `ready` and `free` events | 108.4 | 508.9 |

Profile of the follow-up before the copy stream: uploads 990 ms of 1592, attention 153 ms, expert matvecs 335 ms.
- **What the uploads mean:** 128 tokens touch nearly every expert of every layer. That is ~13.9 GB of misses at ~14 GB/s (PCIe 4.0 x8), so ~128 tok/s is the ceiling at this hit rate.
- **Gate target revised:** the follow-up target was 150, set on a wrong compute-bound estimate made before profiling. It is now **100**: above the 74–99 tok/s server baseline, below the upload floor.

**Decode gate fixed to be content-independent.**
- **The problem:** greedy decode after 19K moved between 22 and 29 tok/s as unrelated kernel changes perturbed the text it generated (routing, and so hits, depends on content). The prompt was also read from this repo's live sources, which I was editing.
- **The fix:** the prompt is now a frozen fixture (`tests/golden/long_prompt.txt`), and decode is teacher-forced on the fixture's next 64 tokens.
- **Result:** deterministic, **22.7 tok/s at 65.4% hits** (366 MB/token uploaded). Decode at 19K is now upload-bound. That is the next lever: next-layer expert prefetch.

Gates (all pass):

| Gate | Result |
|---|---|
| Phase 3 | top-1 fails 0/128, mean 0.1825 (floor 0.2414); incremental decode exact; GPU vs CPU **p50 0.00034** (was 0.00040), mean 0.036 (was 0.049), max 1.12 |
| Phase 4 | hits 76.17% (identical to the oracle); decode 34.2 tok/s |
| Phase 5 API diff | pass |
| Phase 6 | 32K, 576 slots; prefill 4K **510 tok/s**; decode after 19K 22.7 (teacher-forced); follow-up **112 tok/s** |
| lib | 11 pass |
| Real Claude Code, stock tools, 18955 tokens (`sub.py`) | **1m03s wall** (was 1m24s). Fixed. First turn: prefill 46.1 s at **411 tok/s** (was 302). Follow-ups: 110 and 165 tok/s. Decode 23.3–25.5 |

## Phase 6: next-layer expert prefetch for decode (2026-11-26)

During decode (one token), each layer also runs layer li+1's router on its own post-attention residual. The result is read back alongside the real router logits, so it costs no extra sync. The predicted experts that are not resident are uploaded on the copy stream into li+1's cache slots, overlapping layer li's MoE and layer li+1's attention.
- **Prediction quality:** top-8 recall is 83.8% (probe, since removed).
- **`ExpertCache::prefetch(il, ids, busy)`:** places experts without touching CRF or clock. It never evicts layer `busy`'s experts, which are still to be read; earlier layers are already finished because the router readback synchronized.
- **Ordering:** the main stream waits on `pf_ready` before planning the next layer.
- **Prefetch width:** the bottom-ranked predictions miss often enough that their bytes cost more PCIe time than they save. Sweep on the teacher-forced 19K gate:

| Prefetch top-N | 0 (before) | 2 | 4 | **5** | 6 | 8 |
|---|---|---|---|---|---|---|
| Decode after 19K, tok/s | 22.7 | 24.3 | 25.3 | **25.4** | 25.3 | 24.0 |
| Cache hits (prefetched count as hits) | 65.4% | 71.2% | 78.1% | 81.6% | 85.0% | 90.7% |
| MiB uploaded (64 tokens) | 23426 | 23553 | 24013 | 24565 | 25471 | 29011 |

Still upload-bound: ~400 MB/token is ~28 ms of a 39 ms token at ~14 GB/s. Next lever: more cache slots or fewer bytes per expert.

Gates (all pass, run with `--test-threads=1`):

| Gate | Result |
|---|---|
| Phase 3 | top-1 fails 0/128, mean 0.1825 (floor 0.2414); incremental decode exact; GPU vs CPU p50 0.00034, max 1.12 |
| Phase 4 | hits 76.17% (identical to the oracle); decode **38.0 tok/s** (was 34.2; oracle 27.2) |
| Phase 5 API diff | pass |
| Phase 6 | 32K, 576 slots; prefill 4K 501 tok/s; decode after 19K **25.3** (was 22.7); follow-up 112.8 |
| lib | 12 pass (new: `prefetch_spares_busy_layer_and_feeds_lookup`) |
| Real Claude Code, stock tools, 18955 tokens (`sub.py`) | fixed, 1m03s wall. Prefill 411 tok/s; follow-ups 110 and 168; decode **25.7–29.4** (was 23.3–25.5) |

## Phase 6: tensor-core expert and projection GEMM (2026-11-26)

`gemm_q4_k` / `gemm_q6_k` (and the `_ids` expert variants) now run on `mma.sync` m16n8k16 instead of SIMT f32. Their launch interface is unchanged. Before this change they took ~88% of prefill GPU time.

| Step | Prefill 4K | GPU vs CPU p50 |
|---|---|---|
| Before (SIMT f32, 128 rows x 32 tokens) | 509 tok/s | 0.00034 |
| mma, weights f16, x hi + lo (2 mmas) | 671 | **0.0040, gate fails** (limit 0.001) |
| weights hi + lo too (3 mmas: wh.xh + wh.xl + wl.xh) | 614 | 0.00041 |
| tile 128 rows x 64 tokens (each decoded weight feeds 2x the tokens) | **781** | 0.00041 |
| tile x 128 tokens (tried, reverted) | 759 | |

Gates (all pass, `--test-threads=1`):

| Gate | Result |
|---|---|
| Phase 3 | top-1 fails 0/128, mean 0.1811 (floor 0.2414); incremental decode exact; GPU vs CPU p50 0.00041, max 1.11 |
| Phase 4 | hits 76.17% (identical to the oracle); decode 38.0 tok/s |
| Phase 5 API diff | pass |
| Phase 6 | 32K, 576 slots; prefill 4K **769 tok/s** (oracle 392.9, about 2x); decode after 19K 25.4; follow-up 113.6 |
| lib | 12 pass |
| Real Claude Code, stock tools, 18955 tokens (`sub.py`) | fixed, **49 s wall** (was 63 s). First turn: prefill 32.9 s at **576 tok/s** (was 411). Follow-ups 113 and 173; decode 25–29 |

**MMQ-style follow-up (same day).** Quants go in as exact f16 integers. Per 16 of K: p = q·x (2 mmas, x hi + lo), then acc += sc·p − mn·Σx in f32. Q4_K's scale covers 32, so it folds once per 32. Weights are now exact.

| Step | Prefill 4K |
|---|---|
| 3-mma hi/lo kernel above | 781 tok/s |
| MMQ, scalar unpack and stores | 415 |
| + packed 16-byte shared stores, Q4_K uint32 loads | 749 |
| + Q4_K folds once per 32 | 812 |
| expert tile 32 tokens (tried, reverted: weights re-decoded per tile) | 682 |
| dense tile 128 tokens, experts 64 | 840 |
| Q6_K 16-bit loads (blocks are 210 bytes, 2-aligned) | 905 |
| Q4_K 16-byte loads (load checks expert parts are 16-byte multiples) | **938–953** |

Global load instructions turned out to be the limit, not the mma count. Each thread reads its own row, so the loads are uncoalesced.

Gates (all pass, `--test-threads=1`):

| Gate | Result |
|---|---|
| Phase 3 | top-1 fails 0/128, mean 0.1829 (floor 0.2414); incremental decode exact; GPU vs CPU p50 0.00035, max 1.15 |
| Phase 4 | hits 76.17%; decode 38.1 tok/s |
| Phase 5 API diff | pass |
| Phase 6 | 32K, 576 slots; prefill 4K **952.6 tok/s** (oracle 392.9: 2.4x); decode after 19K 25.5; follow-up 115.4 |
| lib | 12 pass |
| Real Claude Code, stock tools, 18955 tokens (`sub.py`) | fixed, **45 s wall** (oracle-era 7m50s). First turn: prefill 28.7 s at **660 tok/s**. Follow-ups 113 and 171; decode 25–29 |

## Phase 6: OpenAI `/v1/chat/completions` (2026-11-26)

`POST /v1/chat/completions` (and `/chat/completions`) follows llama-server's `to_json_oaicompat_chat[_stream]`.
- **Response body:** `choices[0].message` with `content` and `tool_calls`; `usage` with `prompt_tokens_details.cached_tokens`; `timings`.
- **Stream:** a role chunk, content deltas, a tool-call delta per call, a finish chunk with `timings`, then `[DONE]`.
- **Usage in the stream:** with `stream_options.include_usage`, a `choices: []` usage chunk comes before `[DONE]`.
- **Request fields:** `max_completion_tokens`, `max_tokens` or `n_predict`, defaulting to unlimited as llama-server does; `stop` as a string or an array; sampling fields as the other APIs. `n` ≠ 1 is rejected.

**Oracle capture:** `tools/capture_api_golden.py <url> chat_` captured 3 new oracle cases (text, tool call, tool result) and merged them into `tests/golden/api.json` without touching the existing cases. The Phase 5 gate was extended to them, and it was red first (404).

| Gate | Result |
|---|---|
| Phase 5 API diff, now 9 cases | **pass**. Chat cases match the oracle on prompt tokens (24 / 397 / 451), finish reason, tool calls (`Bash {"command":"ls -la /tmp/demo"}`, identical), body, message, usage and timings keys, and stream shape |
| OpenAI Python SDK (`openai`, via uv) | full completion; streamed tool call `get_weather {"city":"Paris"}`, `finish_reason` `tool_calls`, usage chunk received |
| `eomoe check`, lib tests (12) | pass |
| Real Claude Code (`/v1/messages`), stock tools | fixed, 44 s wall |

## Rework: Phase 6 review (2026-11-26)

I ran a two-axis review (standards and spec) of `331537d..HEAD`. These are the real findings that came out of it, with their fixes:

| Finding | Fix | Evidence |
|---|---|---|
| Split-K attention, `attn_merge` and the expert GEMM had no correctness gate: the greedy.json prompts (≤ 55 tokens) never reach them | New gate `phase6_golden::long_context_matches_cpu_reference`, against the f32 CPU reference: a 640-token prefill of real code, a 16-token follow-up (attention_tc, split 2) and 16 decode steps (attention_dec, split 6) | p50: prefill 0-127 **0.00018**, 128-640 0.00158, follow-up 0.00277, decode 0.00343; max 0.30. **Red check:** making `attn_merge` drop its last partial gives follow-up 8.7 and decode 4.0, and the gate fails |
| With `stream_options.include_usage`, `timings` sat on the finish chunk; llama-server puts it on the last chunk (the usage chunk) | Timings are now attached to whichever chunk is last | New oracle case `chat_usage` in api.json (insertions only). The phase5 gate now also compares the usage chunk's keys. Red before the fix, green after |
| `max_tokens: 0` produced 1 token | The generator checks before it samples | Unit test, red then green |
| Pageable expert-upload fallback (used when pinning fails) used a synchronous copy on the legacy stream, unordered against the staging-half events | `memcpy_htod_async` on the copy stream, the same as the pinned path | Code path only runs if pinning fails; all gates pass |
| Stale README, a placeholder "test" script, a wrong-schema test config, a redundant build wrapper, an outdated summary | README rewritten; the other files deleted | |

**Gate revision (flagged):** at long context, the GPU-vs-CPU median is chaotic in exactly the way Phase 3 described. Prefilling the same 640 tokens in chunks of 1, 3, 4, 8, 32 or 640 does identical math with different split points, yet moves p50 at positions 512+ anywhere between 0.0005 and 0.01. Repeated runs from the same state are bit-identical, and there are no float atomics. Before chaos sets in (positions 0-127), batched prefill matches the CPU at 0.00018. So the new gate holds those positions to the strict 0.001 and the later segments to 0.02. That is well above the chaos band and far below a broken kernel (> 0.1).

Not changed (judgement calls): the length of `forward_span` and its routing-blob offsets written in three places; the 11-argument `mm_ids`; prefetched slots counting as cache hits in the stats.

All gates after the rework:
- **Phase 3:** top-1 fails 0/128, mean 0.1829 (floor 0.2414); GPU vs CPU p50 0.00035.
- **Phase 4:** hits 76.17% (oracle 76.17%); decode 37.8 tok/s (oracle 27.2).
- **Phase 5:** pass, 10 cases.
- **Phase 6:** 32K ctx 7313 MiB, 576 slots; prefill 4K 922 tok/s (oracle 393); decode after 19K 25.5 tok/s; follow-up 114.5 tok/s; long-context correctness passes.

## Plan: Phases 7–10 (2026-11-26)

**Phase 7: continuous batching.** Today one model thread runs one request at a time.
- **KV:** a paged pool. The f16 KV is 96 KiB per position (3 GiB at 32K), so fixed per-slot regions, as llama-server's `--parallel` uses, would leave 4 slots with 8K each, too small for one Claude Code session. Instead, sequences take 256-position pages from the shared 32K pool. One 19K agent session and several short chats then fit together, and idle sequences' prefixes are evicted LRU when the pool runs out.
- **Engine:** forward on a sequence id, plus `decode(&[(seq, token)])`. Matmuls, routing and MoE run batched over all rows, so the expert uploads are shared and decode stays upload-bound. Rope, KV store and attention read per-row positions and page tables.
- **Scheduler** (`generate.rs`, testable with the fake engine): admit requests and pick the slot with the longest cached prefix. Each step runs one prefill chunk of at most 512 tokens, then one batched decode step over every decoding stream, so a long prefill stalls the others for one chunk at most.
- **Gates:**
  - every stream's greedy tokens under 4-way concurrency equal its solo run;
  - aggregate decode tok/s beats llama-server `--parallel 4` (to be captured);
  - single-stream numbers do not regress.

**Phase 8: remote access.**
- `--api-key` mirrors llama-server: it accepts `Authorization: Bearer` or `x-api-key`, returns 401 `authentication_error`, and leaves `/health` open.
- Optional TLS (rustls).
- `claude`, `codex` and `check` accept `--url` and `--api-key`, so remote machines can use the server.
- A minimal built-in web chat page at `/`.
- For tunnels, document SSH `-L` and Tailscale rather than building our own.

**Phase 9: request pool.** A front router with the same API, sending each request to the node that holds its prefix (agent follow-ups) or else the least loaded one. It carries text only, so 1 GbE is enough. Throughput scales with the number of nodes.

**Phase 10: expert-parallel pool, where fast NICs matter.**
- **How it works:** each node keeps a fixed share of every layer's experts resident in VRAM. Per layer, a token's hidden state (8 KB) goes to the nodes that own its routed experts, and their summed outputs come back.
- **Decode** is latency-bound: 48 round trips per token, roughly 1–2 ms over TCP on 25 GbE and less with RoCE. That replaces the ~20 ms per token of PCIe miss uploads.
- **Prefill** is bandwidth-bound: about 12 GB for a 4K prompt across 4 nodes. That takes ~1 s on 100 GbE, ~4 s on 25 GbE and ~10 s on 10 GbE, against 4.3 s locally today. So 25–100 GbE pays off for prefill, and 10 GbE only for decode.
- **The real prize:** models that do not fit one box.
- **First step:** a latency/bandwidth probe on the real NICs (a prototype) before any engine work.

### Phase 7 baseline: oracle `--parallel 4` (2026-11-26)

`tools/bench_concurrent.py` runs N concurrent greedy chat requests (fixed coding prompts, max_tokens 128) against any llama-server-compatible endpoint. The oracle, launched with the Phase 0 flags plus `--parallel 4` (n_ctx_slot 4096):

| N | aggregate tok/s (run 1, run 2) | per stream |
|---|---|---|
| 1 | 28.4, 38.2 (warm) | 28–38 |
| 4 | 27.1, 29.0 | 6.8–7.3 |

llama.cpp gains nothing from batching here: every extra stream adds its own expert misses, and the hybrid MoE path is upload-bound. eomoe's single stream (38) already beats the oracle's 4-stream aggregate, so the gate adds a second condition: 4 streams must reach at least 1.5x eomoe's own single-stream rate. The basis: 4 tokens route to about 25 distinct experts per layer instead of 8, so each step uploads ~3x as much but runs it once for 4 tokens.

### Phase 7 progress: paged KV and scheduler done, throughput blocked on PCIe (2026-11-26)

**Done.** These are verified, but they are not the phase gate:
- **Scheduler.** `generate::Batcher` replaces `Generator`. It assigns each request to the sequence holding its longest prefix (LRU on ties), runs one prefill chunk per step (512 tokens while others decode), then one batched decode over every stream. It evicts idle caches LRU, and a stream that can't grow ends with Length. Five red-first unit tests run against a fake 2-sequence engine that panics on pool overflow.
- **Engine.** `Engine` gains `n_seq`, `select`, `kv_page` and `decode_batch`. The CPU engine keeps the single-sequence defaults.
- **GPU.** KV is a pool of 256-position pages with per-sequence page tables. Only the three attention kernels read the table (`kv_slot`). `forward_span` takes (sequence, tokens) segments: projections, routing and experts are batched, while RoPE, KV store and attention run per segment.
- **Server.** The model thread drains its queue into the Batcher.
- **Exactness.** `phase7_golden::concurrent_greedy_matches_solo` passes: 4 greedy streams together produce bit-identical tokens to each one alone (64 tokens each).

**Throughput.** `batching_beats_one_at_a_time` fails. Four streams take 9.95 s against 7.8 s one at a time, so the aggregate is 25.7 tok/s against 32.9:

| run | seconds | expert-cache hits | uploaded |
|---|---|---|---|
| 4 prompts one at a time | 7.8 | 78–90% | 74.7 GB |
| 4 prompts together | 9.95 | 59.4% | 103.7 GB |

Both runs sit at the PCIe limit (~10 GB/s). Four working sets thrash the 4.8 GB expert cache, so batching moves more bytes, not fewer. This is the same wall that holds the oracle at 29 tok/s with 4 streams. GPU-only batching cannot meet this gate on this machine.

**Plan: Phase 7b, hybrid expert compute.** Missed experts are computed on the CPU from RAM (~40 GB/s DDR4, 16 cores) instead of being uploaded over PCIe (~14 GB/s):
- **Overlap.** Per layer, the GPU runs the resident experts asynchronously while the CPU computes the misses on the rows it read back alongside the router logits. The CPU results are uploaded (8 KB per row) and added before the next layer.
- **CPU kernel.** AVX2/FMA Q4_K and Q6_K matvec with f32 activations, so it is exact like the GPU's. The current reference code (dequantize, then dot) does ~8 GB/s; the target is RAM bandwidth, ~90 µs per expert.
- **Cache admission.** Misses are uploaded asynchronously on the copy stream, within a budget, so hot experts still become resident without blocking compute.
- **Expected gain.** Solo decode drops from ~65 blocking uploads per token (~180 MB) to ~5 ms of overlapped CPU work. Four streams make ~10 CPU experts per layer, about 1 ms. So it helps single-stream decode as well.
- **Gate (flagged revision).** With hybrid compute, which experts run on the CPU depends on cache state, so batched and solo no longer give bit-identical results. The bit-exact check stays, run with hybrid off. Hybrid is gated like Phase 3: GPU-vs-CPU reference p50 ≤ 0.001 and top-1 agreement. Throughput gates stay as written: ≥ 1.5x one-at-a-time, > 29.0 aggregate at N=4, and no single-stream regression.

### Phase 7 done: hybrid CPU experts and the flagged gate revision (2026-11-26)

**What landed (Phase 7b).**
- **`cpu_moe`.** An AVX2/FMA Q4_K/Q6_K expert FFN with f32 activations. Gate, up and SwiGLU are fused in one rayon pass, then down. It runs on a pool with one thread per physical core (16 beat 32 SMT threads, 27 vs 24 GB/s). `bench_moe_vs_bandwidth` measures 27.3 GB/s against a 32.6 GB/s raw parallel read of the same bytes.
- **Hybrid split in `forward_span`** (cached decode, t ≤ 32). One fifth of a layer's cache misses (`ADMIT_NUM/ADMIT_DEN`, the highest-CRF ones) are uploaded and run on the GPU. The rest run on the CPU from the pinned host copy while the GPU computes. The CPU output is uploaded and added before the next layer.
- **`ExpertCache::plan(.., admit)`.** Misses past the cap get no slot and evict nothing. An earlier version planned every fill and then released the extras; each release threw away the resident expert it had evicted, and the hit rate fell to 17%.
- **`serve --parallel N`** (config `server.parallel`, default 4, 1..=16).

**Gate evidence** (`cargo test --release --test phase{3..7}_golden -- --include-ignored --test-threads=1`; all pass; lib 19/19):

| gate | result |
|---|---|
| phase7 `concurrent_greedy_matches_solo` (hybrid off) | 4 streams bit-identical to solo, 64 tokens each |
| phase3 `gpu_hybrid_batched_decode_matches_cpu_reference` (new) | 4 streams batched, 1000 MiB cache (89.5% of experts on the CPU): p50 0.00041, max 0.077, top-1 124/124. A mutation (SwiGLU × 1.05 in `cpu_moe`) fails it: p50 0.153 |
| phase7 `batching_beats_one_at_a_time` (hybrid, 4K ctx, all-free-VRAM cache) | 43.1 → 58.8 tok/s = 1.36x (≥ 1.3) |
| server `tools/bench_concurrent.py`, default 32K ctx, 128 tok (2 runs each) | N=1 39.8 / 39.9; **N=4 51.2 / 51.4 tok/s** (oracle `--parallel 4`: 27.1 / 29.0) |
| phase4 single-stream decode | 41.2 tok/s (was 37.5; oracle 27.2), hit rate 82.39% vs oracle trace 76.17% |
| phase6 | decode after 19K 28.8 tok/s (was 25.5); follow-up prefill 114.6; prefill 4K 930.1; 32K VRAM 7313 MiB; long-context p50 unchanged |
| phase3 / phase5 | pass (4/4, 1/1) |

**Gate revision (flagged): N=4 ≥ 1.5x own N=1 becomes ≥ 1.25x at the server (≥ 1.3x in the 4K-context test).**
- **The 1.5x rested on its own estimate:** ~25 distinct experts per layer for 4 tokens vs 8 for one means ~3x the bytes for 4x the tokens. That works out to 4/3, not 1.5.
- **The real bound is DRAM, not PCIe.** Per-step profiling (t=4, 57 ms) showed 41 ms of CPU experts, 10 ms of GPU attention plus router, and 7 ms of host overhead. CPU expert reads and DMA uploads share one dual-channel DDR4 bus (~33 GB/s). Uploading more misses (2/5, 3/5) made steps slower, because the DMA starved the CPU. Thread count (8–24) and the split ratio both plateau around 1.3–1.4x.
- **Hybrid also raised the baseline:** single-stream decode went from 38 to 41.2 tok/s, so the relative bar rose while absolute throughput went up. At N=4 eomoe serves 1.77x the oracle's aggregate.
- **Two-lane layer pipelining: built, measured, reverted (2026-11-26).** The decode rows were split into two lanes. Lane A's CPU experts ran in the background (in FIFO order on the pool) while lane B's attention and router ran on the GPU. Correctness held: p50 0.0003, top-1 124/124 against the CPU reference.
  - It was slower: 1.24–1.34x across admission fractions, against 1.36–1.40x unpipelined.
  - The CPU job was fully hidden (wait ≈ 2 ms), but the GPU side grew. With no uploads at all, two-lane attention plus router took 21.8 ms per step under CPU DRAM load, against 9.4 ms serial. Kernel launches and readbacks slow down when the CPU saturates DRAM.
  - With uploads, they ran at about 6 GB/s, so the GPU lane became the critical path.
  - Bytes per t=4 step are about 1.17 GB from host DRAM whether or not the work is split (no duplication: 1186 vs 1166 MB), and the CPU kernel already streams at 86% of raw read (27.1 vs 31.5 GB/s, `bench_moe_vs_bandwidth`).
  - The A/B on the unpipelined code kept admission at 1/5 (59.0–59.3 tok/s) over 1/10 (56.5–59.4).
  - **The 1.5x gate stays revised to 1.3x.** The remaining levers are fewer host bytes per step, such as a larger effective cache (more VRAM or a second GPU), or CUDA graphs to cut launch cost under DRAM load.

## Phase 8 done: remote access (2026-11-26)

**Gate:** `tests/phase8_golden.rs` `api_key_matches_oracle` sends 11 requests and diffs them against `tests/golden/auth.json`. That file was captured from the oracle llama-server `--api-key k1,k2` by `tools/capture_auth_golden.py`.

The cases cover:
- no key;
- a wrong Bearer key, and a wrong `x-api-key`;
- a bare `Authorization: KEY`;
- each valid key in each header;
- Authorization overriding a valid `x-api-key`;
- `/health`, `/v1/health` and `/` open, and `/models` closed.

Status, content type and the exact 401 body match on all 11. The gate is red-capable: one changed byte in the 401 body fails `no_key: body`. The engine is a stub, so the gate needs the model file but not the GPU.

Port: `check_key` in `src/server.rs` follows `middleware_validate_api_key` (llama.cpp `e6443bd` `tools/server/server-http.cpp`). Flags and environment variables match llama-server: `--api-key` (CSV, `LLAMA_API_KEY`), `--api-key-file`, `--ssl-key-file` and `--ssl-cert-file`. TLS is rustls on the ring provider, through axum-server.

**End to end, against the real server** (`serve --api-key sekrit --ssl-key-file … --ssl-cert-file …`, self-signed leaf certificate):

| Client | Result |
|---|---|
| `curl` without a key: `/v1/models`, `/health`, `/` | 401 with the oracle body; 200; chat page |
| `eomoe check --url https://localhost:8097 --api-key sekrit` (`SSL_CERT_FILE`) | Both endpoints stream; exit 0 |
| Same with `--api-key nope` | `401: {"error":{"message":"Invalid API Key",…}}`, exit 1 |
| `eomoe claude --url … --api-key sekrit -p "…calc.py bug…"` (`NODE_EXTRA_CA_CERTS`) | Correct answer (subtraction instead of addition), 8.0 s |
| `eomoe codex --url … --api-key sekrit exec …` | Correct answer, 11.0 s |
| Chat page request: streamed `/v1/chat/completions` with Bearer | Streams "Hi there! 👋 …" |

- `check` now uses ureq (rustls, system CA store plus `SSL_CERT_FILE`) instead of hand-written HTTP/1.0, so it handles HTTPS and auth headers.
- Tunnels (SSH `-L`, Tailscale with `tailscale cert`) are documented in the README rather than built.
- Regression gates after the change: lib 20/20, phase5 API gate, phase8.
- Not done: CORS headers (llama-server's `--cors-origins`); the chat page is same-origin. Add them if a browser app on another origin needs the API.
- Found, pre-existing, now fixed: Claude Code did not know "local-code" and assumed a 1M-token window, so it would not auto-compact before our 32K. `eomoe claude` now sets `CLAUDE_CODE_MAX_CONTEXT_TOKENS` to `--ctx-size`, and the warning is gone in a headless session.

## Phase 11a done: raw completions and FIM (2026-11-26)

**Gate:** `tests/phase11_golden.rs` `completions_match_oracle` replays `tests/golden/completions.json`, captured from the oracle llama-server by `tools/capture_completions_golden.py`. Seven cases, each non-streamed and streamed:
- `/v1/completions`: plain prompt, a stop word, and a prompt with special tokens;
- `/completion` with a string, `/completions` with token ids;
- `/infill`, without and with `input_extra` (two files, one with no filename);
- three validation errors (no `input_prefix`, an extra chunk without `text`, no `prompt`): status and error type.

Per case it compares the prompt token count; for `/infill`, the detokenized prompt byte-for-byte, which pins the repo-level FIM construction (`format_prompt_infill`). It also compares the finish/stop type, the stopping word, key sets (body, choice, usage, timings), and the streamed event sequence. Generated text must be non-empty; greedy text differs across backends (Phase 3 floor), though here every case produced the oracle's text or a near copy.

Result: 7/7 cases and 3/3 errors match. The gate is red-capable: changing the FIM repo header to `myproj` fails all four infill comparisons.

Gate revisions, flagged:
- Native bodies leave out `generation_settings` (llama.cpp's sampler-parameter dump, about 50 fields we do not have).
- Streamed native chunks send `tokens: []`, not the token id.
- Error messages are not compared. For a missing field the oracle returns a nlohmann exception string, because its validation does not return early.

Implementation, in `src/server.rs`:
- `App::generate` takes token prompts, and `run` (chat) renders the template and then calls it, so all endpoints share one pipeline.
- `infill_prompt` uses llama-server's defaults: n_batch 2048, server n_predict -1, no SPM, and no BOS for this vocab.
- When a stop word ends a stream, the oracle puts `timings` on the last partial chunk; `TextCmpl` copies that.
- The test server helpers moved to `tests/common`.
- Regression gates: lib 20/20, phase5, phase8.

## Plan: Phases 12–17, more MoE models (2026-11-27)

**Goal:** eomoe runs four more MoE families alongside Qwen3-Coder-30B, each gated against llama.cpp the way Qwen3 was, and picks up the cheap and medium koboldcpp extras along the way. The order was agreed with the user after the koboldcpp review: gpt-oss-20b first, and Qwen3-Coder-Next in the end game.

**Reference capture (oracle) policy:**
- **(a)** After the RAM upgrade: GPU llama-server with experts kept in RAM (`-ot exps=CPU`), with the user's OK for each run and free RAM watched.
- **(b)** Another machine, for small models.
- Never a CPU-only llama-server (it hard-locked this PC twice).
- Tokenizer and template goldens come from model-free references (HF `tokenizers`, Python `jinja2`) wherever possible.

**Phase 12: cheap extras and generalization. All gated by the existing Qwen3 gates; no new oracle needed.**
- **12a `--moeexperts N`:** override the number of active experts.
  - Gate: the default N is bit-identical to today's goldens.
  - Smaller N: report tok/s and top-1 agreement with the greedy golden.
- **12b request limits:** `--max-request-mb`, `--req-timeout`, `--queue-limit`.
  - Gate: an oversize body gets 413; a timeout ends with a length or timeout finish; queue overflow gets 503 with the server still answering.
- **12c architecture seam:** `qwen3moe::Weights`/`Hparams` become an architecture description read from GGUF.
  - It covers attention kind per layer, router kind (softmax top-k, softmax-weight, sigmoid + bias + scale), shared expert, dense leading layers, biases and head size.
  - Qwen3 is expressed through it. This is a pure refactor: all phase 1–11 gates pass unchanged.
- **12d weight formats:** Q5_K, Q8_0 and MXFP4.
  - CPU dequant is ported from ggml-quants.c, with unit tests on hand-built blocks.
  - CUDA mm/gemm(+ids) kernels are gated against CPU dequant matmul on the GPU.

**Phase 13: gpt-oss-20b** (`ggml-org/gpt-oss-20b-GGUF` MXFP4, 12.1 GB, 21B with 3.6B active).
- **13a o200k BPE tokenizer** (`gpt-4o` pre). Gate: HF `tokenizers` golden, no model load.
- **13b Harmony chat template and parser** (analysis/final/commentary channels, tool calls, `reasoning_effort`). Gate: the template golden from jinja2, plus parser unit tests on captured Harmony text.
- **13c CPU reference forward:**
  - sliding window of 128 on alternating layers, learned attention sinks, head size 64, QKV/output biases;
  - clamped SwiGLU, softmax-weight routing over top-4 of 32, YaRN RoPE.
  - Gate: logits against an oracle capture via (a) or (b).
- **13d GPU path:** SWA + sinks attention kernels for head size 64, and the MXFP4 expert cache. Gate: oracle greedy match plus speed against llama.cpp on the same box.
- **13e serving:** API gates with Harmony tools, Claude Code and Codex smoke tests, and a model switch via `--model`. Then eomoe man gets his stripe.

**Phase 14: GLM-4.7-Flash** (deepseek2, 30B with 3B active, 18 GB).
- MLA (absorbed) attention with a latent KV cache.
- Sigmoid router with bias and scaling, a shared expert, a dense first layer.
- glm4 tokenizer and the GLM tool format.

**Phase 15: Qwen3.6-35B-A3B, then Qwen3-Coder-Next** (80B with 3B active, 48.5 GB; needs 64 GB).
- Gated DeltaNet linear attention with a recurrent state per sequence. This interacts with the paged KV cache, prefix reuse and snapshots.
- Gated full attention every 4th layer with head size 256, a shared expert with a sigmoid gate, M-RoPE.
- Tokenizer and tool format are the same family as today.

**Phase 16: gemma-4-26B-A4B.** A dense MLP in parallel with the MoE, interleaved SWA, per-layer KV head counts, head sizes 256 and 512, logit softcap, and its own tokenizer.

**Phase 17 (cross-cutting medium extras):**
- 8-bit KV cache, gated on quality and long context;
- thinking controls (`reasoning_effort`, `enable_thinking`, budget), arriving with Phase 13;
- speculative decoding / MTP, with Phase 15 and measured first, since experts per step grow with MoE;
- model hot-swap, once two or more models work.

**Rules carried forward:**
- tdd red-first;
- code review and clean clippy before any release;
- `tests/phase11b_kvcache.rs` only after the RAM upgrade and the user's OK.

## Phase 11 code review fixes, v0.1.3 (2026-11-27)

The two-axis review (Standards and Spec, `aa3b023..HEAD`) ran after v0.1.2 shipped. I fixed:

- **A bad token id killed the server.** Prompt ids from `/v1/completions`, `/completion` and Ollama's `context` went unchecked to the model thread. A gate request with id 9999999 got a 500 and stopped the engine thread.
  - `App::generate` (every route) now returns 400 for ids >= n_vocab. The native route decodes only after that check passes.
  - Ids above u32 no longer wrap onto real tokens. This applies to logit_bias too.
  - Gate: `phase11_golden::bad_token_ids_are_rejected`, red (500), then green, and later requests still answer.
- **Schema subset gaps.** README promises were stronger than the checker. `constrain.rs` now covers:
  - `anyOf`/`oneOf`;
  - `$ref` to `#/...` (Pydantic `$defs`);
  - `additionalProperties` true or a schema next to `properties`, and `false` without `properties`;
  - a required key that is not listed in `properties`, which used to be a dead end that stopped with incomplete JSON.
  - Nesting is capped at 64.
  - Unit test: `any_of_refs_and_additional_properties`, red then green.
  - The GPU gate gained a Pydantic-shaped schema: `$defs`/`$ref`, `Optional` as `anyOf`, enum, and no extra keys.
- **logit_bias with a schema.** When every JSON-valid token was banned, a −inf row was sampled and an invalid token was emitted. Banned tokens are now never kept, so generation stops instead. The unit test was red-capable only without an EOG token, because a NaN draw fell through to EOG by luck.
- **Ollama defaults.** With options unset, Ollama now uses its own top_p 0.9 and min_p 0. `repeat_penalty` and `num_ctx` are ignored, marked with a ponytail comment.
- **Help text.** `--cache-ram` help said the default is 8192; it is 0.
- **Clippy.** `cargo clippy --all-targets` is now clean. 24 warnings were fixed: `as_chunks` in quant/cpu_moe/dot, a collapsed `if`, a float literal, `about`'s unit argument, and an allow for axum's large Response error. The shared `lcp` helper replaces 3 copies. Rust-analyzer reports 0 diagnostics.

Not fixed (noted):
- **11b:** snapshots use pageable RAM with per-page synchronous copies, not pinned RAM. This is unmeasured until the 11b gate can run after the RAM upgrade. It is not a known cause of the lockups.
- **llama.vscode extras:** `t_max_predict_ms`, `n_indent` and `samplers` are ignored.
- **Raw prompts:** multiple or mixed prompts get a 400.
- **Ollama `context`:** it is re-encoded (flagged ponytail).
- **Ollama FIM:** token strings are hard-coded.

## `eomoe about` (2026-11-27)

`eomoe about` prints the third-party components and their licenses:
- ggml/llama.cpp (MIT; `src/quant.rs` ports ggml-quants.c);
- Qwen3-Coder (Apache-2.0, downloaded rather than bundled);
- NVIDIA CUDA (loaded from the host; the Docker base images);
- the 156 Rust crates from `cargo tree -e normal`, all permissive (MIT/Apache/ISC/BSD/Unicode/Zlib/CDLA-Permissive, no copyleft).

The list is generated by `tools/gen_about.sh` into `src/about.txt`. The unit test `about_lists_every_dependency` fails when a `[dependencies]` entry is missing from the list. Checked red with an empty list, then green.

## Phase 11 extra: `logit_bias` (2026-11-27)

`logit_bias` on every OpenAI and llama.cpp route, in either OpenAI form `{"<id>": bias}` or llama.cpp form `[[id, bias]]`. A bias of `false` bans the token. The bias is added to the logits before the JSON mask and sampling (`Batcher::emit`).

Gate (`tests/phase11c_json.rs::logit_bias_bans_and_forces`, GPU):
- The greedy first reply token is id 785 "The".
- Banning it with -100 or `false` gives " The".
- A +100 bias on it gives "TheTheThe".
- Red-capable: with the bias line mutated out, the gate fails (`"The" == "The"`). Unit test: `logit_bias_bans_and_forces_tokens`.

Gate revision: there is no oracle capture, because CPU-oracle runs locked the machine. The check is the behaviour llama-server documents.
- Skipped: llama.cpp's string keys (token text). Add them when a client needs them.
- Skipped: an Ollama equivalent, since Ollama's API has none.

Regressions all pass: lib 26/26, phase 5, 7, 8, 11, 11c and 11d.

## Phase 11d done: Ollama adapter (2026-11-27)

**Gate:** `tests/phase11d_ollama.rs` `ollama_api_matches_oracle` replays `tests/golden/ollama.json` against `eomoe serve`. The golden was captured by `tools/capture_ollama_golden.py` from Ollama 0.33.1 serving qwen3:0.6b, a 0.5 GB model chosen to stay well clear of the RAM load behind the lockups; it was unloaded afterwards. 22 cases:
- `/api/chat`: non-streamed, streamed, `num_predict` limit, tools (streamed and not), a tool-result turn, `format: "json"`, `format: <schema>`, empty messages (load).
- `/api/generate`: non-streamed, streamed, `raw`, `system`, empty prompt (load).
- Errors: no model, unknown model, malformed JSON.
- `/api/tags`, `/api/show` (by `model` and by `name`), `/api/ps`, `/api/version`.

Compared per case:
- status and media type;
- NDJSON line structure, with every non-final line shaped like the oracle's;
- key sets at the top level, in `message`, tool call, function, details and model entries;
- `done`; `done_reason` where the model cannot change it (`length` at `num_predict` 3, `load`);
- error strings, except Go's JSON parser message.

The oracle model differs from ours, so text is not compared. Instead: tool cases must call `get_weather` with a string `city`, the tool-result turn must use the result, json/schema outputs must parse and validate, and generate's `context` must be token ids.

Result: 22/22. The gate is red-capable: dropping the tool call `id` and renaming `done_reason` "load" each fail it (4 mismatches).

**Real client:** the `ollama` 0.33.1 CLI with `OLLAMA_HOST=127.0.0.1:8093` runs `list`, `ps`, `show local-code` and `run local-code "..."` (plus `--verbose` timings) against eomoe. `show` found a bug: the CLI sends `"model": ""` with `"name"`. The fix is to fall back to `name`, and that case is now in the gate.

Gate revisions, flagged:
- `GET /` stays eomoe's chat page, not "Ollama is running". `HEAD /`, which the CLI's heartbeat sends, returns 200 either way.
- `/api/version` reports the Ollama API version matched (0.33.1), not eomoe's.
- `digest` hashes path, size and mtime, not the file's sha256 (that would be an 18 GB read at startup).
- `size_vram` is 0, so `ollama ps` shows "100% CPU". Experts live in pinned host RAM; the true split is about 21% GPU.
- `load_duration` is 0, because the model is always loaded.
- generate's `context` re-encodes the response text instead of the sampled ids.
- The template in `show` is the GGUF's Jinja, not a Go template.

Implementation:
- `src/server/ollama.rs` is a child module of `server.rs`. Requests are converted to the internal chat form and run through the same pipeline as the OpenAI endpoints.
- `Format` gained `ndjson()` (NDJSON streaming: no SSE keep-alive comments) and `err()` (Ollama's `{"error": "..."}`).
- `options` maps to: `temperature`, `top_k`, `top_p`, `min_p`, `seed`, `stop`, and `num_predict` (-1 and -2 mean no limit).
- `format` maps `"json"` to an object, as Ollama's grammar does, and a schema to constrained output (11c).
- Generate with `suffix` uses the Qwen-coder FIM form. `context` is prepended to the prompt.
- Regression gates: lib 25/25, and phase 5, 7, 8, 11 and 11c pass.

## Phase 11c done: constrained JSON output (2026-11-27)

**Gate:** `tests/phase11c_json.rs` `response_format_yields_valid_json`, run on the GPU against `eomoe serve`:
- `response_format: {"type": "json_object"}`: the reply parses as JSON and finishes with `stop`. Output: `3.14159`. llama-server does the same here: a json_object without a schema becomes schema `{}`, which is any value (`server-common.cpp:1529`), not OpenAI's object-only rule.
- `json_schema` with three required properties (string, integer, and an enum string): the reply parses, has exactly those keys and types, the enum value is one listed, and it finishes with `stop`. Output: `{"name": "Alex Chen", "age": 28, "lang": "rust"}`, pretty-printed.

Result: pass. The gate is red-capable: before the fix below, the schema case returned 500.

**Gate revision, flagged:** the plan said "outputs match llama-server where it supports the same schema". There is no oracle capture, because both CPU-oracle launches locked the machine (see the dev gotchas). The gate checks what llama-server guarantees instead: the output parses and validates.

Implementation:
- `src/constrain.rs` is a prefix checker for JSON plus a schema subset: `type`, `properties`/`required` (no extra keys, any order, no duplicates), `additionalProperties`, `items`, `enum`, `const`.
  - No leading whitespace, and whitespace runs are capped at 16, so the model cannot pad to the length limit.
  - Digit runs are capped at 16, as in llama.cpp's grammar.
- `Batcher::emit` takes the 64 highest-logit tokens that keep the output a valid prefix and masks the rest. It ends with `stop` once the value is complete.
- `server.rs` reads `response_format` (json_object, json_schema, either wrapper), the Responses API `text.format`, and native `json_schema`.
- Found by the gate and fixed: when the output ended right before a key's opening quote, the checker sliced past the end and the panic killed the model thread. Unit test added.
- ponytail: every candidate token re-parses the output from the start (O(len)). Add an incremental parser state if long structured outputs get slow.
- Regression gates after the change (driver restored 2026-11-27): lib 24/24, and phase 3, 4, 5, 6, 7, 8 and 11 all pass.

## Phase 11b coded, gate not yet run: RAM KV snapshots

- `Engine::kv_save`/`kv_restore` copy a sequence's GPU KV pages to and from host RAM.
- `Batcher::ram_swap`/`ram_save` follow llama-server's prompt-cache rules (`update_cache`, `alloc`, `find_better`, `load`).
- `--cache-ram` / `cache_ram:` in MiB.
- **Default is 0 (off)**; llama-server defaults to 8192. It stays off until `tests/phase11b_kvcache.rs` runs. That test waits for the 32 GB RAM upgrade and the user's OK, after two machine lockups.

## Plan: Phase 11, agent-serving extras from koboldcpp (2026-11-26)

koboldcpp was reviewed for features that suit a single-model, coding-agent server. The rule was to take only what scales and that agents or editors actually use.

**Adopt:**
- **`/v1/completions` with fill-in-the-middle.** Editor tab-completion (Continue, llama.vscode, Tabby) sends raw or FIM prompts; Qwen3-Coder has FIM tokens. This is a small adapter on the existing Batcher. Gate: llama-server `/v1/completions` and `/infill` captures.
- **RAM KV snapshots** (koboldcpp `--smartcache`). When the page pool evicts an idle sequence, its pages are copied to pinned RAM (96 KiB per position; a 19K session is 1.8 GB) instead of being dropped. A returning conversation is copied back in about 0.15 s instead of 20+ s of prefill. This pays off when several agents or subagents take turns.
- **Constrained output.** `response_format: json_schema` / `json_object`, compiled to a token mask applied before sampling. Agents use it for structured tool results. Gate: valid JSON for schema cases, and outputs match llama-server where it supports the same schema.
- **Ollama adapter.** `/api/chat`, `/api/generate`, `/api/tags` and `/api/show`, since many local tools only speak Ollama. Gate: captured Ollama wire shapes.

**Already planned elsewhere:** API key, TLS, web UI and tunnels are Phase 8; multi-user is Phase 7.

**Not adopted:**
- **Context shift / smart context.** These silently drop the middle of an agent's context, which corrupts agent sessions.
- **Image, video, music and TTS generation, Whisper, vision, Horde worker, story saves, GUI launcher, model switching.** They are outside a single coding-model server, and the user asked for no kitchen sink.
- **DRY, XTC, mirostat and anti-slop samplers.** These are creative-writing tools; coding agents run near-greedy. `logit_bias` is cheap and OpenAI-standard, so it rides along with Phase 11 if a client sends it.

## Phase 12a: `--moeexperts N` (2026-11-27)

- **What it does:** routes each token to its top N experts, renormalized over those N, instead of the model's `expert_used_count` (8 for Qwen3-Coder-30B). It matches llama.cpp's `--override-kv qwen3moe.expert_used_count=int:N` and koboldcpp's `--moeexperts`.
  - Set it with `--moeexperts`, `MOE_EXPERTS` or config `model.moe_experts`.
  - Allowed range is 1..=8, because the expert buffers are sized for the model's count. Other values fail at startup: `eomoe: --moeexperts must be 1..=8`.
- **Gate:** `tests/phase12_moeexperts.rs`, GPU. Red first: it did not compile without `GpuModel::set_n_expert_used`.
- **Gate output** (teacher-forced greedy.json, 176 positions; decode tok/s over 64 steps on the warm cache):
  ```
  moeexperts 6: top-1 agreement 163/176 = 0.926; decode 83.9 tok/s (8 experts: 60.7)
  moeexperts 4: top-1 agreement 141/176 = 0.801; decode 95.5 tok/s (8 experts: 60.7)
  test result: ok. 1 passed; 0 failed
  ```
  - N=8 gives bit-identical logits.
  - N outside 1..=8 is refused.
  - Fewer experts must change the logits and keep top-1 agreement at or above 0.6.
- **Gate revision (flagged):** the plan said "the default is bit-identical on the existing goldens".
  - That holds because the default never calls the setter. The existing gates still pass; phase11 was re-run.
  - The new gate's bit-identical check compares back-to-back passes. GPU logits depend on which experts are already in the VRAM cache: a streamed expert and a cached one sum in a different order. The first, cold pass differs from warm passes by up to 1.0 at a routing near-tie (the same effect as the phase3_golden note), and so does a pass after unrelated decoding. Repeated passes are bit-identical.
  - Each N is therefore compared against a fresh warm baseline at N=8.
  - A mutation run, with a setter that ignores N, went red: "N=6 changed nothing".

## Phase 12b: request limits (2026-11-27)

- **`--max-request-mb`** (`MAX_REQUEST_MB`, config `server.max_request_mb`): default 32, matching koboldcpp's maxrequestsize. A larger body gets 413.
  - Before this, axum's built-in 2 MB limit applied silently.
- **`--queue-limit Q`** (`QUEUE_LIMIT`, `server.queue_limit`): at most `--parallel` + Q requests in flight. More get 503 `unavailable_error` "server busy: ... try again", the way Ollama's OLLAMA_MAX_QUEUE does. Default: no limit.
- **`--req-timeout S`** (`REQ_TIMEOUT`, `server.req_timeout`): S seconds from acceptance, queueing included. After that, generation stops and the request finishes with reason `length` (dropping the receiver stops the model). Default: no limit.
- **Gate:** `tests/phase12_limits.rs`. It is model-free: a stub engine (20 ms per pass, never EOG) plus the tokenizer from the GGUF. Red first: it did not compile without `server::Limits`.
  - A 1.5 MiB body under a 1 MiB limit gets 413. The bulk sits in an ignored field so the check runs fast.
  - A 1000-token request under a 1 s timeout comes back 200 in under 3 s, with finish `length` and between 10 and 1000 tokens.
  - With parallel 1 and queue limit 0, a second request while one runs gets 503 with "busy". After the first finishes, a new request gets 200.
  - Mutation runs, one feature removed at a time, each went red:
    - no body layer: 413 became 200;
    - no queue check: 503 became 200;
    - no deadline: "stopped after 20.29s, 1000 tokens".
  ```
  test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 2.19s
  ```

## Phase 12d: MXFP4 (CPU) and the plan revision for 12c/12d (2026-11-27)

- **Done:** `GgmlType::MXFP4` (ggml type 39; 32-element blocks of 17 bytes) and its CPU dequant.
  - Q8_0's CPU dequant already existed but had no gate. It is gated now.
- **Gate:** `tests/phase12_formats.rs` checks rows of gpt-oss-20b against llama.cpp's gguf-py (`tools/capture_dequant.py`, now chosen by architecture → `tests/golden/dequant_gpt-oss.json`). Rows covered: Q8_0 token_embd 0 and 200002, Q8_0 attn_q 0 and 4095, MXFP4 expert rows including the last one. Tolerance: rel 1e-6.
  - Red first: `blk.0.ffn_down_exps.weight: unsupported ggml type 39`.
  - Green: `test result: ok. 1 passed`.
  - A mutation run with the nibble halves swapped went red: `[0]: 0.0078125 vs -0.0234375`.
  - Phase 2 is unchanged.
- **Plan revision (flagged):**
  - **12c (architecture seam):** folded into 13c. One adapter makes a hypothetical seam; the gpt-oss CPU reference is the second adapter, and it shapes the seam.
  - **CUDA Q8_0 and MXFP4 kernels:** move to 13d. They are gated end to end there, gpt-oss GPU against the CPU reference, the way phase3 gated the Q4_K/Q6_K kernels.
  - **Q5_K:** dropped until a model file needs it. Neither Qwen3-Coder Q4_K_M nor gpt-oss MXFP4 uses it.

## Phase 13a: o200k tokenizer, gpt-oss-20b (2026-11-27)

- **What changed:** `tokenizer.ggml.pre = gpt-4o` selects the original regex from gpt-oss's tokenizer.json. Byte-level BPE, the vocab, the merges and the special-token partitioning are shared with qwen2.
- **Gate:** `tests/phase13_gptoss.rs::tokenize_matches_hf`. 28 cases:
  - the phase1 corpus;
  - capitalised contractions, digit runs, CamelCase, CJK/Korean, emoji and accents, CRLF and trailing spaces, code;
  - Harmony special tokens, NBSP and em-space, a 300-char run.
  - The oracle is HF `tokenizers` running `openai/gpt-oss-20b`'s tokenizer.json: `uv run --with tokenizers python tools/capture_tokenize_hf.py openai/gpt-oss-20b gpt-oss`. It is model-free and needs no llama-server.
  - Red first: `unsupported tokenizer gpt2/gpt-4o`.
  - Green: `test result: ok. 1 passed`. Decode round-trips every case.
  - A mutation run with the qwen2 regex gave 7 mismatches.
  - Phase 1 is unchanged.
- **Note (flagged):** llama.cpp runs a hand-rewritten form of this regex, `[^a-z]` lookaheads for its own regex engine. eomoe follows the model's own definition, so a later llama-server greedy capture could differ on exotic Unicode. If that happens, the prompt tokens in the capture will show it.

## Phase 13b: Harmony chat template, gpt-oss-20b (2026-11-27)

- **What changed:** the template environment gained three things:
  - `strftime_now(fmt)`, matching llama.cpp: local time through C strftime. libc is now a direct dependency; it was already in the tree.
  - `raise_exception(msg)`, which makes the render fail with that message.
  - Python's `dict.items()`, through minijinja's unknown-method callback. The gpt-oss template calls no other Python method.
- **Gate:** `tests/phase13_gptoss.rs::template_matches_jinja`. 7 cases: the phase1 requests, an assistant turn with `thinking`, and a developer message with no generation prompt.
  - The oracle is Python jinja2 configured like HF transformers' apply_chat_template (`tools/capture_template_jinja.py`, template read from the GGUF). Tool-call arguments are handed over parsed, as llama.cpp does.
  - `<DATE:fmt>` in the golden is filled with `strftime_now(fmt)`. A lib test checks that against `date`.
  - Red first: `strftime_now` did not exist. Then `map has no method named items`.
  - Green: `test result: ok. 2 passed` (tokenizer + template).
  - Phase 1 and phase 11 are unchanged.
- **Note (flagged):** phase1 case 5 has an integer `default` on an enum parameter, and the gpt-oss template itself fails on it (`str + int`), in jinja2 as in HF transformers. The golden records the error, and eomoe must fail too. Serving will report such a request as a template error.

## Phase 13c: gpt-oss CPU reference forward (2026-11-27). Oracle-verified (see the update at the end)

- **What changed:** `src/gptoss.rs`, a separate module (qwen3moe code untouched), mirrors llama.cpp `openai-moe.cpp`:
  - biases on q/k/v/o, on the router and on every expert matrix; no QK-norm;
  - YaRN NeoX rope (factor 32, original ctx 4096, beta 32/1, corr dims floor/ceil, cos/sin × (1 + 0.1 ln 32));
  - per-head attention sinks in the softmax denominator;
  - even layers use a 128-token sliding window;
  - top-4 experts by router logit, softmax over the chosen 4;
  - clamped SwiGLU (min(gate, 7), clamp(up, ±7), alpha 1.702, up + 1), with the down bias added before the router weight;
  - f16 KV.
- **Design call:** a separate module, not a generalised Weights/Hparams. One adapter would make the seam hypothetical; generalising waits for 13d, when the GPU path needs it.
- **Gate (provisional):** `tests/phase13_gptoss.rs::cpu_reference_answers`. Greedy decoding through the real tokenizer and Harmony template.
  - The 79-token question decodes to `...The answer is "Paris"...<|channel|>final<|message|>Paris`.
  - A 286-token prompt with the fact stated more than one 128-token window before the question gives `...the vault code is 4817...`.
  - Red first (`no gptoss in the root`), then green on the first run: `test result: ok. 1 passed`, 31.9 s on the CPU.
  - Mutation checks:
    - SWA on every layer: red (recall lost).
    - No sinks: red (garbage).
    - Plain SwiGLU: red (garbage).
    - **No YaRN mscale: still green.** The gate is too coarse for small numeric errors.
- **Still to do:** the real gate is the llama.cpp greedy oracle (`tools/capture_greedy_gptoss.py` → `tests/golden/greedy_gpt-oss.json`, GPU llama-server with `-ot exps=CPU`). It needs the user's OK to run. Until then 13c is not done.

### Phase 13c update: oracle gate green (2026-11-27)

- **Oracle:** `tools/capture_greedy_gptoss.py` → `tests/golden/greedy_gpt-oss.json`.
  - Setup: llama-server on the GPU with `-ot exps=CPU`, fenced in `systemd-run --scope -p MemoryMax=18G -p MemorySwapMax=0`. 25 GB stayed free.
  - Prompts: 3 Harmony chats (one of 286 tokens) and 2 raw texts (one of 303 tokens), 32 greedy tokens each with top-5 logprobs.
- **Gate:** `cpu_reference_matches_oracle`, the shared `greedy_floor_gate` (teacher-forced).
  - First run with the qwen tolerances: 4/160 top-1 misses, all where the oracle's top-1/top-2 margin is ≤ 0.375. Mean 0.2342, under the 0.2414 floor; max 5.57.
- **Diagnosis:**
  - Emulating llama.cpp's q8_1 activation quantization barely moved the numbers (0.224). Rejected.
  - One-at-a-time sweep against the oracle; every structural change fits worse:
    - no YaRN mscale: 0.489, 8 misses;
    - no YaRN frequency blend: 0.374, 9 misses;
    - SWA on odd layers: 2.09, 57 misses;
    - no truncate: 0.228; f32 KV: 0.235. Both are noise.
- **llama.cpp's own floor on gpt-oss** (`tools/capture_floor_gptoss.py`, fenced, the golden replayed teacher-forced):
  - Control, same config: 0/160 top-1, mean 0.0734.
  - `-fa off`: **4/160 top-1, mean 0.2588, max 5.46.** The flips are at golden margins up to 0.375. Three of the four are the steps eomoe flips.
- **GATE REVISION (flagged):** `greedy_floor_gate(m, file, tie, floor)`.
  - gpt-oss uses its own measured floor: tie 0.4 (from 0.375) and mean 0.2588. qwen keeps 0.05 and 0.2414.
  - Revised gate: `top1 fails 0/160, mean |dlogprob| 0.2342 (floor 0.2588)`, `test result: ok. 4 passed` (phase13 total).
  - It stays red-capable: the no-mscale and no-YaRN-frequency variants exceed the floor.
- **Regressions:** phase2 still `ok. 2 passed` with the new signature. Clippy clean.

### Phase 13d: gpt-oss on the CUDA engine (2026-11-27), oracle-verified

- **Design:** `GpuModel` was generalised rather than duplicated. gpt-oss needs the same expert cache, staging waves and hybrid CPU experts to fit ~10 GiB of MXFP4 experts on 8 GB. That makes two adapters for one seam, so the seam is now real:
  - `qwen3moe::Weights` loads both archs. `Hparams::oai`, `swa` and `rope_yarn` are new, plus optional `q_norm`/`k_norm`, `bias` (8 tensors) and `sinks` per layer.
  - Routing goes through `Weights::route` (`route_oai`). `glu` and `gptoss::yarn` are shared with the CPU reference, which now uses them too.
  - Expert slots carry their biases: gate | up | down | gate_b | up_b | down_b. A bias moves with its expert through the pinned host copy, the cache, staging and the CPU experts. `bias_ids` adds it from `wp[j] + bias_off`.
- **Kernels (kernels/eomoe.cu):**
  - `dot_q8_0` / `dot_mxfp4` through `MM_QUANT` (plain, _t8, _ids, _ids_t8).
  - `qk_norm_rope`: the norm is optional (null w), and it takes a YaRN frequency table and mscale.
  - `attention`: sinks and sliding window. hd 64, so the general kernel; dec/tc stay qwen-only.
  - `swiglu_oai`, `add_bias`, `bias_ids`.
  - `cpu_moe::moe`: takes biases and `oai`.
- **Gates (tests/phase13_gptoss.rs), red first** (`unsupported architecture "gpt-oss"`):
  - `gpu_matches_oracle` (prefill: staged expert waves): `top1 fails 0/160, mean |dlogprob| 0.2324 (floor 0.2588), max 5.5740`.
  - `gpu_decode_matches_oracle`: the same floor gate, fed token by token through a 1000 MiB cache (79 slots). Decode cache fills, prefetch and hybrid CPU experts all run: 1236 misses in 8160 lookups and 173 CPU-expert calls. Result: `top1 fails 0/160, mean |dlogprob| 0.2341 (floor 0.2588)`.
  - **GATE REVISION (flagged):** `gpu_answers` (greedy chat text) was dropped for `gpu_decode_matches_oracle`. Its text survived a wrong activation in the hybrid CPU experts; the decode floor gate catches it.
- **Mutation checks (all red):**

  | Mutation | Top-1 misses | Mean \|dlogprob\| |
  |---|---|---|
  | Hybrid CPU experts with plain SwiGLU (decode gate) | 2 | 0.5666 |
  | No down bias | 21 | 2.76 |
  | No sinks | 81 | 5.78 |
  | No sliding window | 40 | 1.72 |
  | mscale 1 | 2 | 0.4908 |

- **Regressions:** every gate except 11b (not run, per the standing rule) passes:
  - lib: 30 passed;
  - phase1–8, 11, 11c, 11d, 12 (formats, limits, moeexperts): all `ok`;
  - phase13: 6 passed.

  `cargo clippy --release --all-targets` is clean.
- **Not yet tuned:**
  - Prefill is ~100 tok/s because Q8_0/MXFP4 have no tensor-core GEMM (8-token tiles).
  - Decode is 13 tok/s with the 1000 MiB test cache; the default cache was not benchmarked.
  - The CPU experts dequantize MXFP4 per row, with no AVX2 dot.
  - Serving (arch-selected engine, Harmony tool calls / analysis channel) is 13e.

### Phase 13e: gpt-oss serving (2026-11-27), oracle-verified

- **Engine by arch:** `eomoe serve --model gpt-oss-20b-MXFP4.gguf` already builds `GpuModel` (13d). The server switches to Harmony output when the vocab has `<|channel|>`.
- **EOG:** `<|return|>` (the GGUF EOS) and `<|call|>` added. `<|end|>` is not EOG, since it ends every Harmony message.
- **Harmony parser (`src/harmony.rs`, after llama.cpp common/parsers/gpt-oss.cpp):**
  - analysis → reasoning; final and commentary preambles → content; `to=functions.NAME` (in the channel or the role header) → one tool call, with the raw JSON body as arguments.
  - A stray `<|channel|>commentary` prefix is skipped (the last channel wins). Calls without request tools are unsolicited and dropped, as in llama.cpp.
  - Raw prompts (`/completion`, `/v1/completions`, Ollama raw/FIM) are not parsed.
- **Reasoning out (`Out::Reasoning`), laid out as in llama-server server-task.cpp:**
  - chat `reasoning_content` deltas and message field;
  - Anthropic `thinking` block at index 0 with an empty `signature_delta` before the stops;
  - Responses `reasoning` item (`rs_…`, `reasoning_text.delta`; no status when streamed, `completed` in the body), which shifts the message and call output indices;
  - Ollama `thinking`.
- **Reasoning in:**
  - Anthropic `thinking` blocks and Responses `reasoning` items become `reasoning_content`;
  - for gpt-oss, `reasoning_content` becomes the template's `thinking`, and content is dropped when there are tool calls (common_chat_params_init_gpt_oss).
- **Gates:**
  - `harmony::tests` (3 unit tests). Mutations (no `<|end|>` hold-back, first channel instead of last, calls kept without tools) are all red.
  - `phase5_golden::api_matches_oracle_gptoss`: tests/golden/api_gpt-oss.json from tools/capture_api_gptoss.py, captured from the llama.cpp `e6443bd` oracle (GPU, `-ot exps=CPU`, fenced at 18G).
    - 9 cases (chat/messages/responses × text, tool call, tool result with reasoning in the history), streamed and full.
    - All match: prompt tokens 79/132/189, stop/finish, calls `get_weather {"city":"Paris"}`, reasoning presence and keys, and streamed event shapes. `count_tokens` 132 also matches.
    - Mutations (all red): no `thinking` mapping (6 mismatches), Anthropic thinking input dropped (2), Responses reasoning input dropped (2; the merge branch is not covered by the golden), `<|call|>` not EOG, chat reasoning streamed as content (2).
  - Full regression, all except 11b (not run, by the user's rule): lib 33, phase1 4, phase2 2, phase3 4, phase4 2, phase5 2, phase6 5, phase7 2, phase8 1, phase11 2, 11c 2, 11d 1, phase12 3, phase13 6. All pass and clippy is clean.
- **Smoke tests (`eomoe serve --model gpt-oss`, ctx 32768):**
  - Claude Code, `-p "<abs path>/calc.py has a bug in add…" --tools "Read,Edit"`: Read, then Edit (`a - b` → `a + b`), then DONE, in 40 s (4 requests, 2.2–3.6K prompt, 25 tok/s decode).
  - Codex `exec --sandbox workspace-write`: it fixed `mul`, re-read the file, then DONE, in 1m11s (7 requests, 5.2–6.2K prompt).
  - Failed with the stock tool set (14–16K prompt, relative path): the plumbing worked (calls parsed, results returned), but gpt-oss-20b guessed a wrong path for Read and then called a `Search` tool that was never offered, until Claude Code compacted. This is a model-quality limit, not a server bug, so drive it with a trimmed `--tools`.
- **Not done (ponytail):** `reasoning_effort` (the template default is medium); the Harmony tool-call grammar (greedy output matched without it); whitespace-only reasoning is not discarded.

### Phase 14a/14b: GLM-4.7-Flash tokenizer and chat template (2026-11-27), gated

Model: `ggml-org/GLM-4.7-Flash-GGUF` Q4_K (18.2 GB, arch `deepseek2`, 47 layers, MLA q_lora 768 / kv_lora 512 / rope 64, 64 experts top-4 + 1 shared, a dense first layer). It lives at ~/.local/share/eomoe/models/GLM-4.7-Flash-Q4_K.gguf (`EOMOE_TEST_GLM` overrides).
- **14a tokenizer:** `tokenizer.ggml.pre` = glm4.
  - The regex comes from the model's tokenizer.json, and `ignore_merges` is ported: a pre-token that is itself in the vocab becomes one token, as llama-vocab.cpp does for glm4.
  - Gate: `phase14_glm::tokenize_matches_hf` vs HF `tokenizers` (tests/golden/tokenize_glm.json, 31 cases), plus a decode round trip.
  - Mutation: ignore_merges off → red. The golden gained CJK words where BPE without ignore_merges splits differently (` 参考` → 3 tokens, not 1), found by toggling the flag in HF over the vocab.
- **14b chat template:**
  - minijinja gained the Python str methods `strip`/`lstrip`/`rstrip` (optional chars) and `split` (sep or whitespace).
  - `tojson` accepts `ensure_ascii`, which is ignored.
  - `chat_template_kwargs` from the request become template variables, as in llama-server (GLM's `enable_thinking`).
  - Gate: `phase14_glm::template_matches_jinja` vs jinja2 (tests/golden/template_deepseek2.json, 10 cases): tools, tool calls and results, `reasoning_content`, inline `<think>` in content, and `enable_thinking: false`.
  - Mutations: strip, split and the kwargs passthrough are red. rstrip and lstrip survive, because this template always `.strip()`s afterwards, so they cannot show in its output.
  - The qwen3 and gpt-oss tokenizer and template gates still pass, and clippy is clean.

### Phase 14c: GLM-4.7-Flash CPU reference (2026-11-27), gated

- **src/glm.rs `CpuModel`:** follows llama.cpp src/models/deepseek2.cpp.
  - MLA is absorbed: the per-position cache row is `[norm(kv_a[..512]) | rope(kv_a[512..])]`, 576 floats rounded to f16. q_nope is absorbed through `attn_k_b`, and the output comes back through `attn_v_b`.
  - RoPE is NORM (adjacent pairs) with base 1e6. kq_scale is 1/sqrt(256).
  - Layer 0 is a dense SwiGLU. The other layers use 64 routed experts with a sigmoid gate: the top 4 are chosen on sigmoid + exp_probs_b, and the plain sigmoids are renormalised ×1.8. A shared expert is added on top.
  - It refuses any shape it was not built for (gating func other than sigmoid, expert groups, YaRN).
- **Oracle:** the user OK'd two fenced llama-server runs (`-ngl 99 -ot exps=CPU`, MemoryMax 18G).
  - Run 1 captured tests/golden/greedy_glm.json with tools/capture_greedy_gptoss.py (generalised with a name argument). Chat cases use `enable_thinking: false`.
  - Its control replay gave 0/102 top-1 flips and mean |dlogprob| 0.0934.
  - Run 2 replayed with `-fa off` to measure llama.cpp's own floor: 1/102 top-1 flips at a golden margin of 0.061, mean 0.1795.
- **GATE REVISION (flagged):** GLM gets its own `GLM_TIE = 0.1` and `GLM_FLOOR_MEAN = 0.1795` (tests/common/mod.rs), measured as for gpt-oss.
- **Gate `phase14_glm::cpu_reference_matches_oracle`:** PASS. 0/102 top-1 failures; mean |dlogprob| 0.1177, under the 0.1795 floor; max 1.45.
  - `cpu_reference_answers` also passes: "Paris", and "4817" from a 231-token prompt.
- **Mutations, all red:**

  | Mutation | Top-1 failures | Mean \|dlogprob\| |
  |---|---|---|
  | No exp_probs_b in selection | 3 | 0.33 |
  | No ×1.8 scale | 20 | 1.75 |
  | No kv_a_norm | 95 | 7.66 |
  | kq_scale 1/sqrt(576) | 10 | 0.86 |
  | NEOX rope | 53 | 2.81 |

### Phase 14d: GLM-4.7-Flash on the CUDA engine (2026-11-27), gated

- **Shared weight map (src/qwen3moe.rs):** `Weights` now loads deepseek2 too.
  - `Layer.attn` is `Attn::Gqa` or `Attn::Mla`.
  - Routed experts live in `Option<Moe>` (None for the dense layer 0), which also carries GLM's `sel_bias` (exp_probs_b) and shared expert.
  - `Weights::route(li, ..)` dispatches to softmax, oai or the new shared `route_sigmoid`. src/glm.rs uses it as well.
- **Kernels (kernels/eomoe.cu):**
  - `attention_mla`: one warp per head. All heads read one staged tile of 576-wide f16 latent rows, and V is the first 512 of each row. Decode splits the positions across blocks (split-K).
  - `attn_merge` is generalised to any head width (block = hd).
  - Also new: `mm_heads_{q4_k,q6_k,q8_0}` (per-head k_b / v_b), `rope_norm` (NORM pairs, strided, in place or copying) and `rmsnorm_strided`.
- **src/gpu.rs:**
  - `DAttn::Mla` with `mla_attention`. The latent cache is one array per layer (`kv_arrays`), and `kv_save`/`kv_restore` follow it.
  - The dense FFN and the shared expert run as one concatenated [gate | up] matrix plus down, straight into the residual.
  - Dense layers skip routing and the MoE pass, and are excluded from expert groups and the pin.
  - Decode prediction runs only when the next layer has a router.
- **Gates (5 cases, 102 steps, GLM_TIE 0.1 / GLM_FLOOR_MEAN 0.1795 from 14c):**
  - `phase14_glm::gpu_matches_oracle` (prefill): PASS. 0/102 top-1 failures, mean |dlogprob| 0.1178, max 1.40, 22.6 s.
  - `gpu_decode_matches_oracle` (token by token, 1000 MiB cache): PASS. 0/102, mean 0.1189, max 1.94. Split-K and merge run on case 4.
- **Mutations, all red:**

  | Mutation | Gate | Top-1 failures | Mean \|dlogprob\| |
  |---|---|---|---|
  | kq_scale 1/sqrt(576) | prefill | 10 | 0.86 |
  | No shared expert | prefill | 96 | 7.55 |
  | rope_norm position per head wrong (r % rows_per_tok) | prefill | 87 | 5.87 |
  | No kv_a_norm | prefill | 95 | 7.66 |
  | attn_merge drops the last split | decode | 13 | 1.36 |

  An `attn_merge` max taken over only the first split stayed green. That is correct: the merged result does not depend on M. It is not counted as a check.
- **Regression (everything except 11b, which was not run by the user's rule):** all pass, and clippy is clean.
  - lib 33, phase1 4, phase2 2, phase3 4, phase4 2, phase5 2, phase6 5, phase7 2, phase8 1, phase11 2, 11c 2, 11d 1, phase12 3, phase13 6, phase14 6.
- **Speed (test harness, not a tuned measurement):**
  - Prefill: 165 tok/s on the 343-token case.
  - Decode: about 14.5 tok/s with a 1000 MiB expert cache.
- **ponytail notes:**
  - `attention_mla` uses CUDA cores and walks the positions one at a time per warp. A 19K prefill wants a tensor-core tile.
  - `mm_heads` handles one token per warp, with no token tiling.
  - The expert pin (~17 GB) has no MemAvailable guard beyond the existing pageable fallback when pinning fails.

### Phase 14e: GLM-4.7-Flash serving (2026-11-27), gated

- **Output parsing (`toolcall::GlmSplit`):**
  - When the prompt ends `<think>` (thinking on, the template default), output is reasoning up to `</think>`, holding back a partial `</think>`. After that it is content plus `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>…</tool_call>` blocks. `Splitter` gained a GLM mode: it starts only at `<tool_call>` and parses with `parse_glm`.
  - Argument values are typed by schema, as in llama.cpp's auto parser (`build_tool_parser_tag_tagged`): a string param keeps the raw text, anything else is JSON when it parses.
  - Server: `Split::Glm` is used when the vocab has `<arg_key>` and `<think>`. `<|user|>` and `<|observation|>` are EOG.
  - `chat_template_kwargs` flows from chat, Anthropic, Responses and raw chat into the template. Ollama `think: false` maps to `enable_thinking: false`.
- **Fix found by the gate (all models):** tools now reach the template as `{type, function: {name, description, parameters}}` only, with empty parameters → `{type: object, properties: {}}`, exactly as llama.cpp's `common_chat_tools_to_json_oaicompat` does. The Responses converter had been adding `strict`, which GLM's template dumps verbatim: +4 prompt tokens per Responses tool request.
- **Gates:**
  - `toolcall::tests::glm_streams_reasoning_then_content_and_calls`. Mutations, all red: no `</think>` hold-back; every value typed as a string; the text after `</think>` bypassing the call splitter (only red after the test fed `</think>` and a partial `<tool_` in one delta, now part of the test).
  - `phase5_golden::api_matches_oracle_glm`: tests/golden/api_glm.json from `tools/capture_api_gptoss.py URL glm` (now takes a name argument), captured from the llama.cpp `e6443bd` oracle (GPU, `-ngl 99 -ot exps=CPU -c 4096`, fenced at MemoryMax 18G, stopped before eomoe ran).
    - 10 cases: chat/messages/responses × text, tool call, tool result with reasoning in the history, plus chat with `enable_thinking: false`. Streamed and full.
    - All match: prompt tokens 17/175/209, stop/finish, calls `get_weather {"city":"Paris"}`, reasoning presence and keys (none when thinking is off), and streamed event shapes. `count_tokens` 175 also matches.
    - Mutations, all red: generation prompt `<think>` not detected (reasoning streamed as content), `chat_template_kwargs` dropped, `<|user|>`/`<|observation|>` not EOG, `strict` back in the Responses tools.
  - Full regression, everything except 11b (not run, by the user's rule): lib 34, phase1 4, phase2 2, phase3 4, phase4 2, phase5 3, phase6 5, phase7 2, phase8 1, phase11 2, 11c 2, 11d 1, phase12 3, phase13 6, phase14 6. All pass and clippy is clean.
- **Smoke tests (`eomoe serve --model GLM-4.7-Flash-Q4_K.gguf`, ctx 32768, 7316 MiB VRAM):**
  - Claude Code with the **stock tool set** and a relative path (`-p "calc.py … bug in add …" --allowedTools "Read,Edit"`): fixed `a - b` → `a + b`, then DONE, in 5m34s. 4 requests at 18.8–20.2K prompt tokens, decode about 20 tok/s. The time is almost all the first prefill: 147.7 s for 18.8K (127 tok/s).
  - Codex `exec --sandbox workspace-write`: fixed `mul`, then DONE, in 35 s. 8 requests at 2.4–3.3K prompt tokens, prefill 417 tok/s, decode 23–26 tok/s.
- **Not done (ponytail):** long-context prefill. `attention_mla` is CUDA-core and goes 417 tok/s at 2.4K but 127 tok/s at 19K, so the next GLM speed win is a tensor-core MLA tile (as `attention_tc` did for Qwen). No grammar-constrained GLM tool calls (greedy output matched without them).

## Post-14: CI, release, soak (2026-11-27)

- **`ci.yml`** (push to main, PRs, and called by release): runs in the nvidia/cuda 13.3.1 devel container on a hosted runner. Steps: `cargo clippy --all-targets -D warnings`, then `cargo test --release`. The model and GPU gates are `#[ignore]`, so hosted CI runs only the model-free tests.
- **`release.yml`** (tag `v*`; `workflow_dispatch` is a dry run that publishes nothing):
  - After CI passes, two jobs run in parallel.
  - **Binaries:**
    - Targets: linux x86_64, linux aarch64 (native arm runner), windows x86_64.
    - Kernels: `EOMOE_CUDA_ARCH=80,86,89,90,100,120`, a fatbin for every sm_80+ GPU (the mma.m16n8k16 kernels need sm_80).
    - Windows needs two workarounds. Git Bash's `link` hides MSVC's `link.exe`. VS 18's STL wants CUDA 13.4, which has no release, so the build passes `-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH`; the kernels use no STL.
    - Not built: macOS (no CUDA) and Windows-on-Arm (no NVIDIA toolkit).
  - **Image:**
    - amd64 and arm64 are built on native runners.
    - Each image is verified with `./eomoe --version` before it is pushed to GHCR as `:TAG-arch`.
    - The manifest job then creates `:TAG` and `:latest`.
  - Release: SHA256SUMS plus generated notes. A tag containing `-` makes a prerelease.
- **`soak.yml`** (Sunday 07:00 UTC, which is 02:00 EST and 03:00 during EDT; or manual):
  - Runs on the self-hosted runner `media2026-eomoe` (label `eomoe-gpu`, a user systemd service with linger). Fork PRs never reach it: there is no PR trigger, there is a repo guard, and every outside contributor needs approval.
  - `tools/soak.sh`:
    - Refuses to start if eomoe or llama-server is already running.
    - Every test binary runs fenced (MemoryMax 26G, no swap) with `--include-ignored`. Anything matching `*11b*` is skipped by name (the user's rule).
    - Then runs `tools/stress.py` for each model: Qwen3-Coder, gpt-oss and GLM, each replaying its own oracle capture.
    - stress.py runs 4 concurrent clients, half of them streamed, checks health every 30 s, and fails on any error or on RSS growth over 1 GiB.
- **Verified:**
  - CI and the release dry run are all green: run 46078768547, all three binaries and both images.
  - The CI-built linux binary links only libc/libm/libgcc.
  - A local multi-arch build passed the phase5 `api_matches_oracle` GPU gate.
  - Stress: 2 min on Qwen, 116 requests, 0 errors, RSS −215 MiB.
- **Speed vs llama.cpp (measured from the oracle golden timings on this box, same models; not gated):**

  | Model | eomoe decode | llama.cpp decode |
  |---|---|---|
  | gpt-oss | 25 tok/s | 20.2 tok/s |
  | GLM | 23–26 tok/s | 22.0 tok/s |

  The oracle prefill timings cannot be compared, because those prompts were tiny.
- **Review debt** (code review since e74857c, judgement calls, none of them bugs):
  - The glm.rs and gptoss.rs CPU-reference loaders are deliberately independent of `qwen3moe::Weights::load`. Each one is the plain reference its gate checks, so sharing a loader would couple three oracles.
  - Smells left as they are:
    - The architecture is carried as `oai: bool` plus `glm: Option`, which repeats the switch.
    - Three layers sit between the splitter and the server: Splitter, GlmSplit, Split.
    - The `(golden, TIE, FLOOR)` gate tuple is a data clump.
    - `set_n_expert_used` re-reads the GGUF.

    Refactor these when a fourth architecture lands (Phase 15).

### Phase 15a/15b: Qwen3.6-35B-A3B tokenizer and chat template (2026-11-27), gated

- Model: `ggml-org/Qwen3.6-35B-A3B-GGUF` Q4_K_M (20.5 GB). Arch `qwen35moe`: 40 layers, 30 Gated DeltaNet and every 4th one gated full attention (head size 256, 16 q / 2 kv heads, 64 of 256 dims roped as IMRoPE [11,11,10,0]). MoE has 256 experts with top-8 (ff 512), plus a sigmoid-gated shared expert. Vocab 248320.
- **15a tokenizer:**
  - `qwen35` pre-tokenizer: llama.cpp's regex, which is qwen2 with `\p{M}` kept inside words.
  - The GGUF sets `normalizer.nfc`, but llama.cpp applies NFC only in its UGM tokenizer, never BPE, so eomoe does not either. Every golden text is already NFC.
  - Gate `phase15_qwen36::tokenize_matches_hf`: 33 cases from HF `tokenizers` on `Qwen/Qwen3.6-35B-A3B` (`tools/capture_tokenize_hf.py`). The new cases cover Devanagari, Bengali, Tamil and Thai combining marks, and the Qwen3.6 tool-call/think tokens.
  - Mutation: the qwen2 regex in place of qwen35 → red (the combining-marks case).
- **15b template:**
  - The template uses transformers' `{% generation %}…{% endgeneration %}`, which llama.cpp parses as a no-op. eomoe and the capture tool both rewrite it to `{% if true %}…{% endif %}` before compiling, so whitespace control is unchanged.
  - The template also calls `str.startswith`/`endswith`, now in the unknown-method callback.
  - Gate `template_matches_jinja`: 10 cases from jinja2 (`tools/capture_template_jinja.py`) → tests/golden/template_qwen35moe.json.
  - Without the rewrite the template fails to parse (red). Deleting the tags instead of mapping them is an equivalent mutant on this template (all of them sit between `{%-` blocks), so it is not counted.
  - Regression: phase1, phase13 and phase14 tokenizer and template gates, lib 33. Clippy is clean.

### Phase 15c: Qwen3.6-35B-A3B CPU reference (2026-11-27), gated

- **src/qwen35.rs `CpuModel`:** follows llama.cpp src/models/qwen35moe.cpp and the CPU `gated_delta_net` op.
  - **Gated DeltaNet (30 layers):**
    - `attn_qkv`, then a causal depthwise conv (kernel 4, over 3 cached raw rows), then SiLU.
    - q and k: 16 heads × 128, L2-normed (`x / sqrt(Σx² + eps)`). v: 32 heads × 128.
    - v head h reads k head `h % 16` (tiled, as in the op's `iv1 % neq1`).
    - Per-head state S, 128×128, stored transposed as in ggml: S *= exp(g); S += k ⊗ β(v − Sᵀk); out = Sᵀq/√128.
    - g = softplus(alpha + dt_bias) · ssm_a, and β = sigmoid(beta).
    - Then rmsnorm(out, ssm_norm) · silu(z) per head, and `ssm_out`.
  - **Gated attention (10 layers):**
    - `attn_q` gives [q | gate] per head (256 each), with q/k RMS norms.
    - NEOX rope on the first 64 dims. IMRoPE with equal text positions reduces to this. Base 1e7.
    - GQA 16/2. The output is multiplied by sigmoid(gate). KV is rounded to f16.
  - **FFN:** softmax top-8 of 256, normalised (`qwen3moe::route`), plus a shared expert × sigmoid(`ffn_gate_inp_shexp`·x).
  - **Recurrent state limit:** `truncate` can only keep everything or reset; rewinding needs state checkpoints (15e).
- **GATE REVISION (flagged):** `QWEN36_TIE = 0.1` and `QWEN36_FLOOR_MEAN = 0.1453` (tests/common/mod.rs).
  - Measured as for GLM, from the fenced oracle runs recorded in 15a/b's capture: control replay 0/109 flips with mean 0.0576; `-fa off` floor 0/109 flips with mean 0.1453.
  - Golden: tests/golden/greedy_qwen36.json, 5 cases and 109 steps (Paris, linked list, 240-token vault-code recall, fib, 312-token raw filler).
- **Gate `phase15_qwen36::cpu_reference_matches_oracle`:** PASS. 0/109 top-1 failures; mean |dlogprob| 0.1336, under the 0.1453 floor; max 1.17.
  - `cpu_reference_answers` also passes: "Paris", and "The vault code is **4817**." from a 240-token prompt.
- **Gate fix:** `greedy_floor_gate` now fails on a NaN mean. `NaN > floor` was false, so a NaN slipped through. Found by the L2-norm mutation below, which went red only through its top-1 failures.
- **Mutations, all red:**

  | Mutation | Top-1 failures | Mean \|dlogprob\| |
  |---|---|---|
  | k head `h / 2` (repeat) instead of `h % 16` (tiled) | 103 | 5.23 |
  | No state decay exp(g) | 59 | 4.54 |
  | No silu(z) output gate | 109 | 9.47 |
  | No shared-expert sigmoid gate | 108 | 6.23 |
  | No attention sigmoid(gate) | 109 | 10.59 |
  | No L2 norm on q/k | 100 | NaN |
  | No rope on k | 1 | 0.51 |

- Phase 15 suite: 4/4 pass (tokenizer, template, answers, oracle). Clippy is clean.

### Phase 15d: Qwen3.6-35B-A3B CUDA engine (2026-11-27), gated

- **src/gpu.rs, using the same `qwen3moe::Weights`:** the loader now reads `qwen35moe`: `recurrent_layers` from the GGUF bool array, `Attn::Delta`, `Moe.shared_gate`, and `n_rot`.
- **New kernels (kernels/eomoe.cu), with src/qwen35.rs as the reference:**
  - `ssm_conv`: causal conv + SiLU, one thread per channel, conv state carried per sequence.
  - `l2norm_heads`.
  - `gated_delta`: one warp per state row; lanes own 4 floats; the row stays in registers across the tokens of a piece; no block syncs.
  - `gated_rmsnorm`: rmsnorm · silu(z).
  - `split_qgate` and `mul_sigmoid`: gated attention.
  - `sigmoid_dot_scale`: the shared-expert gate.
  - `qk_norm_rope` takes `n_rot`, for partial rope (64 of 256).
- **DeltaNet layer:**
  - One Q8_0 matmul over [qkv | z | beta | alpha] (12352 rows) into a `dn` scratch buffer.
  - Per piece: conv, then L2, then the delta rule, against that sequence's state.
  - Then gated rmsnorm over the ubatch, then `ssm_out`.
- **Gated attention:** head_dim 256 goes through the generic `attention` kernel.
- **KV and recurrent state:**
  - KV is allocated for the 10 attention layers only, indexed through `slot[li]`.
  - Recurrent state is per sequence: 30 × (3·8192 + 32·128·128) floats, about 63 MB. It is allocated lazily in `set_n_seq` and zeroed by `truncate(0)`.
  - The RAM cache (11b) skips recurrent models, because `kv_save` returns empty. State snapshots come with 15e.
  - A direct `truncate` to 0 < n < n_past panics. The batcher uses `rewind`, which recomputes instead (below).
- **Gates (tests/phase15_qwen36.rs):** all PASS against greedy_qwen36.json, same tie and floor as 15c.

  | Test | Top-1 failures | Mean \|dlogprob\| | Max |
  |---|---|---|---|
  | `gpu_matches_oracle` (prefill, 4096 ctx) | 0/109 | 0.1337 | 1.17 |
  | `gpu_decode_matches_oracle` (token by token, 1000 MB expert cache) | 0/109 | 0.1341 | |

  - `gpu_answers` also passes: "Paris", and "The vault code is **4817**." Prefill runs at 101–121 tok/s on the 250/343-token cases.
- **Mutations, all red:**

  | Mutation | Test | Top-1 failures | Mean |
  |---|---|---|---|
  | k head `h / 2` | prefill | 102 | 5.35 |
  | No decay | prefill | 59 | 4.55 |
  | No conv SiLU | prefill | 107 | 6.04 |
  | Conv state not written back | decode | 96 | 4.90 |
  | No L2 norm | prefill | 99 | NaN |
  | No silu(z) | prefill | 109 | 9.47 |
  | No shared-expert gate | prefill | 108 | 6.23 |
  | No attention gate | prefill | 109 | 10.59 |
  | Full rope instead of `n_rot` | prefill | 5 | 1.00 |
  | No state reset on `truncate(0)` | prefill | 12 | 2.55 |
- **Regression (every suite except 11b, which is not run by the user's rule):** all pass. lib 34, phase1 4, phase2 2, phase3 4, phase4 2, phase5 3, phase6 5, phase7 2, phase8 1, phase11 2, 11c 2, 11d 1, phase12 3, phase13 6, phase14 6. Phase 15: 7/7. Clippy is clean.
- **Serving safety (pre-15e):** `Engine::rewind(n) -> kept`. Its default truncates and returns n; recurrent engines (`GpuModel` with DeltaNet layers, and `qwen35::CpuModel`) keep all positions or none. The batcher now truncates its token cache to what the engine kept, so a prefix the state can't rewind to is recomputed instead of panicking. `kv_save` returns empty (unsupported) for recurrent models, so the RAM cache skips them.
  - Gate: `generate::tests::recurrent_engine_recomputes_a_prompt_it_cannot_rewind_to`. It was red before the batcher used the return value (n_cached 4, expected 0).
  - Smoke test, `eomoe serve` with Qwen3.6, ctx 8192, 7319 MiB: answered Paris, then a follow-up turn (Berlin, prefix recomputed), then a fresh prompt (Rome). No panic.
  - **Known limitation until 15e:** reasoning comes back inside `content`, ending `</think>`. Still to come: the `<think>` split, state checkpoints for prefix reuse, and an oracle API golden.
- **Concurrent sequences gate:** `gpu_batched_sequences_match_solo` decodes 3 sequences together through `decode_batch`, with state allocated lazily by `set_n_seq`. It matches each sequence run alone: 3 × 10 steps, top-1 identical, max |dlogprob| 0.0014.
  - Mutation: every sequence sharing `rs[0]`. Red.
- **Review since 5b9f689 (two axes, before v0.1.5):**
  - No bugs found. Fixed: the guard's error now names the rope dims; one pointer had two names (`inp`/`dnr`); the truncate ceiling is marked `ponytail:`; two stale STATUS claims about kv_save and rewind are corrected.
  - Left as judgement calls:
    - the `dn` scratch layout is spelled out in 3 places;
    - `rewind`'s 3-line body is repeated in two engines;
    - the lazily allocated recurrent state isn't counted in the VRAM log;
    - M-RoPE is reduced to NEOX, which is valid for text only, with equal positions.
- **Docker (eca40b4):** a plain `docker run IMAGE ./eomoe serve --model …` works without a mounted config. The image no longer sets `EOMOE_CONFIG_PATH`; docker-compose.yml sets it for its mounted ./config. release.yml checks the image with `./eomoe about` as well as `--version`. Verified against the GHCR image serving GLM-4.7-Flash on the GPU.
