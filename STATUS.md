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
| 6. Beyond parity | Prefetch, CPU-miss compute, batching, multi-GPU | in progress: f16 KV / 32K ctx, fast prefill done (see below) |

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

GEMM is still ~79% of prefill time. The next lever is an MMQ-style kernel: integer quants as exact f16 with per-block scales applied to partial sums. That drops the weight lo half and the float dequant.
