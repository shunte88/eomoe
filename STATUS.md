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
| 3. CUDA engine | Matches phase 2 / `greedy.json` | not started |
| 4. Expert cache | Hit rate on `ref_trace.mct` ≥ oracle; decode tok/s ≥ oracle | not started |
| 5. Server + one-stop CLI | Reference `check` passes; real Claude Code + Codex tool sessions | not started |
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
