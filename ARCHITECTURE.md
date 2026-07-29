# Architecture

Structure of the `profile` binary: data flow, module boundaries, key types.

How to change the code, and the merge gate (`fmt`, `clippy`, `audit`, `deny`, `test`, plus OSV, Semgrep, and Socket in CI): [CONTRIBUTING.md](CONTRIBUTING.md).

What the tool does and why: [README](README.md).

Single Rust binary. `profile diagnose` runs an interactive closed loop: collect, analyze, recommend, wait for the operator to apply the fix, re-collect, compute delta, repeat.

## Data flow

```text
CLI flags
  └─ cli (clap) → cli::diagnose::execute
       │
       ├─ profiler::run_diagnose
       │    collectors::          all I/O; 250ms cadence inside each collection window (sampling.rs)
       │      ├─ gpu/             NVML (nvidia.rs) or amdgpu (amd.rs), polled via polling.rs
       │      ├─ vllm.rs          HTTP /metrics scrape
       │      └─ host_memory.rs   host RAM + container cgroup cap, for kv-offload sizing
       │           → one RawSnapshot per collection window (2s or 10s; many 250ms samples inside)
       │    aggregate_windows → run-level RawSnapshot (DiagnoseResult.snapshot)
       │    config::build_config(snapshot + CLI; best-effort GET /v1/models and /info)
       │    StaticContext from catalogs + config
       │
       ├─ engine::build_report_for_diagnose(windows, aggregate AnalysisInput)
       │    wraps rules::build_report_for_windows (eval.rs); may add post-DAG recommendations (e.g. MU)
       │    baseline/ + limiter.rs feed the Report
       │
       ├─ output::stdout  (initial report)
       │
       └─ profiler::loop_runner  only if any_evaluable, not all_idle, and
            n_eval >= ENGINE_MIN_PERSISTENT_WINDOWS
            wait → re-run_diagnose → delta → drift → loop
```

## Source tree

```text
src/
├── main.rs, lib.rs
├── cli/            flags, preflight, GPU assignment
├── collectors/     all I/O: vllm.rs (/metrics), gpu/ (NVML, amdgpu), config.rs, host_memory.rs
├── context/        catalogs (GPU, model, prices); StaticContext, RuntimeWindow
├── engine/
│   ├── baseline/   roofline physics: math.rs, roofline.rs
│   ├── rules/      r1..r7, eval.rs, format.rs, suppression (mod.rs)
│   ├── limiter.rs  no-issue verdicts
│   └── mod.rs      report assembly, post-DAG additions
├── profiler/       windows, aggregate, loop_runner, delta, drift
└── output/         stdout rendering
```

## Module responsibilities

| Module            | Owns                                                                    | Never touches          |
| ----------------- | ----------------------------------------------------------------------- | ---------------------- |
| `src/collectors/` | All I/O: GPU, vLLM metrics, config, host memory; the scrape cadence     | No reasoning, no rules |
| `src/context/`    | StaticContext + RuntimeWindow from raw collector output; the catalogs   | No I/O, no rules       |
| `src/engine/`     | All reasoning: baseline, rules, limiter                                 | No I/O, no interaction |
| `src/profiler/`   | Orchestration: loop, state, delta, drift detection; drives collection   | No rule or collector logic |
| `src/cli/`        | Arg parsing, GPU assignment, dispatch to profiler                       | No reasoning, no rules |
| `src/output/`     | Emit bus: stdout                                                        | No reasoning           |

`engine/` never imports from `cli/`. Engine is deterministic; CLI is interactive.

## Key types

| Type | Lives in | Job |
| --- | --- | --- |
| `StaticContext` | `context/types.rs` | Model, GPU, and vLLM config, resolved once per `run_diagnose` from the catalogs. Re-baselined on config drift. |
| `RuntimeWindow` | `context/types.rs` | Wraps one per-window `RawSnapshot`; the snapshot carries its own timestamp. |
| `AnalysisInput<'a>` | `context/types.rs` | `{ ctx: &StaticContext, window: &RuntimeWindow }`. The engine's input; no copies in the hot loop. |
| `RawSnapshot` | `collectors/types.rs` | One collection window's result (250ms samples inside). One per window, plus a run-level aggregate (`DiagnoseResult.snapshot`). |
| `PhysicsBaseline` | `engine/baseline/roofline.rs` | The physics: decode and prefill ceilings, efficiency and headroom, weight footprint with dtype provenance, spec-guard fields. No causality. |
| `Recommendation` | `engine/rules/` | One fired rule: name, DAG layer (2-6), impact (1-5), confidence, display lines, `terminal` (no server-local knob left). |
| `LimiterVerdict` | `engine/limiter.rs` | The no-issue path: names the boundary capping a healthy server, or declines (ceiling unknown, waiting unread, speculation suspected) rather than guess. |
| `Report` | `engine/mod.rs` | Recommendations plus baseline and skip counts. Built by `build_report_for_diagnose`, which wraps `rules::build_report_for_windows` and may append post-DAG recommendations. Input to stdout. |
| `window_is_evaluable`, `window_is_idle` | `collectors/types.rs` | Shared gates. Evaluable: positive duration and the endpoint answered. Idle: evaluable with no meaningful traffic. Rules skip idle windows; idle is valid telemetry, not a failure. |

Semantics that bite:

- `PhysicsBaseline` is physics only; its struct doc comments are canonical for the full field list. `spec_suspected` / `spec_window_counts` clear efficiency claims when measured decode beats the one-token-per-read roof.
- `Recommendation` ranking: impact x confidence within the winning DAG layer; the layer filter and suppression table enforce one signal per root cause. Soft field (`limiter::soft_field`) skips the R6→R1 ME row so Under-batching can win first fire; Prefill/Prefix stay in `suppressed_recs` for the remeasure reveal. The bound path keeps that ME row and its terminals.

## Where things go

- **A new rule:** `src/engine/rules/rN_name.rs`, wired into `rules::build_report_for_windows` in `src/engine/rules/eval.rs`. Follow the checklist in [CONTRIBUTING.md](CONTRIBUTING.md).
- **A new metric source:** `src/collectors/`. I/O only; no reasoning.
- **A new GPU or model:** `src/context/gpu_catalog.rs`, `src/context/model_catalog.rs`, `src/context/gpu_prices.json`.
- **Closed-loop behavior:** `src/profiler/` (loop_runner, delta, drift, state). Orchestrates collection; implements no collector logic and no rules.
- **Config enrichment:** `src/collectors/config.rs` (`build_config`). Runs after collection inside `run_diagnose`; snapshot fields + CLI, then best-effort `/v1/models`, `/info`, and `/server_info`. Scheduler knobs (`enable_chunked_prefill`, `max_num_batched_tokens`) are filled from those JSON/text endpoints when Prometheus omits them (modern vLLM keeps them on `SchedulerConfig`, not `cache_config_info`).
- **Output formatting:** `src/output/stdout.rs`. Convention: `~` marks values derived from estimated ceilings; measured values carry no tilde.
- **Rule thresholds:** named constants at the top of each rule file in `src/engine/rules/`. Semantics and edge cases are in the [rules documentation](https://jungledesh.github.io/profile/docs.html#rules).

## Fork addition: llama.cpp / Apple Silicon support

This fork (`llamacpp-macos-support`) adds a second inference-server backend and a
second GPU-telemetry backend, selected by `--engine {vllm,llama-cpp}` (default
`vllm`, preserving upstream behavior when unset).

- **`cli/mod.rs`**: `Engine::{Vllm, LlamaCpp}` (upstream already carries the enum
  and `--engine`/`--url` flags as unwired scaffolding; this fork is what actually
  consumes them) selects the collector and config-builder below.
- **`cli/gpu_assignment.rs`**: short-circuits to a fixed `{tp: 1, indices: [0]}`
  for `--engine llama-cpp` -- Apple Silicon is single-GPU with no tensor-parallel
  launch scope to detect.
- **`collectors/llamacpp.rs`**: scrapes `llama-server`'s `/metrics` (throughput
  counters/gauges, plus TTFT/TPOT histograms and a KV-cache-usage gauge added
  upstream in `llama.cpp` itself specifically to support this tool -- see
  "Upstream" below) and `/props` (`total_slots` -> `max_num_seqs`, `n_ctx` ->
  `max_model_len`, model info). Reuses `vllm.rs`'s generic Prometheus
  histogram/counter math (`pub(crate)`) rather than duplicating it. Fields with
  no llama.cpp source (prefill/queue latency split, prompt-length distribution,
  preemption/swap, CPU KV offload, prefix-cache hit rate) stay `None`, same as
  any other missing gauge.
- **`collectors/config.rs`**: `build_llamacpp_config` mirrors `build_config`,
  sourced from `/props` instead of `/v1/models`+`/info`. Reuses `VllmConfig` as-is
  -- downstream `engine`/`rules` code only cares about resolved values, not which
  server produced them. `model_ftype` (the GGUF quantization string, e.g. "Q4_K -
  Medium") is stored in both `dtype` and `vllm_reported_quantization` since
  llama.cpp conflates precision and quantization scheme in one field.
- **`collectors/gpu/apple.rs`**: `#[cfg(target_os = "macos")]`-gated telemetry via
  `macmon::Sampler` (sudoless IOReport read, no subprocess) for GPU util/power/
  clock/temp; system-wide unified memory stands in for VRAM (Apple Silicon has no
  dedicated VRAM pool). Selected entirely by `#[cfg(target_os = "macos")]` in
  `gpu/mod.rs`, not a runtime fallback -- `libamdgpu_top` has no macOS backend and
  panics if reached there. `macmon` is a `target.'cfg(target_os =
  "macos")'.dependencies` entry, not unconditional like `nvml-wrapper`/
  `libamdgpu_top`, to avoid shipping either vendor crate's non-functional-but-
  compiles-fine-on-macOS runtime path into the collection code path.
- **`context/gpu_catalog.rs`**: Apple Silicon chip entries (M1-M4 families),
  matched on `macmon`'s `chip_name` (e.g. "Apple M3 Pro") -- documented as
  approximate/unified-memory, same caveat as the NVIDIA GB10 entry.
- **`engine/baseline/roofline.rs`**: KV cache headroom branches on GPU type.
  vLLM's `vram_gb * gpu_memory_utilization - buffer - weight_gb/tp` assumes a
  dedicated VRAM pool sized by policy fraction; for unified-memory GPUs (detected
  by GPU name substring "apple", `is_unified_memory_gpu`)
  `math::kv_headroom_gb_unified_memory` instead nets live system-wide memory
  usage against total unified memory, since there's no dedicated pool or
  utilization fraction to reserve against.

### Upstream

The llama.cpp collector depends on three `/metrics` additions upstreamed into
`llama.cpp`'s own `tools/server/server-context.cpp` alongside this project:
`llamacpp:ttft_seconds` / `llamacpp:tpot_seconds` histograms (in the same
cumulative-bucket Prometheus shape vLLM uses) and a
`llamacpp:kv_cache_usage_ratio` gauge. Without these, llama.cpp's stock
`/metrics` has only throughput counters and queue-depth gauges -- no latency
distribution, no KV-cache-usage signal, which several rules and the limiter
depend on.
