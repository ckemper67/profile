//! Apple Silicon GPU/memory telemetry via `macmon` - sudoless, no subprocess.
//!
//! Apple Silicon has unified memory (no dedicated VRAM pool shared between
//! CPU/GPU/OS), so `vram_*_mb` here carries system-wide memory usage, not a
//! GPU-exclusive allocation - see `ARCHITECTURE.md` for how the physics
//! baseline accounts for that. There is also no separate "memory controller
//! utilization" signal the way NVML reports it, so `mem_util_pct` stays
//! `None` rather than guessing from RAM fill.

use std::time::{Duration, SystemTime};

use anyhow::Result;
use macmon::Sampler;

use super::super::GpuRawMetrics;
use super::super::sampling::{SAMPLE_INTERVAL, sample_count_for};
use super::polling::{GpuPoll, PollAggregateState};

const BYTES_PER_MB: u64 = 1024 * 1024;

fn poll_from_metrics(m: &macmon::Metrics) -> GpuPoll {
    GpuPoll {
        util_gpu: Some((m.gpu_scaled_ratio.clamp(0.0, 1.0) * 100.0).round() as u32),
        util_mem: None,
        power_watts: Some(f64::from(m.gpu_power)),
        vram_used_mb: Some(m.memory.ram_usage / BYTES_PER_MB),
        vram_total_mb: Some(m.memory.ram_total / BYTES_PER_MB),
        temperature_c: Some(f64::from(m.temp.gpu_temp_avg)).filter(|t| *t > 0.0),
        sm_clock_mhz: Some(m.gpu_freq_mhz),
    }
}

/// Single-shot scan for gpu_assignment. Not exercised on the llama.cpp path today -
/// `cli::gpu_assignment` short-circuits to a fixed single-GPU assignment before this
/// would be called - kept for API completeness with the NVIDIA/AMD backends.
pub(super) fn scan_host_gpus() -> Option<Vec<super::GpuScanEntry>> {
    let mut sampler = Sampler::new().ok()?;
    let soc = sampler.get_soc_info().clone();
    let m = sampler.get_metrics(100).ok()?;
    Some(vec![super::GpuScanEntry {
        idx: 0,
        name: soc.chip_name,
        vram_used_mb: m.memory.ram_usage / BYTES_PER_MB,
        vram_total_mb: m.memory.ram_total / BYTES_PER_MB,
        pids: vec![],
    }])
}

/// No FP8 compiler concept on the Metal/GGML path (llama.cpp selects KV cache
/// quantization via `--cache-type-k`/`-v`, not a toolchain probe).
pub(super) fn fp8_compiler_available() -> bool {
    false
}

/// Returns `(metrics, observed_at, host_count)`. `host_count` is always `Some(1)` when the
/// sampler initializes - Apple Silicon exposes one GPU regardless of core count.
pub(super) fn collect(
    window: Duration,
    _explicit_indices: Option<&[u32]>,
) -> Result<(Vec<GpuRawMetrics>, SystemTime, Option<u32>)> {
    let Ok(mut sampler) = Sampler::new() else {
        return Ok((vec![], SystemTime::now(), None));
    };
    let soc = sampler.get_soc_info().clone();

    let sample_count = sample_count_for(window);
    let mut state = PollAggregateState::default();
    for _ in 0..sample_count {
        // get_metrics blocks for the requested interval while it integrates one IOReport
        // delta, so this loop's own wall time already matches the window - no extra sleep
        // (unlike NVML/AMD polling, which sample instantaneously and sleep between ticks).
        if let Ok(m) = sampler.get_metrics(SAMPLE_INTERVAL.as_millis() as u32) {
            state.update(&poll_from_metrics(&m));
        }
    }

    let agg = state.finish();

    let metrics = agg.into_gpu_raw_metrics(
        Some(soc.chip_name),
        Some(0),
        None,
        None,
        None, // no power-limit signal on the Metal/GGML path
    );

    Ok((vec![metrics], SystemTime::now(), Some(1)))
}
