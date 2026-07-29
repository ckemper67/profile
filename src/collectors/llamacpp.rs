//! llama-server `/metrics` + `/props` scrape.
//!
//! `llama-server`'s Prometheus surface is smaller than vLLM's: no per-request
//! histogram was available upstream until this codebase's counterpart commit
//! added `llamacpp:ttft_seconds` / `llamacpp:tpot_seconds` / `llamacpp:kv_cache_usage_ratio`
//! to `tools/server/server-context.cpp` (see ARCHITECTURE.md). Fields with no
//! llama.cpp source (prefill/queue latency split, prompt-length distribution,
//! preemption/swap, CPU KV offload, prefix-cache hit rate) stay `None`, same
//! as any other missing gauge - the engine and rules already null-check.

use std::time::{Duration, Instant, SystemTime};

use anyhow::{Context, Result};
use prometheus_parse::{Scrape, Value};

use super::sampling::{run_sampling_loop, sample_count_for};
use super::vllm::{
    counter_delta_per_sec, fetch_metrics_body, histogram_window_delta_buckets,
    histogram_window_mean_ms, histogram_window_p95_ms, histogram_window_p99_ms, metrics_url,
    sum_metric_samples,
};
use super::EngineRawMetrics;

const REQ_TIMEOUT: Duration = Duration::from_secs(10);

/// llama-server exposes metrics with a `llamacpp:` prefix; `prometheus-parse` only accepts
/// `\w+` in metric names, so normalize to underscores before parsing (same reason as vLLM's `:`).
fn normalize_llamacpp_prometheus_text(body: &str) -> String {
    body.replace("llamacpp:", "llamacpp_")
}

fn scrape_from_body(body: &str) -> Result<Scrape> {
    let normalized = normalize_llamacpp_prometheus_text(body);
    Scrape::parse(normalized.lines().map(|s| Ok(s.to_string())))
        .context("failed to parse Prometheus text format")
}

fn first_gauge(scrape: &Scrape, name: &str) -> Option<f64> {
    scrape
        .samples
        .iter()
        .find(|s| s.metric == name)
        .and_then(|s| match s.value {
            Value::Gauge(v) | Value::Untyped(v) => Some(v),
            _ => None,
        })
}

fn kv_cache_usage_perc_from_scrape(scrape: &Scrape) -> Option<f64> {
    first_gauge(scrape, "llamacpp_kv_cache_usage_ratio").map(|v| v * 100.0)
}

pub fn collect_llamacpp_metrics_for(
    input: &str,
    window: Duration,
) -> Result<(EngineRawMetrics, SystemTime)> {
    let client = reqwest::blocking::Client::builder()
        .timeout(REQ_TIMEOUT)
        .build()
        .context("failed to build HTTP client")?;
    let url = metrics_url(input);
    let sample_count = sample_count_for(window);
    let mut window_start: Option<Instant> = None;
    let mut first_scrape: Option<Scrape> = None;
    let mut last_scrape: Option<Scrape> = None;
    let mut kv_cache_peak_perc: Option<f64> = None;

    run_sampling_loop(sample_count, |i| {
        let body = fetch_metrics_body(&client, &url)?;
        let scrape = scrape_from_body(&body)?;
        if let Some(k) = kv_cache_usage_perc_from_scrape(&scrape).filter(|x| x.is_finite()) {
            kv_cache_peak_perc = Some(kv_cache_peak_perc.map_or(k, |p| p.max(k)));
        }
        if i == 0 {
            window_start = Some(Instant::now());
            first_scrape = Some(scrape);
        } else {
            last_scrape = Some(scrape);
        }
        Ok(())
    })?;

    let window_secs = window_start
        .map(|t| t.elapsed().as_secs_f64())
        .unwrap_or(0.0);

    let first_scrape = first_scrape.context("llama.cpp gauge window missing first scrape")?;
    let last_scrape = last_scrape.context("llama.cpp gauge window missing last scrape")?;

    let mut m = EngineRawMetrics {
        num_requests_running: first_gauge(&last_scrape, "llamacpp_requests_processing"),
        num_requests_waiting: first_gauge(&last_scrape, "llamacpp_requests_deferred"),
        kv_cache_usage_perc: kv_cache_usage_perc_from_scrape(&last_scrape),
        kv_cache_avg_perc: kv_cache_usage_perc_from_scrape(&last_scrape),
        kv_cache_peak_perc,
        ..Default::default()
    };

    m.ttft_ms = histogram_window_mean_ms(&first_scrape, &last_scrape, "llamacpp_ttft_seconds");
    if let Some(q) = histogram_window_p99_ms(&first_scrape, &last_scrape, "llamacpp_ttft_seconds")
    {
        m.ttft_p99_ms = Some(q.value);
        m.ttft_p99_clamped = q.clamped;
    }
    if let Some(q) = histogram_window_p95_ms(&first_scrape, &last_scrape, "llamacpp_ttft_seconds")
    {
        m.ttft_p95_ms = Some(q.value);
        m.ttft_p95_clamped = q.clamped;
    }
    m.ttft_p99_buckets =
        histogram_window_delta_buckets(&first_scrape, &last_scrape, "llamacpp_ttft_seconds");

    m.tpot_ms = histogram_window_mean_ms(&first_scrape, &last_scrape, "llamacpp_tpot_seconds");
    if let Some(q) = histogram_window_p99_ms(&first_scrape, &last_scrape, "llamacpp_tpot_seconds")
    {
        m.tpot_p99_ms = Some(q.value);
        m.tpot_p99_clamped = q.clamped;
    }
    if let Some(q) = histogram_window_p95_ms(&first_scrape, &last_scrape, "llamacpp_tpot_seconds")
    {
        m.tpot_p95_ms = Some(q.value);
        m.tpot_p95_clamped = q.clamped;
    }
    m.tpot_p99_buckets =
        histogram_window_delta_buckets(&first_scrape, &last_scrape, "llamacpp_tpot_seconds");

    m.generation_tokens_total = sum_metric_samples(&last_scrape, "llamacpp_tokens_predicted_total");
    m.generation_tokens_per_sec = counter_delta_per_sec(
        sum_metric_samples(&first_scrape, "llamacpp_tokens_predicted_total"),
        sum_metric_samples(&last_scrape, "llamacpp_tokens_predicted_total"),
        window_secs,
    );
    m.prompt_tokens_per_sec = counter_delta_per_sec(
        sum_metric_samples(&first_scrape, "llamacpp_prompt_tokens_total"),
        sum_metric_samples(&last_scrape, "llamacpp_prompt_tokens_total"),
        window_secs,
    );

    if window_secs.is_finite() && window_secs > f64::EPSILON {
        m.window_duration_secs = Some(window_secs);
    }

    Ok((m, SystemTime::now()))
}

/// `GET /props` fields used for config resolution. Best-effort; all `None` on failure.
#[derive(Debug, Clone, Default)]
pub(crate) struct LlamaCppProps {
    /// `params.n_parallel` - concurrent request slots, llama.cpp's analog to `max_num_seqs`.
    pub total_slots: Option<u32>,
    /// `default_generation_settings.n_ctx` - per-slot context size.
    pub n_ctx: Option<u32>,
    pub model_alias: Option<String>,
    /// GGUF file type string (e.g. "Q4_K - Medium"); conflates dtype and quantization scheme,
    /// unlike vLLM where they're reported separately. Stored in both fields for display.
    pub model_ftype: Option<String>,
    pub model_path: Option<String>,
}

/// One-shot startup scrape for `--max-num-seqs` before diagnose collection starts, mirroring
/// `vllm::preflight_max_num_seqs`. Sourced from `GET /props`'s `total_slots` (llama.cpp's
/// `--parallel`), not `/metrics` - no gauge equivalent exists there.
pub(crate) fn preflight_max_num_seqs(url: &str, timeout: Duration) -> Option<u32> {
    let client = reqwest::blocking::Client::builder()
        .use_rustls_tls()
        .timeout(timeout)
        .build()
        .ok()?;
    let base = super::config::base_url_from_metrics(url);
    fetch_llamacpp_props(&client, &base).total_slots
}

pub(crate) fn fetch_llamacpp_props(client: &reqwest::blocking::Client, base_url: &str) -> LlamaCppProps {
    let url = format!("{}/props", base_url.trim_end_matches('/'));
    let text = match client.get(&url).send().and_then(|r| r.text()) {
        Ok(t) => t,
        Err(_) => return LlamaCppProps::default(),
    };
    let val: serde_json::Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(_) => return LlamaCppProps::default(),
    };
    LlamaCppProps {
        total_slots: val
            .get("total_slots")
            .and_then(|v| v.as_u64())
            .and_then(|n| u32::try_from(n).ok()),
        n_ctx: val
            .get("default_generation_settings")
            .and_then(|v| v.get("n_ctx"))
            .and_then(|v| v.as_u64())
            .and_then(|n| u32::try_from(n).ok()),
        model_alias: val
            .get("model_alias")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string),
        model_ftype: val
            .get("model_ftype")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string),
        model_path: val
            .get("model_path")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_replaces_colon_prefix() {
        let body = "llamacpp:requests_processing 1\n";
        assert_eq!(
            normalize_llamacpp_prometheus_text(body),
            "llamacpp_requests_processing 1\n"
        );
    }

    #[test]
    fn kv_cache_usage_perc_scales_ratio_to_percent() {
        let body = "# TYPE llamacpp:kv_cache_usage_ratio gauge\nllamacpp:kv_cache_usage_ratio 0.25\n";
        let scrape = scrape_from_body(body).unwrap();
        assert_eq!(kv_cache_usage_perc_from_scrape(&scrape), Some(25.0));
    }

    #[test]
    fn scrape_parses_counters_and_gauges() {
        let body = "\
# TYPE llamacpp:requests_processing gauge
llamacpp:requests_processing 2
# TYPE llamacpp:requests_deferred gauge
llamacpp:requests_deferred 0
# TYPE llamacpp:tokens_predicted_total counter
llamacpp:tokens_predicted_total 448
";
        let scrape = scrape_from_body(body).unwrap();
        assert_eq!(
            first_gauge(&scrape, "llamacpp_requests_processing"),
            Some(2.0)
        );
        assert_eq!(
            sum_metric_samples(&scrape, "llamacpp_tokens_predicted_total"),
            Some(448.0)
        );
    }
}
