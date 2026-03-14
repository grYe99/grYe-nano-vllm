# tests/test_metrics.py
import time
from unittest.mock import MagicMock
from nanovllm.utils.metrics import MetricsCollector, RequestMetrics


def make_seq(seq_id, num_prompt_tokens, num_cached_tokens=0):
    seq = MagicMock()
    seq.seq_id = seq_id
    seq.num_prompt_tokens = num_prompt_tokens
    seq.num_cached_tokens = num_cached_tokens
    return seq


def test_request_metrics_properties():
    m = RequestMetrics(seq_id=0, num_prompt_tokens=100, num_cached_tokens=50)
    m.itl_list = [0.01, 0.02, 0.03]
    assert abs(m.mean_itl - 0.02) < 1e-9
    assert abs(m.cache_hit_rate - 0.5) < 1e-9


def test_ttft_recorded():
    mc = MetricsCollector()
    seq = make_seq(0, 10)
    mc.on_request_added(seq)
    time.sleep(0.01)
    mc.on_prefill_done(seq)
    assert mc._pending[0].ttft > 0.005


def test_itl_recorded():
    mc = MetricsCollector()
    seq = make_seq(0, 10)
    mc.on_request_added(seq)
    mc.on_prefill_done(seq)
    # first on_token_generated after prefill: should NOT append to itl_list
    mc.on_token_generated(seq)
    assert len(mc._pending[0].itl_list) == 0, "first token after prefill should not be in itl_list"
    # subsequent tokens: should be recorded
    time.sleep(0.01)
    mc.on_token_generated(seq)
    time.sleep(0.01)
    mc.on_token_generated(seq)
    assert len(mc._pending[0].itl_list) == 2
    assert all(itl > 0.005 for itl in mc._pending[0].itl_list)


def test_request_finished_moves_to_completed():
    mc = MetricsCollector()
    seq = make_seq(0, 10)
    mc.on_request_added(seq)
    mc.on_prefill_done(seq)
    mc.on_token_generated(seq)
    mc.on_request_finished(seq)
    assert len(mc.completed) == 1
    assert 0 not in mc._pending
    assert mc.completed[0].e2e_latency > 0


def test_preemption_count():
    mc = MetricsCollector()
    seq = make_seq(0, 10)
    mc.on_preempt(seq)
    mc.on_preempt(seq)
    assert mc.preemption_count == 2


def test_summary_keys():
    mc = MetricsCollector()
    seq = make_seq(0, 100, num_cached_tokens=50)
    mc.on_request_added(seq)
    mc.on_prefill_done(seq)
    mc.on_generate_start()
    mc.on_token_generated(seq)
    mc.on_request_finished(seq)
    mc.on_generate_end()
    s = mc.summary()
    for key in ["ttft_p50", "ttft_p99", "itl_p50", "itl_p99",
                "e2e_p50", "e2e_p99", "cache_hit_rate_mean",
                "preemption_count", "throughput_tok_s"]:
        assert key in s, f"missing key: {key}"
    assert s["throughput_tok_s"] >= 0


def test_spec_step():
    mc = MetricsCollector()
    mc.on_spec_step(4, 3)
    mc.on_spec_step(4, 2)
    # need at least one completed request for summary
    seq = make_seq(0, 10)
    mc.on_request_added(seq)
    mc.on_prefill_done(seq)
    mc.on_generate_start()
    mc.on_token_generated(seq)
    mc.on_request_finished(seq)
    mc.on_generate_end()
    s = mc.summary()
    assert "spec_acceptance_rate" in s
    assert abs(s["spec_acceptance_rate"] - 5/8) < 1e-9
