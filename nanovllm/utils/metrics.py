# nanovllm/utils/metrics.py
from dataclasses import dataclass, field
from time import perf_counter
import statistics


@dataclass
class RequestMetrics:
    seq_id: int
    num_prompt_tokens: int
    num_cached_tokens: int = 0
    ttft: float = 0.0
    itl_list: list[float] = field(default_factory=list)
    e2e_latency: float = 0.0

    @property
    def mean_itl(self):
        return statistics.mean(self.itl_list) if self.itl_list else 0.0

    @property
    def cache_hit_rate(self):
        return self.num_cached_tokens / self.num_prompt_tokens if self.num_prompt_tokens else 0.0


class MetricsCollector:
    """
    Collects per-request latency metrics for LLM inference.

    Expected call order per request:
      on_request_added → on_prefill_done → on_token_generated* → on_request_finished

    on_preempt may be called any time between added and finished.
    on_generate_start/end wrap the full generate() call.
    on_spec_step is called after each speculative decoding step.
    """

    def __init__(self):
        self._start_times: dict[int, float] = {}
        self._last_token_times: dict[int, float] = {}
        self._pending: dict[int, RequestMetrics] = {}
        self.completed: list[RequestMetrics] = []
        self.preemption_count: int = 0
        self._generate_start: float = 0.0
        self._generate_wall: float = 0.0
        self._spec_drafted: int = 0
        self._spec_accepted: int = 0
        self._prefill_done_sids: set[int] = set()

    def on_request_added(self, seq):
        now = perf_counter()
        self._start_times[seq.seq_id] = now
        self._pending[seq.seq_id] = RequestMetrics(seq.seq_id, seq.num_prompt_tokens)

    def on_prefill_done(self, seq):
        now = perf_counter()
        m = self._pending[seq.seq_id]
        m.ttft = now - self._start_times[seq.seq_id]
        m.num_cached_tokens = seq.num_cached_tokens
        self._last_token_times[seq.seq_id] = now
        self._prefill_done_sids.add(seq.seq_id)   # mark for first-token skip

    def on_token_generated(self, seq):
        sid = seq.seq_id
        if sid not in self._last_token_times:
            return  # seq not yet prefilled
        now = perf_counter()
        if sid in self._prefill_done_sids:
            # first token after prefill: update timestamp but don't record ITL
            # (the interval from on_prefill_done to here is near-zero overhead, not real ITL)
            self._prefill_done_sids.discard(sid)
            self._last_token_times[sid] = now
            return
        m = self._pending[sid]
        m.itl_list.append(now - self._last_token_times[sid])
        self._last_token_times[sid] = now

    def on_request_finished(self, seq):
        sid = seq.seq_id
        m = self._pending.pop(sid)
        m.e2e_latency = perf_counter() - self._start_times.pop(sid)
        self._last_token_times.pop(sid, None)
        self._prefill_done_sids.discard(sid)   # clean up in case of 1-token completion
        self.completed.append(m)

    def on_preempt(self, seq):
        self.preemption_count += 1

    def on_spec_step(self, num_drafted: int, num_accepted: int):
        """Speculative decoding 钩子，记录 draft acceptance。"""
        self._spec_drafted += num_drafted
        self._spec_accepted += num_accepted

    def on_generate_start(self):
        self._generate_start = perf_counter()

    def on_generate_end(self):
        self._generate_wall += perf_counter() - self._generate_start

    def summary(self) -> dict:
        ttfts = [m.ttft for m in self.completed]
        itls  = [x for m in self.completed for x in m.itl_list]
        e2es  = [m.e2e_latency for m in self.completed]
        hits  = [m.cache_hit_rate for m in self.completed]
        total_tokens = sum(len(m.itl_list) for m in self.completed)

        def pct(data, p):
            if not data:
                return 0.0
            idx = min(int(len(data) * p / 100), len(data) - 1)
            return sorted(data)[idx]

        throughput = total_tokens / self._generate_wall if self._generate_wall > 0 else 0.0

        result = {
            "ttft_p50":            pct(ttfts, 50),
            "ttft_p99":            pct(ttfts, 99),
            "itl_p50":             pct(itls, 50),
            "itl_p99":             pct(itls, 99),
            "e2e_p50":             pct(e2es, 50),
            "e2e_p99":             pct(e2es, 99),
            "cache_hit_rate_mean": statistics.mean(hits) if hits else 0.0,
            "preemption_count":    self.preemption_count,
            "throughput_tok_s":    throughput,
        }
        if self._spec_drafted > 0:
            result["spec_acceptance_rate"] = self._spec_accepted / self._spec_drafted
        return result

    def print_summary(self):
        s = self.summary()
        print("\n=== Metrics Summary ===")
        print(f"  Requests completed : {len(self.completed)}")
        print(f"  Throughput         : {s['throughput_tok_s']:.1f} tok/s")
        print(f"  TTFT  p50={s['ttft_p50']*1000:.1f}ms  p99={s['ttft_p99']*1000:.1f}ms")
        print(f"  ITL   p50={s['itl_p50']*1000:.1f}ms  p99={s['itl_p99']*1000:.1f}ms")
        print(f"  E2E   p50={s['e2e_p50']*1000:.1f}ms  p99={s['e2e_p99']*1000:.1f}ms")
        print(f"  Cache hit rate     : {s['cache_hit_rate_mean']:.1%}")
        print(f"  Preemptions        : {s['preemption_count']}")
        if "spec_acceptance_rate" in s:
            print(f"  Spec accept rate   : {s['spec_acceptance_rate']:.1%}")
