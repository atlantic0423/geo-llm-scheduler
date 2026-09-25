"""Memory/disk admission and deadline guards for worker scheduling."""

from geo_llm_scheduler.experiments.campaign.support import process_peak_rss_gb, safe_worker_count


def test_worker_cap_respects_cpu_memory_and_maximum():
    assert safe_worker_count(16, 8.0, 1.0, 2.0, 5) == 3
    assert safe_worker_count(2, 8.0, 1.0, 2.0, 5) == 1
    assert safe_worker_count(16, 2.0, 1.0, 2.0, 5) == 0


def test_process_peak_rss_is_positive_and_finite():
    peak = process_peak_rss_gb()
    assert 0 < peak < 1000
