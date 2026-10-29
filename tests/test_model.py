import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ma_scoring.model import bayes_posterior, score_deal, vector_novelty  # noqa: E402


SAMPLE = {
    "fcf_ev": 0.14, "wacc": 0.11, "vector_distance": 0.8, "sigma": 1.5,
    "graph_synergy": 0.86, "integration_risk": 0.24, "alpha": 0.45, "beta": 0.55, "gamma": 1.8
}


def test_vector_novelty_is_bounded():
    assert 0 < vector_novelty(0.8, 1.5) <= 1
    assert vector_novelty(0, 1.5) == 1


def test_more_distance_means_lower_gaussian_similarity():
    assert vector_novelty(1.2, 1.0) < vector_novelty(0.4, 1.0)


def test_higher_risk_reduces_score_when_other_inputs_fixed():
    low = score_deal(SAMPLE, integration_risk=0.1)["score"]
    high = score_deal(SAMPLE, integration_risk=0.8)["score"]
    assert high < low


def test_positive_evidence_increases_posterior_for_given_example():
    assert bayes_posterior(0.25, 0.85, 0.12, True) > 0.25


def test_score_components_have_expected_ranges():
    result = score_deal(SAMPLE)
    assert 0 <= result["novelty"] <= 1
    assert 0 <= result["synergy"] <= 1
    assert 0 <= result["integration_risk"] <= 1
