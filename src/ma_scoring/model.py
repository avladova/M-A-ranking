"""Transparent illustrative M&A scoring model for the accelerator MVP."""
from math import exp, tanh


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def financial_component(fcf_ev: float, wacc: float) -> float:
    """Simplified financial indicator; not a replacement for a DCF model."""
    return tanh(float(fcf_ev) - float(wacc))


def vector_novelty(distance: float, sigma: float) -> float:
    """Gaussian similarity/strategic-fit proxy from vector distance."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return exp(-(float(distance) ** 2) / (2 * float(sigma) ** 2))


def bayes_posterior(prior: float, sensitivity: float, prevalence: float, positive: bool = True) -> float:
    """P(conflict | evidence) for a binary test with sensitivity and false-positive rate."""
    p = clamp(prior)
    se = clamp(sensitivity)
    fp = clamp(prevalence)
    if positive:
        numerator = p * se
        denominator = numerator + (1 - p) * fp
    else:
        numerator = p * (1 - se)
        denominator = numerator + (1 - p) * (1 - fp)
    if denominator == 0:
        return p
    return numerator / denominator


def score_deal(deal: dict, alpha: float | None = None, beta: float | None = None,
               gamma: float | None = None, integration_risk: float | None = None) -> dict:
    """Return score and interpretable components. Weights are demo parameters."""
    a = float(deal["alpha"] if alpha is None else alpha)
    b = float(deal["beta"] if beta is None else beta)
    g = float(deal["gamma"] if gamma is None else gamma)
    risk = clamp(deal["integration_risk"] if integration_risk is None else integration_risk)
    if a < 0 or b < 0 or g < 0:
        raise ValueError("alpha, beta and gamma must be non-negative")
    financial = financial_component(deal["fcf_ev"], deal["wacc"])
    novelty = vector_novelty(deal["vector_distance"], deal["sigma"])
    synergy = clamp(deal["graph_synergy"])
    strategic = a * novelty + b * synergy
    risk_multiplier = exp(-g * risk)
    score = financial * strategic * risk_multiplier
    return {
        "financial": financial,
        "novelty": novelty,
        "synergy": synergy,
        "strategic": strategic,
        "integration_risk": risk,
        "risk_multiplier": risk_multiplier,
        "score": score,
    }
