import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ma_scoring.model import bayes_posterior, score_deal  # noqa: E402

DEALS = json.loads((ROOT / "data" / "deals.json").read_text(encoding="utf-8"))
EVIDENCE = json.loads((ROOT / "data" / "dd_evidence.json").read_text(encoding="utf-8"))

st.set_page_config(page_title="M&A Vector Scoring", layout="wide")
st.title("M&A Vector Scoring")
st.caption("Демонстратор MVP · синтетические сделки · прозрачная модель")

SCENARIOS = {
    "Bolt-on: синергия": {"deal": "TARGET-A", "alpha": 0.35, "beta": 0.65, "gamma": 1.8,
        "note": "Акцент на расширении клиентской базы и кросс-продажах."},
    "Венчурная технология: новизна": {"deal": "TARGET-B", "alpha": 0.80, "beta": 0.20, "gamma": 1.2,
        "note": "Акцент на технологической новизне при умеренной текущей финансовой отдаче."},
    "Высокий интеграционный риск": {"deal": "TARGET-C", "alpha": 0.30, "beta": 0.70, "gamma": 2.2,
        "note": "Экспоненциальный множитель снижает привлекательность сделки с высоким риском интеграции."},
    "Due Diligence: новое свидетельство": {"deal": "TARGET-A", "alpha": 0.45, "beta": 0.55, "gamma": 2.0,
        "note": "Новое свидетельство из Data Room обновляет вероятность конфликта и пересчитывает итоговый балл."},
}

scenario_name = st.selectbox("Сценарий демонстрации", list(SCENARIOS))
cfg = SCENARIOS[scenario_name]
deal = next(d.copy() for d in DEALS if d["id"] == cfg["deal"])
st.info(cfg["note"])

with st.sidebar:
    st.header("Параметры стратегии")
    alpha = st.slider("α — векторная составляющая", 0.0, 1.0, float(cfg["alpha"]), 0.05)
    beta = st.slider("β — графовая синергия", 0.0, 1.0, float(cfg["beta"]), 0.05)
    gamma = st.slider("γ — штраф за риск", 0.0, 4.0, float(cfg["gamma"]), 0.1)
    st.caption("Коэффициенты демонстрационные, не обучены на исторических сделках.")

risk = float(deal["integration_risk"])
prior = risk
posterior = risk
evidence_name = "Не применяется"
if "Due Diligence" in scenario_name:
    evidence_name = st.selectbox("Свидетельство из Data Room", [e["name"] for e in EVIDENCE])
    evidence = next(e for e in EVIDENCE if e["name"] == evidence_name)
    prior = float(evidence["p0"])
    posterior = bayes_posterior(prior, evidence["sensitivity"], evidence["prevalence"], evidence["positive"])
    risk = posterior

result = score_deal(deal, alpha=alpha, beta=beta, gamma=gamma, integration_risk=risk)
baseline = score_deal(deal, alpha=alpha, beta=beta, gamma=gamma, integration_risk=prior)

st.subheader(deal["name"])
st.write(deal["description"])
c1, c2, c3, c4 = st.columns(4)
c1.metric("Итоговый скоринг T", f'{result["score"]:.4f}', delta=f'{result["score"]-baseline["score"]:+.4f}' if "Due Diligence" in scenario_name else None)
c2.metric("Финансовый компонент", f'{result["financial"]:.3f}')
c3.metric("Векторная близость", f'{result["novelty"]:.3f}')
c4.metric("Графовая синергия", f'{result["synergy"]:.3f}')

st.subheader("Разложение балла")
parts = pd.DataFrame([
    {"Компонент": "Финансовый F_base", "Значение": result["financial"]},
    {"Компонент": "Новизна N_vec", "Значение": result["novelty"]},
    {"Компонент": "Синергия S_graph", "Значение": result["synergy"]},
    {"Компонент": "Множитель риска exp(−γR)", "Значение": result["risk_multiplier"]},
    {"Компонент": "Интеграционный риск R", "Значение": result["integration_risk"]},
])
st.dataframe(parts, use_container_width=True, hide_index=True)
st.caption("T = tanh(FCF/EV − WACC) × (α·N_vec + β·S_graph) × exp(−γ·R_int)")

if "Due Diligence" in scenario_name:
    st.subheader("Байесовское обновление риска")
    left, right = st.columns(2)
    left.metric("Априорная вероятность", f"{prior:.1%}")
    right.metric("Апостериорная вероятность", f"{posterior:.1%}", delta=f"{posterior-prior:+.1%}")
    st.write(evidence["description"])
    st.caption("Обновление условно на заданные чувствительность и частоту ложноположительного сигнала. Их необходимо калибровать.")

st.subheader("Рейтинг синтетического портфеля")
ranked = []
for item in DEALS:
    item_result = score_deal(item, alpha=alpha, beta=beta, gamma=gamma)
    ranked.append({"Цель": item["name"], "Сектор": item["sector"], "Скоринг T": item_result["score"],
                   "Новизна": item_result["novelty"], "Синергия": item_result["synergy"],
                   "Риск": item_result["integration_risk"]})
ranked.sort(key=lambda row: row["Скоринг T"], reverse=True)
st.dataframe(pd.DataFrame(ranked).round(4), use_container_width=True, hide_index=True)

with st.expander("Методология и ограничения"):
    st.markdown("""
    - Данные и значения параметров синтетические; это демонстрация принципа, не инвестиционная рекомендация.
    - Скоринговый балл не является вероятностью успеха сделки.
    - Новизна в текущей реализации — гауссовская функция расстояния; интерпретацию и направление шкалы нужно закрепить при калибровке.
    - Графовая синергия задаётся в данных, не вычисляется из реального графа клиентов и продуктов.
    - Для пилота нужны исторические данные закрытых и неуспешных сделок, определение целевой переменной, backtesting и проверка устойчивости.
    """)
