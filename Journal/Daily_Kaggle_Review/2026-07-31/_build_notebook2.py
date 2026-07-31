# -*- coding: utf-8 -*-
"""
Part 2: original markdown cells (raw, English) + Japanese explanations
(What/Why for each code cell, adapted translations for original markdown),
intro/metric/critique cells, and final nbformat v4 JSON assembly.
"""
import json

def eq(s):
    return s.replace('#EQ#', '=')

with open('_raw_code_cache.json', encoding='utf-8') as f:
    CODE = json.load(f)  # already eq()-resolved, keys are str(int)

# ============================================================
# RAW ORIGINAL MARKDOWN CELLS (index -> source, English, as extracted)
# ============================================================
RAW_MD = {}

RAW_MD[0] = '''### Key Contributions

**1. Ensemble Feature Selection Framework**

We developed a 7-method ensemble feature selection pipeline that goes beyond traditional approaches:

- **Granger Causality Selector** — Applies Granger causality tests to identify features with true predictive power. Tests multiple lags (up to 5 periods) with significance threshold p < 0.10. Focus: temporal causality, not just correlation.
- **Hierarchical Feature Clusterer** — Agglomerative clustering on the feature correlation matrix. Reduces redundancy by selecting representative features from each cluster. Preserves information diversity while eliminating multicollinearity.
- **Regime Aware Feature Selector** — Detects market regimes using rolling volatility windows (Low/Medium/High). Selects features robust across different market conditions. Ensures model stability during regime shifts.
- **Rolling Window Ensemble Selector** — Validates feature importance across 5 rolling windows (~1 year each). Ensemble of RandomForest + GradientBoosting per window. Prioritizes temporally stable features.
- **Feature Interaction Generator** — Creates synergistic feature combinations (multiplicative, additive, ratio). Captures non-linear relationships through engineered interactions. Tests up to 2-way interactions with top-performing features.
- **Multi-Method Importance Ranking** — Final ranking using RandomForest + GradientBoosting + LASSO ensemble. Combines tree-based and linear model perspectives. Creates nested feature sets (top_10 through top_40).

Result: Reduced 95+ features to 101 carefully selected features, then ranked into 7 optimized feature sets.

**2. Competition-Specific Optimization**

Direct ScoreMetric Optimization — unlike standard approaches that optimize MSE or R², this notebook directly optimizes the competition's official metric:

`ScoreMetric = Sharpe_Ratio / (Volatility_Penalty × Return_Penalty)`

Position Multiplier Optimization — each model receives its own optimized position multiplier through L-BFGS-B optimization: `position = 1.0 + predicted_return × multiplier`. Transforms regression into optimal portfolio allocation, maximizes ScoreMetric on validation set, enforces competition constraints (positions in [0.0, 2.0]).

**3. Advanced ML Model Suite**

Six regression models across seven feature sets (35+ configurations): LightGBM, XGBoost, Ridge, ElasticNet, RandomForest, GaussianProcess — tuned via Bayesian hyperparameter optimization (Optuna, TPE sampler), optimizing the competition metric directly rather than a proxy loss.'''

RAW_MD[2] = '''## I. LOAD DATA'''

RAW_MD[4] = '''## II. VISUALIZE TARGET DISTRIBUTION'''

RAW_MD[6] = '''### Executive Summary

This report analyzes S&P 500 excess returns over approximately 35 years of trading data. Key findings reveal a highly efficient market with near-zero predictability, significant tail risks, and volatility clustering patterns that inform investment strategy.

**1. Market Efficiency & Return Behavior** — Mean excess return is ~0.00005 (essentially zero); distribution is approximately normal with fat tails. The market demonstrates high efficiency: daily excess returns fluctuate around zero, indicating no systematic free gains.

**2. Tail Risk Analysis** — The Q-Q plot shows the left tail (losses) deviates below the theoretical normal line (crashes are more severe than normal distribution predicts) while the right tail (gains) deviates above it (rallies can exceed expectations).

**3. Volatility Regime Analysis** — Historical volatility clusters into identifiable crisis/calm periods (e.g., 2008 crisis ~30-50% annualized vol, 2010-2019 bull market ~10-15%, 2020 COVID crash ~40-55%). High-volatility periods persist for weeks/months before subsiding, creating actionable signals (reduce position when vol rising, increase when falling).

**4. Cumulative Returns** — Two major drawdowns of 40-50% are visible, with 2-4 year recovery periods.

**5. Return Predictability** — Autocorrelation is approximately zero at all lags (1 to 8,000 days): yesterday's return provides no information about today's return, so momentum and mean-reversion strategies both lack a statistical edge here.

**6. Risk-Return Tradeoff** — No clear relationship between volatility and return is visible; risk-adjusted metrics (Sharpe ratio) are argued to give better guidance than raw returns.

**7. Seasonality** — No statistically significant monthly pattern is detected; "Sell in May" and the "January Effect" are not supported by this dataset.'''

RAW_MD[7] = '''## III. MISSING DATA ANALYSIS'''

RAW_MD[11] = '''## IV. CORRELATION ANALYSIS (Time-Varying!)'''

RAW_MD[14] = '''### Executive Summary

**1. Correlation Distribution by Feature Group** — All feature groups (Volatility, Market, Sentiment, Regime, Price, Economic, Interest) cluster tightly around zero correlation with the target. No single category provides meaningful linear predictive power on its own.

**2. Statistical Significance** — The p-value histogram is close to uniform between 0 and 1. Most correlations are not statistically significant; even the "top" features may be spurious. With 94 features tested, about 5 would show p < 0.05 by chance alone (the multiple-testing problem), and after a Bonferroni correction (p < 0.05/94 ≈ 0.0005), essentially none remain significant.

**3. Data Availability vs. Correlation Strength** — A suspicious pattern appears: features with fewer valid samples tend to show higher correlations. This is likely an artifact (survivorship bias, small-sample inflation, or regime-specific relationships that don't generalize) rather than real signal.'''

RAW_MD[15] = '''## V. FEATURE-TARGET RELATIONSHIP ANALYSIS OVER TIME'''

RAW_MD[18] = '''### 1. ROLLING CORRELATION ANALYSIS'''

RAW_MD[20] = '''### 2. ROLLING MUTUAL INFORMATION'''

RAW_MD[23] = '''### 3. STRUCTURAL BREAK DETECTION'''

RAW_MD[26] = '''### 4. FEATURE STABILITY SCORING'''

RAW_MD[29] = '''### Executive Summary

This analysis reveals a critical finding: feature-target relationships are **not stable over time**. Correlations that appear useful in one period may flip sign or disappear entirely in another. This has profound implications for model development — static models will fail.

**1. Rolling Correlation Analysis** — The 3-month rolling correlation for top features swings from -0.6 to +0.6, with massive spikes around the COVID period (date ~6000-7000) and near-zero correlation most of the calm periods. A feature showing +0.05 correlation on the *full* dataset might have been +0.4 in one period and -0.3 in another — the long-run average hides the instability.

**2. Rolling Mutual Information** — Unlike correlation (linear only), mutual information captures non-linear dependencies too, and it spikes dramatically during the COVID crisis (date ~6500-8000): the market became temporarily more predictable, and non-linear relationships strengthened during that volatility.

**3. Sign Flipping** — Feature M1 shows a *negative* correlation with the target in Period 1 but a *positive* correlation in Period 4. A model trained only on Period 1 data would short when M1 is high; trained on Period 4, it should go long when M1 is high. A static, single-period model fails here by construction.

**4. Feature Stability Ranking** — Features V10, M1, V7, V13, M17 rank as most stable (stability score > 0.85, safe to use); I2, P8, E19 rank as least stable (score < 0.70, caution advised — avoid or use adaptively).'''

RAW_MD[31] = '''## VI. FEATURE SELECTION FOR ML MODELING

### 1. Strategy: Multi-Method Ensemble Approach

Based on the EDA findings, the notebook combines multiple feature selection methods: Causality-Driven Selection (Granger Causality), Hierarchical Clustering, Regime-Aware Dynamic Importance, Rolling Window Ensembles, and Feature Interaction Discovery — combined into an ensemble score to identify the most robust features.'''

RAW_MD[40] = '''### Key Findings

**What Drives Market Predictions** — Market indicators (27.5%), Price patterns (25%), Volatility measures (17.5%), Economic data (15%), Sentiment (12.5%), Interest rates (2.5%). Technical analysis (Price + Market = 52.5%) drives predictions more than fundamentals; both chart patterns and economic data appear necessary.

**Feature Correlation** — High correlation (>0.8) between features means redundant information; low correlation (<0.3) means the features capture different aspects. Diverse, low-correlated features avoid "echo chambers" where ten correlated features are really one perspective repeated ten times.

**The 80/20 Rule** — Mean feature importance is only 0.12; 80% of features contribute almost nothing. Just 10 features capture ~60% of predictive power, 35 features capture ~80%, and 49 features capture ~90%. More data is not automatically better — most features are noise, and the vital few matter most.

**Feature Group Performance** — Price (~0.19) and Volatility (~0.17) show moderate consistency; Market (~0.14) shows high variance (some excellent, some worthless); Sentiment/Economic/Interest Rate hover around ~0.09-0.10.'''

RAW_MD[44] = '''### 2. Position Optimization'''

RAW_MD[46] = '''### 3. Train Models with Different Feature Sets'''

RAW_MD[48] = '''#### Bayesian Optimization for hyperparameter tuning'''

RAW_MD[55] = '''## VII. SUBMISSION'''

with open('_raw_md_cache.json', 'w', encoding='utf-8') as f:
    json.dump({str(k): eq(v) for k, v in RAW_MD.items()}, f)

print("RAW_MD cells loaded:", len(RAW_MD))
assert len(RAW_MD) == 19, f"expected 19 markdown cells, got {len(RAW_MD)}"
print("OK: wrote _raw_md_cache.json")
