# Speculative Bubbles in Meme Stocks — PSY Test (2014–2025)

> Master's degree project — *N.F.*
> All rights reserved. See `LICENSE` for details.

---

## Overview

This project applies the **PSY methodology** (Phillips, Shi & Yu, 2015) to detect and date-stamp speculative bubbles in four meme stocks — GameStop (GME), AMC Entertainment (AMC), BlackBerry (BB), and Nokia (NOK) — over the period January 2014 to December 2025, using the Russell 2000 (^RUT) as the market benchmark.

The analysis follows the approach of Basele, Phillips & Shi (2025) and covers three main steps: construction of price/index ratios, recursive bubble detection via BSADF/GSADF sequences with Monte Carlo critical values, and out-of-sample portfolio construction comparing equal-weight vs inverse-volatility weighting schemes.

All data is downloaded automatically via `yfinance`. No local files are required.

---

## Project Structure

```
.
└── meme_stocks_psy_bubbles.ipynb   # Main Jupyter notebook (full analysis)
```

---

## Methodology

### 1. Data and Price/Index Ratios
Monthly adjusted close prices are downloaded via `yfinance` for GME, AMC, BB, NOK, and ^RUT over January 2014 – December 2025 (144 monthly observations). Each price series is **normalized to base 100** from the first available observation. The **price/index ratio** is then constructed by dividing each stock's normalized price by the normalized Russell 2000 — a ratio above 1 indicates outperformance relative to the index, below 1 underperformance.

Visual inspection already reveals the key pattern: GME reaches a ratio of approximately 7 in early 2021 (the GameStop short squeeze), AMC shows a similar but less extreme spike, while BB and NOK exhibit steady downward trends with no meaningful relative outperformance.

### 2. PSY Bubble Detection

#### BSADF and GSADF statistics
The **PSY test** detects explosive dynamics in a time series by applying a right-tailed ADF test over all backward-expanding subsamples ending at each point t. For each point t, the **BSADF** statistic is defined as:

```
BSADF(t) = sup { ADF(r1, t) : r1 ∈ [0, t - min_w] }
```

The **GSADF** is the supremum of the BSADF sequence over the entire sample and serves as the global test statistic for the presence of at least one bubble episode.

The minimum window size follows the PSY-recommended formula:

```python
min_w = int(np.ceil((0.01 + 1.8 / np.sqrt(n)) * n))
```

All ADF regressions use a **constant**, lag selection via **BIC**, and `maxlag=4`. The BSADF sequence is computed in parallel via `joblib` for computational efficiency.

#### Monte Carlo critical values
Critical values are simulated under the null hypothesis of a random walk (10,000 replications, n = 144). The **95% critical value** is **1.318**. The full simulated distribution is plotted alongside the normal approximation for diagnostic purposes.

#### Results

| Ticker | GSADF | H₀ rejected at 95%? |
|--------|-------|----------------------|
| GME | 3.789 | Yes |
| AMC | 1.910 | Yes |
| BB | 1.465 | Yes |
| NOK | 0.932 | No |

GME, AMC, and BB exceed the critical threshold, providing statistical evidence of at least one explosive episode. NOK does not.

#### Date stamping
Bubble episodes are identified as the periods during which the BSADF sequence exceeds the 95% critical value. The start of a bubble corresponds to the first crossing above the threshold; the end to the first crossing back below.

| Ticker | Episode | Start | End |
|--------|---------|-------|-----|
| GME | 1 | 2021-01 | 2021-05 |
| AMC | 1 | 2017-08 | 2017-09 |
| AMC | 2 | 2021-06 | 2021-07 |
| BB | 1 | 2022-12 | 2023-01 |
| NOK | — | No episode | — |

The GME episode (January–May 2021) is the most prominent and prolonged, directly corresponding to the Reddit-driven short squeeze. AMC shows two brief episodes. BB presents a more recent episode spanning end-2022 to early 2023. NOK shows no statistically significant explosive dynamics throughout the sample.

### 3. Portfolio Construction and Performance (2021–2025)
Two portfolios are constructed on the four meme stocks over January 2021 – December 2025 and benchmarked against the Russell 2000:

**Equal-Weight (EW):** fixed weights of 1/N = 25% on each stock, implicitly rebalanced monthly.

```python
w_ew = np.ones(len(returns.columns)) / len(returns.columns)
ret_ew = returns.dot(w_ew)
```

**Inverse-Volatility (IV):** weights inversely proportional to each stock's trailing 12-month return standard deviation, rebalanced quarterly.

```python
vol    = returns.rolling(12).std()
w_iv   = (1 / vol).div((1 / vol).sum(axis=1), axis=0)
w_iv   = w_iv.iloc[::3].reindex(returns.index, method='ffill')
ret_iv = (w_iv * returns).sum(axis=1, min_count=1)
```

Performance is evaluated using `quantstats` with `periods=12` (monthly data).

#### Results (January 2021 – December 2025)

| Portfolio | CAGR (%) | Volatility (%) | Sharpe | Sortino | Max DD (%) |
|-----------|----------|----------------|--------|---------|------------|
| Equal-Weight | −15.38 | 54.96 | −0.041 | −0.064 | −82.29 |
| Inverse-Vol | −13.31 | 33.28 | −0.260 | −0.347 | −60.73 |
| Russell 2000 | +3.72 | 20.03 | +0.281 | +0.436 | −28.06 |

Both portfolios produce negative Sharpe and Sortino ratios across the full period, confirming that meme stocks destroyed value in the post-bubble phase. The EW portfolio captured the January 2021 spike (peak cumulative return ~+80%) but subsequently collapsed. The IV portfolio reduced volatility to 33.28% but at the cost of systematically overweighting the weakest stocks (BB, NOK, AMC), as the inverse-vol mechanism assigns higher weights to lower-volatility assets — precisely the ones that continued to deteriorate. The Russell 2000 is the only strategy to close the period in positive territory (+3.72% CAGR, −28.06% max drawdown).

---

## Key Findings

- The PSY test detects statistically significant bubble episodes in **GME, AMC, and BB**; Nokia shows no evidence of explosive dynamics.
- The **GameStop bubble** (Jan–May 2021) is the clearest case: a GSADF of 3.789, well above the 95% critical value of 1.318, and five consecutive months of explosive BSADF.
- **Inverse-volatility weighting does not improve performance** in this universe: the diversification benefit (lower volatility) is offset by a structural bias toward the worst-performing stocks.
- Holding any long-only position in meme stocks from 2021 to 2025 underperformed the Russell 2000 by a wide margin, regardless of the weighting scheme.

---

## Requirements

```bash
pip install numpy pandas matplotlib seaborn yfinance fmfinance quantstats \
            statsmodels scipy scikit-learn joblib
```

```bash
pip install https://github.com/FraMedda/fmfinance/archive/refs/heads/main.zip
```

Python 3.9+ recommended. Run in Jupyter Notebook or JupyterLab.

> **Note on computation time:** The BSADF sequence and Monte Carlo simulation are computationally intensive. Both are parallelized via `joblib` (`n_jobs=-1`). On a modern multi-core machine, expect approximately 5–15 minutes for the full notebook to run from start to finish.

---

## Usage

1. Clone the repository
2. Install the required packages (see above)
3. Open and run `meme_stocks_psy_bubbles.ipynb` sequentially from top to bottom

All data is downloaded automatically within the notebook. No local files are required.

---

## References

- Phillips, P.C.B., Shi, S., & Yu, J. (2015). *Testing for Multiple Explosive Periods.* Journal of the Royal Statistical Society: Series B, 77(2), 467–495.
- Basele, T., Phillips, P.C.B., & Shi, S. (2025). *Date-stamping speculative bubbles.*

---

## License

This project is protected by copyright. See the [`LICENSE`](LICENSE) file for full terms.
**Reusing, copying, modifying, or redistributing** the code or data without explicit written permission from the author is not permitted.
