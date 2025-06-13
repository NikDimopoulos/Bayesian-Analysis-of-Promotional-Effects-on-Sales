# Bayesian-Analysis-of-Promotional-Effects-on-Sales
Bayesian regression on sales data using Gibbs sampling — FEM21026 assignment (Erasmus University)

## NOTE!!!
This project was submitted as part of an individual assignment and is intended for educational purposes only. Data files are not included, as it was provided by Erasmus University specifically for coursework.

## Problem Description

We model the logarithm of unit sales (`log(sales_t)`) using the following regression:

log(sales_t) = α_t + β₁·display_t + β₂·coupon_t + β₃·log(price_t) + ε_t, ε_t ∼ N(0, σ²)


where:
- `display_t` and `coupon_t` are binary indicators (0/1) for promotions.
- `price_t` is the price in week _t_.
- `α_t` varies over time:
  - For t ≤ 62, α_t = β₀
  - For t > 62, α_t = γ·β₀

## Bayesian Analysis

The parameters {β₀, β₁, β₂, β₃, γ, σ²} are estimated using a **Gibbs sampler**. The prior setup is:

- γ ∼ N(b=1, σ²·B=1)
- Improper flat priors for β_j (j = 0,...,3): p(β_j) ∝ 1
- Jeffrey's prior for σ²: p(σ²) ∝ σ⁻²

## Dataset

The analysis uses weekly data on:
- Sales
- Price
- Coupon promotions
- Display promotions

for a selected brand over 124 weeks.

## Files

- `Code- Nikos Dimopoulos.py`: Python implementation of the Gibbs sampler.
- Data files (not included in the repository):
  - `sales.xls`
  - `price.xls`
  - `coupon.xls`
  - `display.xls`

## Output

The code provides:
- Posterior means and 10th–90th percentiles for each parameter
- Pr (γ < 1|y)

