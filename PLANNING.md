# FL Wind Forecasting — Project Plan

## Problem
Predict next-hour wind speed at 10 FMI coastal weather stations using
federated learning (GTVMin framework). Motivation: maritime wind forecasting
for sailors on the SW Finnish coast and archipelago.

## FL Theory
Three-step workflow:
1. Formulate as GTVMin: min Σᵢ Lᵢ(w⁽ⁱ⁾) + α Σ_{i,i'} Aᵢᵢ' ‖w⁽ⁱ⁾ − w⁽ⁱ'⁾‖²
2. Fixed-point equation: w⁽ⁱ⁾ = F⁽ⁱ⁾(w⁽¹⁾,...,w⁽ⁿ⁾)
   where F⁽ⁱ⁾ = w⁽ⁱ⁾ − η[∇Lᵢ(w⁽ⁱ⁾) + 2α Σᵢ'∈N(i) Aᵢᵢ'(w⁽ⁱ⁾ − w⁽ⁱ'⁾)]
3. Iterate F⁽ⁱ⁾ synchronously until convergence → FL algorithm

## Nodes — n=10 FMI Stations
Exposed (open sea / archipelago):
- Porvoo Kalbådagrund     FMISID: 101022
- Raasepori Jussarö       FMISID: verify at fetch time
- Parainen Utö            FMISID: verify at fetch time
- Kustavi Isokari         FMISID: verify at fetch time
- Mustasaari Valassaaret  FMISID: verify at fetch time

Sheltered (coastal cities):
- Helsinki Kaisaniemi     FMISID: 100971
- Turku Rajakari          FMISID: verify at fetch time
- Rauma                   FMISID: verify at fetch time
- Pori Tahkoluoto         FMISID: verify at fetch time
- Vaasa                   FMISID: verify at fetch time

## Data
Source: FMI Open Data (https://en.ilmatieteenlaitos.fi/open-data)
Library: fmiopendata (Python)
Period: Jan–Dec 2024, hourly observations

Features x ∈ R⁵:
- wind speed at hour r [m/s]
- sin(wind direction)
- cos(wind direction)
- pressure MSL [hPa]
- air temperature [°C]

Label y: wind speed at hour r+1

Preprocessing:
- Drop rows with any missing values
- Chronological split: train 70% / val 15% / test 15%
  (approx Jan–mid Sep / mid Sep–mid Nov / mid Nov–Dec)
- Standardize features: zero mean, unit variance using train-set statistics only

## Local Model and Loss
Model: linear regression  ŷ⁽ⁱ⁾ = w⁽ⁱ⁾ᵀx,  w⁽ⁱ⁾ ∈ R⁵
Loss:  MSE  Lᵢ(w⁽ⁱ⁾) = (1/mᵢ) Σᵣ (yᵣ − w⁽ⁱ⁾ᵀxᵣ)²
Gradient: ∇Lᵢ(w⁽ⁱ⁾) = (2/mᵢ) Xᵢᵀ(Xᵢw⁽ⁱ⁾ − yᵢ)
Choice justified: convex loss + linear model guarantees convergence of GTVMin

## Two FL Systems (structural difference in edge definition)
System A — geographic proximity:
- k=3 nearest neighbors by Euclidean lat/lon distance
- Edge weight: exp(−d/σ), σ=200 km
- Baseline: map proximity as proxy for useful collaboration

System B — wind correlation:
- Top-k=3 most correlated neighbors per station
- Pearson correlation computed on TRAINING SET wind speed only (no leakage)
- Edge weight: correlation value
- Hypothesis: shared wind regime beats map proximity

Baseline: α=0 (local-only, no collaboration)

## FL Algorithm
Update rule (synchronous):
  w⁽ⁱ,t+1⁾ = w⁽ⁱ,t⁾ − η[∇Lᵢ(w⁽ⁱ,t⁾) + 2α Σᵢ'∈N(i) Aᵢᵢ'(w⁽ⁱ,t⁾ − w⁽ⁱ',t⁾)]

Initialisation: w⁽ⁱ,0⁾ = 0 for all i
Iterations: T = 200
Learning rate: η = 0.01 (reduce to 0.001 if training loss diverges)
α tuning: grid search over {0.001, 0.01, 0.1, 1, 10}
           select α by minimum average validation MSE across all stations

## Implementation Files
src/fl_wind/
  fetch_data.py      — pull FMI data, verify station names, save to data/raw/
  preprocess.py      — build features, sin/cos encode, split, standardize
  graph.py           — build adjacency matrices for System A and B
  fl_algorithm.py    — GTVMin update loop (pure NumPy)
  experiments.py     — run baseline, System A, System B; α sweep
  plots.py           — loss curves, scatter plots, α sweep, results table

## Required Sanity Checks
1. Loss curves: training MSE decreases monotonically for convex loss
2. Baseline comparison: FL (best α) must improve on α=0
3. Predictions vs actuals: scatter plot for 2-3 stations, points near diagonal
4. α sweep: small α ≈ local-only, large α ≈ shared model
5. System comparison: table of train/val/test MSE for baseline, A, B
6. Exposed vs sheltered breakdown: do exposed stations benefit more from FL?

## Report Structure (5 pages, LaTeX template from course repo)
1. Introduction — maritime motivation, FL setting
2. Problem formulation — nodes, data, features, model, loss, GTVMin
3. Methodology — fixed-point equation, algorithm, System A vs B
4. Numerical experiments — sanity checks, comparison table, discussion
5. Conclusion + References