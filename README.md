 # variational-socio-ecological-dynamics

**A Variational Framework for Socio-Ecological Dynamics under Ecological Constraint Landscapes**

Replication repository for the manuscript submitted to *Ecological Modelling*. This repository contains the raw World Bank data, the panel analysis and visualisation code, and instructions to reproduce all empirical results and figures reported in the paper.

---

## Overview

This project develops a variational ecological-economic framework in which socio-economic trajectories evolve under a constraint landscape $V_D$ representing ecological and institutional limits. Material accumulation ($q_A$) and experiential activity ($q_K$) are the dynamical state variables. Dissipation generates directional adjustment toward dynamically stable configurations, and non-convex constraint landscapes produce path-dependent multi-regime dynamics. An empirical illustration uses World Bank panel data from five structurally diverse economies (Brazil, China, India, Indonesia, Japan) over 2000–2021 to assess whether observed trajectories are consistent with the framework's dynamical predictions.

---

## Repository Structure

```
variational-socio-ecological-dynamics/
│
├── data/
│   ├── raw/                         # Original World Bank WDI downloads (CSV)
│   │   ├── Agriculture/             # AG: NV.AGR.TOTL.ZS
│   │   ├── Co2/                     # AG: EN.GHG.CO2.AG.MT.CE.AR5
│   │   ├── Energy/                  # NY.ADJ.DNGY.GN.ZS
│   │   ├── Forest-area/             # AG.LND.FRST.ZS
│   │   └── Urban-population/        # SP.URB.TOTL
│   └── processed/
│       └── panel_final.csv          # Balanced panel: 5 countries × 22 years × 5 indicators
│
├── code/
│   └── panel_analysis.py            # Main analysis and figure generation script
│
├── outputs/
│   ├── fig2_phase_space.png         # Phase-space trajectories (manuscript Figure 4)
│   ├── fig3_cross_correlation.png   # Cross-correlograms (manuscript Figure 5)
│   ├── fig4_structural_breaks.png   # Chow test heatmap
│   ├── fig5_xai_importance.png      # XAI feature importance
│   ├── fig6_mann_kendall_heatmap.png# Mann-Kendall τ heatmap
│   └── fig1_individual_series.png   # Individual indicator time series
│
└── README.md
```

---

## Data

All raw data are downloaded directly from the [World Bank World Development Indicators](https://databank.worldbank.org/source/world-development-indicators) and are included in `data/raw/` without modification.

| Indicator | WDI Code | Theoretical Role |
|---|---|---|
| Forest area (% of land) | `AG.LND.FRST.ZS` | $V_D$ proxy — ecological constraint landscape |
| Agricultural CO₂ emissions (Mt CO₂e) | `EN.GHG.CO2.AG.MT.CE.AR5` | $q_A$ component — accumulation pressure |
| Energy depletion (% of GNI) | `NY.ADJ.DNGY.GN.ZS` | $q_A$ component — depletion intensity |
| Agricultural value added (% of GDP) | `NV.AGR.TOTL.ZS` | $q_K$ component (inverse) — structural shift |
| Urban population | `SP.URB.TOTL` | $q_K$ component — urbanisation shift |

**Coverage:** Brazil, China, India, Indonesia, Japan — 2000 to 2021 — balanced panel of 110 country-year observations.

Composite proxies are constructed as within-country min-max normalised means:
```
q_A = ( norm(agri_co2) + norm(energy_dep) ) / 2
q_K = ( norm(urban_pop) + (1 - norm(agri_gdp)) ) / 2
V_D = forest_area (% of land, within-country normalised)
```

---

## Code

### Dependencies

```
python >= 3.8
pandas
numpy
scipy
scikit-learn
matplotlib
```

Install all dependencies with:

```bash
pip install pandas numpy scipy scikit-learn matplotlib
```

### Running the analysis

Place `panel_final.csv` in the working directory (or set `INPUT_FILE` in the configuration block), then run:

```bash
python panel_analysis.py
```

This executes the following steps in sequence:

1. Data loading and composite proxy construction
2. Phase-space trajectory plots (`fig2_phase_space.png`)
3. Mann-Kendall monotonic trend tests per country per indicator
4. Cross-correlation analysis of $q_A$ vs $q_K$ (`fig3_cross_correlation.png`)
5. Panel Granger causality tests (pooled OLS F-test, lags 1–3)
6. Chow structural break tests at 2008, 2014, 2020 (`fig4_structural_breaks.png`)
7. Gradient Boosting + Permutation Importance XAI decomposition (`fig5_xai_importance.png`)
8. Mann-Kendall summary heatmap (`fig6_mann_kendall_heatmap.png`)

All figures are saved to `OUTPUT_DIR` (default: working directory). Numerical results are printed to stdout.

### Key configuration options

| Variable | Default | Description |
|---|---|---|
| `INPUT_FILE` | `panel_final.csv` | Path to balanced panel CSV |
| `OUTPUT_DIR` | `./` | Directory for output figures |
| `COUNTRIES` | Brazil, China, India, Indonesia, Japan | Panel countries |
| `YEARS` | 2000–2021 | Analysis period |

---

## Outputs

| File | Description | Used in manuscript |
|---|---|---|
| `fig2_phase_space.png` | Phase-space trajectories $(q_A, q_K)$ for five economies | Yes — Figure 4 |
| `fig3_cross_correlation.png` | Cross-correlograms $q_A \leftrightarrow q_K$, lags −5 to +5 | Yes — Figure 5 |
| `fig4_structural_breaks.png` | Chow test p-value heatmap | Supplementary |
| `fig5_xai_importance.png` | XAI feature importance (MDI and permutation-based) | Supplementary |
| `fig6_mann_kendall_heatmap.png` | Mann-Kendall τ and theoretical consistency heatmap | Supplementary |
| `fig1_individual_series.png` | Individual indicator time series, all countries | Supplementary |

---

## Empirical Analysis Notes

- **Mann-Kendall tests** are non-parametric and make no distributional assumptions, suitable for the sample size of 22 observations per country series.
- **Panel Granger causality** uses a pooled OLS F-test approach across countries at lags 1–3. Cross-sectional dependence is not explicitly modelled given the small panel.
- **Chow tests** are applied at three candidate break years (2008, 2014, 2020) corresponding to the global financial crisis, a mid-period structural transition, and the COVID-19 shock.
- **XAI decomposition** uses a Gradient Boosting Regressor (`n_estimators=200`, `max_depth=3`, `learning_rate=0.05`) with permutation importance computed over 100 repetitions. Results are treated as illustrative only given N=110.
- Japan is treated throughout as a post-transition comparator expected to exhibit attenuated $q_A$–$q_K$ coupling and stationary constraint dynamics.

---

## Citation

If you use this code or data in your research, please cite the manuscript:

> [Author]. (2026). A Variational Framework for Socio-Ecological Dynamics under Ecological Constraint Landscapes. *Ecological Modelling*. [DOI to be assigned]

---

## License

Data are sourced from the World Bank Open Data platform and are subject to the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). Code is released under the [MIT License](https://opensource.org/licenses/MIT).

---

## Contact

Correspondence regarding the code or data should be directed to the corresponding author. Requests for the full replication package including the LaTeX manuscript source may be made via the journal's data availability statement.
