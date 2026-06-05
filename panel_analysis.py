"""
=============================================================================
PANEL EMPIRICAL ANALYSIS — VARIATIONAL SUSTAINABILITY FRAMEWORK
=============================================================================
Manuscript : Sustainability as a Variational Problem (Sus-Science)
Data       : World Bank WDI — 5 countries × 5 indicators × 2000–2021
Countries  : Brazil, China, India, Indonesia, Japan
Indicators :
    forest_area  — Forest area (% of land area)         [AG.LND.FRST.ZS]
    agri_co2     — Agri CO2 emissions (Mt CO2e)          [EN.GHG.CO2.AG.MT.CE.AR5]
    agri_gdp     — Agriculture value added (% of GDP)   [NV.AGR.TOTL.ZS]
    energy_dep   — Energy depletion (% of GNI)          [NY.ADJ.DNGY.GN.ZS]
    urban_pop    — Urban population                      [SP.URB.TOTL]

Theoretical mapping:
    q_A proxy    : agri_co2 + energy_dep  (material accumulation / depletion)
    q_K proxy    : urban_pop + (1 - agri_gdp)  (consumption / structural shift)
    V_D proxy    : forest_area  (ecological constraint landscape)

Research question:
    Do observed trajectories exhibit dynamical properties — bounded evolution,
    directional adjustment, coupling structure, structural breaks — consistent
    with a dissipative variational system under a non-convex constraint landscape?

Analysis steps:
    1. Data loading and operationalisation
    2. Descriptive phase-space trajectories
    3. Mann-Kendall trend tests (per country, per indicator)
    4. Cross-correlation and coupling analysis (q_A vs q_K)
    5. Panel Granger causality (Dumitrescu-Hurlin style, manual)
    6. Structural break analysis (Chow tests)
    7. XAI illustrative decomposition (Gradient Boosting + Permutation Importance)
    8. Summary visualisation

NOTE: XAI results are illustrative only — N=110 is modest for ML methods.
      All inferential claims are based on steps 3–6.
=============================================================================
"""

# ── 0. IMPORTS ────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

# ── 1. CONFIGURATION ──────────────────────────────────────────────────────────
INPUT_FILE   = 'panel_final.csv'   # Input: balanced panel CSV
OUTPUT_DIR   = './'                 # Output directory for figures
COUNTRIES    = ['Brazil', 'China', 'India', 'Indonesia', 'Japan']
YEARS        = list(range(2000, 2022))

# Country aesthetics — consistent across all plots
COUNTRY_COLORS  = {
    'Brazil':    '#1a9641',
    'China':     '#d7191c',
    'India':     '#ff7f00',
    'Indonesia': '#984ea3',
    'Japan':     '#377eb8',
}
COUNTRY_MARKERS = {
    'Brazil': 'o', 'China': 's', 'India': '^', 'Indonesia': 'D', 'Japan': 'v'
}

# Theoretical role labels
THEORY_LABELS = {
    'forest_area': '$V_D$ proxy — Constraint landscape',
    'agri_co2':    '$q_A$ proxy — Accumulation pressure',
    'agri_gdp':    '$q_K$ proxy (inverse) — Structural shift',
    'energy_dep':  '$q_A$ proxy — Depletion intensity',
    'urban_pop':   '$q_K$ proxy — Urbanisation shift',
}

# ── 2. HELPER FUNCTIONS ───────────────────────────────────────────────────────

def minmax_norm(series):
    """Min-max normalisation to [0, 1]."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return series * 0.0
    return (series - mn) / (mx - mn)


def mann_kendall(x):
    """
    Mann-Kendall monotonic trend test.
    Returns: S statistic, Z score, p-value, Kendall tau, trend direction, significance.
    Non-parametric — no distributional assumptions. Suitable for small samples.
    """
    x = np.array(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    s = sum(np.sign(x[j] - x[i]) for i in range(n-1) for j in range(i+1, n))
    var_s = n * (n-1) * (2*n+5) / 18
    z = (s-1)/np.sqrt(var_s) if s > 0 else ((s+1)/np.sqrt(var_s) if s < 0 else 0)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    tau = s / (0.5 * n * (n-1))
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    return {'S': s, 'Z': round(z,4), 'p': round(p,4),
            'tau': round(tau,4), 'trend': 'Inc' if z>0 else 'Dec', 'sig': sig}


def chow_test(y, x, bp_idx):
    """
    Chow test for structural break in linear regression y ~ x at index bp_idx.
    Tests H0: no structural break (same intercept and slope in both sub-samples).
    Returns: F statistic, p-value.
    """
    n = len(y)
    X_full = np.column_stack([np.ones(n), x])
    b_full = np.linalg.lstsq(X_full, y, rcond=None)[0]
    RSS_full = np.sum((y - X_full @ b_full)**2)
    k = 2
    def ols_rss(yy, xx):
        X = np.column_stack([np.ones(len(yy)), xx])
        b = np.linalg.lstsq(X, yy, rcond=None)[0]
        return np.sum((yy - X @ b)**2)
    RSS_split = ols_rss(y[:bp_idx], x[:bp_idx]) + ols_rss(y[bp_idx:], x[bp_idx:])
    F = ((RSS_full - RSS_split)/k) / (RSS_split/(n - 2*k))
    p = 1 - stats.f.cdf(F, k, n - 2*k)
    return round(F, 4), round(p, 4)


def panel_granger(panel_df, cause_col, effect_col, lag=1):
    """
    Panel Granger causality test (pooled OLS F-test approach).
    Tests H0: cause_col does NOT Granger-cause effect_col across the panel.
    For each country, constructs lagged variables and pools across countries.
    Returns: F statistic, p-value, interpretation string.
    """
    restricted, unrestricted = [], []
    for country in panel_df['country'].unique():
        sub = panel_df[panel_df['country']==country].sort_values('year')
        y   = sub[effect_col].values
        x   = sub[cause_col].values
        if len(y) <= lag + 2:
            continue
        y_dep  = y[lag:]
        y_lag  = y[lag-1:-1]
        x_lag  = x[lag-1:-1]
        n_eff  = len(y_dep)
        # Restricted: y ~ y_lag
        Xr = np.column_stack([np.ones(n_eff), y_lag])
        br = np.linalg.lstsq(Xr, y_dep, rcond=None)[0]
        restricted.append(np.sum((y_dep - Xr @ br)**2))
        # Unrestricted: y ~ y_lag + x_lag
        Xu = np.column_stack([np.ones(n_eff), y_lag, x_lag])
        bu = np.linalg.lstsq(Xu, y_dep, rcond=None)[0]
        unrestricted.append(np.sum((y_dep - Xu @ bu)**2))
    RSS_r = sum(restricted)
    RSS_u = sum(unrestricted)
    N_c   = len(restricted)
    k, df2 = 1, N_c * (len(y_dep) - 3)
    F = ((RSS_r - RSS_u)/k) / (RSS_u/df2) if df2 > 0 else np.nan
    p = 1 - stats.f.cdf(F, k, df2) if not np.isnan(F) else np.nan
    sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
    return round(F,4), round(p,4), sig


# ── 3. LOAD DATA & BUILD COMPOSITES ───────────────────────────────────────────
print("="*70)
print("LOADING DATA AND BUILDING COMPOSITE PROXIES")
print("="*70)

df = pd.read_csv(INPUT_FILE)
df = df[df['year'].isin(YEARS)].sort_values(['country','year']).reset_index(drop=True)

print(f"Panel shape: {df.shape}  |  Countries: {df['country'].nunique()}  |  "
      f"Years: {df['year'].min()}–{df['year'].max()}")
print(f"Missing values: {df.isnull().sum().sum()}")

# Normalise per country (within-country min-max) for composite construction
df['urban_pop_m'] = df['urban_pop'] / 1e6  # convert to millions

for col in ['agri_co2','energy_dep','urban_pop_m','agri_gdp','forest_area']:
    df[f'{col}_n'] = df.groupby('country')[col].transform(minmax_norm)

# Composite proxies (within-country normalised)
df['qA'] = (df['agri_co2_n'] + df['energy_dep_n']) / 2
df['qK'] = (df['urban_pop_m_n'] + (1 - df['agri_gdp_n'])) / 2

print("\nOperationalisation mapping:")
print("  q_A = (norm(agri_co2) + norm(energy_dep)) / 2")
print("  q_K = (norm(urban_pop) + (1 - norm(agri_gdp))) / 2")
print("  V_D = forest_area (% of land)")


# ── 4. STEP 2 — MANN-KENDALL TREND TESTS ─────────────────────────────────────
print("\n" + "="*70)
print("STEP 2 — MANN-KENDALL TREND TESTS")
print("="*70)
print(f"\n{'Country':<12} {'Indicator':<35} {'τ':>7} {'Z':>8} {'p':>8} {'Sig':>5} {'Dir':>5}")
print("-"*80)

mk_results = {}
for country in COUNTRIES:
    sub = df[df['country']==country].sort_values('year')
    for col, label in [('forest_area','Forest Area (VD)'),
                       ('agri_co2',   'Agri CO2 (qA)'),
                       ('agri_gdp',   'Agri GDP share (qK inv)'),
                       ('energy_dep', 'Energy Depletion (qA)'),
                       ('urban_pop_m','Urban Population (qK)')]:
        r = mann_kendall(sub[col].values)
        mk_results[(country, col)] = r
        print(f"{country:<12} {label:<35} {r['tau']:>7.4f} {r['Z']:>8.4f} "
              f"{r['p']:>8.4f} {r['sig']:>5} {r['trend']:>5}")
    print()

print("Significance: *** p<0.001  ** p<0.01  * p<0.05  ns = not significant")


# ── 5. STEP 3 — CROSS-CORRELATION: q_A vs q_K ────────────────────────────────
print("\n" + "="*70)
print("STEP 3 — CROSS-CORRELATION ANALYSIS: q_A vs q_K")
print("="*70)

ccf_results = {}
for country in COUNTRIES:
    sub = df[df['country']==country].sort_values('year')
    qA_c = sub['qA'].values
    qK_c = sub['qK'].values
    n = len(qA_c)
    print(f"\n  {country}:")
    print(f"  {'Lag':>5} {'r':>8} {'p':>8} {'Sig':>5}  Direction")
    print(f"  {'-'*50}")
    for lag in range(0, 6):
        if lag == 0:
            r, p = pearsonr(qA_c, qK_c)
            direction = "Contemporaneous"
        else:
            r, p = pearsonr(qA_c[:-lag], qK_c[lag:])
            direction = f"qA leads qK by {lag}yr"
        sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
        ccf_results[(country, lag)] = (r, p, sig)
        print(f"  {lag:>5} {r:>8.4f} {p:>8.4f} {sig:>5}  {direction}")


# ── 6. STEP 4 — PANEL GRANGER CAUSALITY ──────────────────────────────────────
print("\n" + "="*70)
print("STEP 4 — PANEL GRANGER CAUSALITY")
print("="*70)
print("\nH1: q_A Granger-causes Forest Area (V_D)")
for lag in [1, 2, 3]:
    F, p, sig = panel_granger(df, 'qA', 'forest_area', lag)
    print(f"  Lag {lag}: F={F:.4f}, p={p:.4f} {sig}")

print("\nH2: Forest Area Granger-causes q_A (reverse)")
for lag in [1, 2, 3]:
    F, p, sig = panel_granger(df, 'forest_area', 'qA', lag)
    print(f"  Lag {lag}: F={F:.4f}, p={p:.4f} {sig}")

print("\nH3: q_A Granger-causes q_K")
for lag in [1, 2, 3]:
    F, p, sig = panel_granger(df, 'qA', 'qK', lag)
    print(f"  Lag {lag}: F={F:.4f}, p={p:.4f} {sig}")

print("\nH4: q_K Granger-causes q_A (reverse)")
for lag in [1, 2, 3]:
    F, p, sig = panel_granger(df, 'qK', 'qA', lag)
    print(f"  Lag {lag}: F={F:.4f}, p={p:.4f} {sig}")


# ── 7. STEP 5 — STRUCTURAL BREAK ANALYSIS ────────────────────────────────────
print("\n" + "="*70)
print("STEP 5 — STRUCTURAL BREAK ANALYSIS (Chow Test)")
print("="*70)
print(f"\n{'Country':<12} {'Break Year':>12} {'Series':>12} {'F':>10} {'p':>10} {'Sig':>5}")
print("-"*65)

chow_results = {}
break_years  = [2008, 2014, 2020]
for country in COUNTRIES:
    sub  = df[df['country']==country].sort_values('year')
    t    = np.arange(len(sub))
    years_arr = sub['year'].values
    for series, label in [('qA','qA'), ('qK','qK')]:
        y = sub[series].values
        for byr in break_years:
            idx = np.where(years_arr == byr)[0]
            if len(idx) == 0 or idx[0] < 3 or idx[0] > len(y)-3:
                continue
            F, p = chow_test(y, t, idx[0])
            sig  = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
            chow_results[(country, byr, series)] = (F, p, sig)
            print(f"{country:<12} {byr:>12} {label:>12} {F:>10.4f} {p:>10.4f} {sig:>5}")
    print()


# ── 8. STEP 6 — XAI ILLUSTRATIVE DECOMPOSITION ───────────────────────────────
print("="*70)
print("STEP 6 — XAI ILLUSTRATIVE DECOMPOSITION (Gradient Boosting + Permutation)")
print("="*70)
print("Target  : Forest Area (V_D proxy)")
print("Features: agri_co2, agri_gdp, energy_dep, urban_pop_m")
print("NOTE    : Illustrative only — N=110 is modest for ML methods\n")

feat_cols  = ['agri_co2','agri_gdp','energy_dep','urban_pop_m']
feat_names = ['Agri CO₂ (qA)', 'Agri GDP share (qK⁻¹)',
              'Energy Depletion (qA)', 'Urban Pop (qK)']

X = df[feat_cols].values
y = df['forest_area'].values

# Scale features
scaler = StandardScaler()
X_sc   = scaler.fit_transform(X)

model = GradientBoostingRegressor(
    n_estimators=200, max_depth=3, learning_rate=0.05,
    subsample=0.8, random_state=42
)
model.fit(X_sc, y)
r2 = model.score(X_sc, y)
print(f"In-sample R² = {r2:.4f}  (inflated with N=110 — interpret directionally only)\n")

# Impurity-based importance
imp = model.feature_importances_
print("Impurity-based Feature Importance:")
for name, i in sorted(zip(feat_names, imp), key=lambda x: -x[1]):
    print(f"  {name:<35}: {i:.4f}")

# Permutation importance
perm = permutation_importance(model, X_sc, y, n_repeats=100, random_state=42)
print("\nPermutation Feature Importance (mean ± std):")
for name, m, s in sorted(zip(feat_names, perm.importances_mean,
                               perm.importances_std), key=lambda x: -x[1]):
    print(f"  {name:<35}: {m:.4f} ± {s:.4f}")

# Partial dependence direction
print("\nPartial Dependence — direction of effect on Forest Area:")
for j, name in enumerate(feat_names):
    x_range = np.linspace(X_sc[:,j].min(), X_sc[:,j].max(), 30)
    pd_vals = []
    for xv in x_range:
        Xc = X_sc.copy(); Xc[:,j] = xv
        pd_vals.append(model.predict(Xc).mean())
    slope = np.polyfit(x_range, pd_vals, 1)[0]
    direction = "↑ positive" if slope > 0 else "↓ negative"
    print(f"  {name:<35}: {direction}  (slope={slope:.4f})")

# Country-specific XAI
print("\nCountry-specific feature importances (permutation, top feature per country):")
for country in COUNTRIES:
    sub  = df[df['country']==country]
    Xc   = sub[feat_cols].values
    yc   = sub['forest_area'].values
    Xc_s = scaler.transform(Xc)
    mc   = GradientBoostingRegressor(n_estimators=100, max_depth=2,
                                      learning_rate=0.1, random_state=42)
    mc.fit(Xc_s, yc)
    pc = permutation_importance(mc, Xc_s, yc, n_repeats=50, random_state=42)
    top_idx  = np.argmax(pc.importances_mean)
    top_name = feat_names[top_idx]
    print(f"  {country:<12}: top driver = {top_name}  "
          f"(imp={pc.importances_mean[top_idx]:.4f})")


# ── 9. FIGURES ────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("GENERATING FIGURES")
print("="*70)

# ── Figure 1: Individual time series per indicator, all countries ─────────────
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle("Individual Indicator Trajectories — 5 Countries, 2000–2021\n"
             "Theoretical roles noted per panel",
             fontsize=13, fontweight='bold', y=0.98)

plot_vars = [
    ('forest_area', 'Forest Area (% land)', '$V_D$ — Constraint landscape'),
    ('agri_co2',    'Agri CO₂ (Mt CO₂e)',   '$q_A$ — Accumulation pressure'),
    ('agri_gdp',    'Agri value added (% GDP)', '$q_K$ proxy (inverse)'),
    ('energy_dep',  'Energy Depletion (% GNI)', '$q_A$ — Depletion intensity'),
    ('urban_pop_m', 'Urban Population (millions)', '$q_K$ — Urbanisation shift'),
]

for ax, (col, ylabel, theory) in zip(axes.flat[:5], plot_vars):
    for country in COUNTRIES:
        sub = df[df['country']==country].sort_values('year')
        ax.plot(sub['year'], sub[col],
                color=COUNTRY_COLORS[country],
                marker=COUNTRY_MARKERS[country],
                markersize=3, linewidth=1.8, label=country)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlabel('Year', fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.3, linestyle='--')
    ax.text(0.97, 0.05, theory, transform=ax.transAxes,
            fontsize=8, ha='right', va='bottom', style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    for byr, ls in [(2008,'--'),(2020,':')]:
        ax.axvline(byr, color='gray', linestyle=ls, linewidth=0.8, alpha=0.6)

handles = [Line2D([0],[0], color=COUNTRY_COLORS[c], marker=COUNTRY_MARKERS[c],
                  linewidth=1.8, markersize=5, label=c) for c in COUNTRIES]
axes.flat[5].axis('off')
axes.flat[5].legend(handles=handles, loc='center', fontsize=11, frameon=True,
                    title='Countries', title_fontsize=11)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}fig1_individual_series.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Figure 1 saved: Individual time series")

# ── Figure 2: Phase-space trajectories per country ───────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Phase-Space Trajectories: $q_A$ vs $q_K$ — 2000–2021\n"
             "Arrow direction = time; colour = year",
             fontsize=13, fontweight='bold')

for ax, country in zip(axes.flat[:5], COUNTRIES):
    sub = df[df['country']==country].sort_values('year')
    qA_c = sub['qA'].values
    qK_c = sub['qK'].values
    yrs  = sub['year'].values
    sc   = ax.scatter(qA_c, qK_c, c=yrs, cmap='plasma', s=60, zorder=5)
    for i in range(len(qA_c)-1):
        ax.annotate('', xy=(qA_c[i+1], qK_c[i+1]), xytext=(qA_c[i], qK_c[i]),
                    arrowprops=dict(arrowstyle='->', color=COUNTRY_COLORS[country],
                                   lw=1.2))
    for yr in [2000, 2008, 2014, 2021]:
        mask = sub['year'] == yr
        if mask.any():
            ax.annotate(str(yr),
                        (sub.loc[mask,'qA'].values[0], sub.loc[mask,'qK'].values[0]),
                        textcoords='offset points', xytext=(5,4), fontsize=7)
    plt.colorbar(sc, ax=ax, label='Year', shrink=0.8)
    ax.set_title(country, fontsize=11, fontweight='bold',
                 color=COUNTRY_COLORS[country])
    ax.set_xlabel('$q_A$ composite', fontsize=9)
    ax.set_ylabel('$q_K$ composite', fontsize=9)
    ax.grid(alpha=0.3, linestyle='--')

axes.flat[5].axis('off')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}fig2_phase_space.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Figure 2 saved: Phase-space trajectories")

# ── Figure 3: Cross-correlogram panel ────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Cross-Correlogram: $q_A$ ↔ $q_K$ by Country (Lags 0–5)\n"
             "Positive lag = $q_A$ leads $q_K$; dashed lines = 95% CI",
             fontsize=12, fontweight='bold')
ci = 1.96 / np.sqrt(22)

for ax, country in zip(axes.flat[:5], COUNTRIES):
    sub  = df[df['country']==country].sort_values('year')
    qA_c = sub['qA'].values
    qK_c = sub['qK'].values
    lags, rvals = [], []
    for lag in range(-5, 6):
        if lag == 0:
            r, _ = pearsonr(qA_c, qK_c)
        elif lag > 0:
            r, _ = pearsonr(qA_c[:-lag], qK_c[lag:])
        else:
            r, _ = pearsonr(qK_c[:-abs(lag)], qA_c[abs(lag):])
        lags.append(lag); rvals.append(r)
    colors = [COUNTRY_COLORS[country] if abs(v) > ci else '#cccccc' for v in rvals]
    ax.bar(lags, rvals, color=colors, edgecolor='white', linewidth=0.3)
    ax.axhline(0,   color='black',  linewidth=0.8)
    ax.axhline(ci,  color='gray',   linestyle='--', linewidth=1)
    ax.axhline(-ci, color='gray',   linestyle='--', linewidth=1)
    ax.set_title(country, fontsize=10, fontweight='bold',
                 color=COUNTRY_COLORS[country])
    ax.set_xlabel('Lag (years)', fontsize=8)
    ax.set_ylabel('Pearson r', fontsize=8)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(alpha=0.3, linestyle='--', axis='y')

axes.flat[5].axis('off')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}fig3_cross_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Figure 3 saved: Cross-correlograms")

# ── Figure 4: Structural breaks heatmap ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Structural Break Analysis — Chow Test p-values\n"
             "Cells show p-value; shading indicates significance",
             fontsize=12, fontweight='bold')

for ax, series_label in zip(axes, [('qA','$q_A$ composite'), ('qK','$q_K$ composite')]):
    series_key = series_label[0]
    matrix = np.ones((len(COUNTRIES), len(break_years)))
    for i, country in enumerate(COUNTRIES):
        for j, byr in enumerate(break_years):
            key = (country, byr, series_key)
            if key in chow_results:
                matrix[i,j] = chow_results[key][1]
    im = ax.imshow(matrix, cmap='RdYlGn_r', vmin=0, vmax=0.1, aspect='auto')
    ax.set_xticks(range(len(break_years)))
    ax.set_xticklabels(break_years, fontsize=10)
    ax.set_yticks(range(len(COUNTRIES)))
    ax.set_yticklabels(COUNTRIES, fontsize=10)
    ax.set_title(f'Breaks in {series_label[1]}', fontsize=11)
    plt.colorbar(im, ax=ax, label='p-value')
    for i in range(len(COUNTRIES)):
        for j in range(len(break_years)):
            key = (COUNTRIES[i], break_years[j], series_key)
            if key in chow_results:
                F_val, p_val, sig = chow_results[key]
                ax.text(j, i, f'{p_val:.3f}\n{sig}', ha='center', va='center',
                        fontsize=8, color='black' if p_val > 0.03 else 'white')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}fig4_structural_breaks.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Figure 4 saved: Structural break heatmap")

# ── Figure 5: XAI feature importance ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("XAI Illustrative Decomposition: Drivers of Constraint Landscape ($V_D$)\n"
             "Target = Forest Area; Features = $q_A$ and $q_K$ proxies\n"
             "Note: illustrative only — N=110",
             fontsize=11, fontweight='bold')

xai_colors = ['#c0392b','#2980b9','#e67e22','#27ae60']

sorted_idx = np.argsort(imp)
axes[0].barh([feat_names[i] for i in sorted_idx], imp[sorted_idx],
             color=[xai_colors[i] for i in sorted_idx], edgecolor='white')
axes[0].set_xlabel('Impurity-based Importance', fontsize=10)
axes[0].set_title('Impurity-based\n(MDI)', fontsize=11)
axes[0].grid(axis='x', alpha=0.3, linestyle='--')

sorted_idx2 = np.argsort(perm.importances_mean)
axes[1].barh([feat_names[i] for i in sorted_idx2],
             perm.importances_mean[sorted_idx2],
             xerr=perm.importances_std[sorted_idx2],
             color=[xai_colors[i] for i in sorted_idx2],
             edgecolor='white', capsize=4)
axes[1].set_xlabel('Permutation Importance (mean ± std)', fontsize=10)
axes[1].set_title('Permutation-based\n(more robust)', fontsize=11)
axes[1].axvline(0, color='black', linewidth=0.8)
axes[1].grid(axis='x', alpha=0.3, linestyle='--')

for ax in axes:
    ax.text(0.97, 0.03, 'Illustrative only\n(N=110)', transform=ax.transAxes,
            fontsize=8, color='gray', ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}fig5_xai_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Figure 5 saved: XAI feature importance")

# ── Figure 6: Mann-Kendall summary heatmap ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Mann-Kendall Trend Analysis — Kendall τ and Significance\n"
             "Expected directions per theoretical role noted below",
             fontsize=12, fontweight='bold')

mk_vars = ['forest_area','agri_co2','agri_gdp','energy_dep','urban_pop_m']
mk_var_labels = ['Forest Area\n(VD)', 'Agri CO₂\n(qA)',
                 'Agri GDP\n(qK inv)', 'Energy Dep\n(qA)', 'Urban Pop\n(qK)']
expected = ['↑ ambiguous', '↑ expected', '↓ expected', '↑ expected', '↑ expected']

tau_matrix = np.zeros((len(COUNTRIES), len(mk_vars)))
sig_matrix = np.empty((len(COUNTRIES), len(mk_vars)), dtype=object)

for i, country in enumerate(COUNTRIES):
    for j, col in enumerate(mk_vars):
        r = mk_results.get((country, col), {})
        tau_matrix[i,j] = r.get('tau', 0)
        sig_matrix[i,j] = r.get('sig','ns')

im = axes[0].imshow(tau_matrix, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
axes[0].set_xticks(range(len(mk_vars)))
axes[0].set_xticklabels(mk_var_labels, fontsize=8)
axes[0].set_yticks(range(len(COUNTRIES)))
axes[0].set_yticklabels(COUNTRIES, fontsize=10)
axes[0].set_title("Kendall τ values", fontsize=11)
plt.colorbar(im, ax=axes[0], label='τ')
for i in range(len(COUNTRIES)):
    for j in range(len(mk_vars)):
        tau_val = tau_matrix[i,j]
        sig_val = sig_matrix[i,j]
        axes[0].text(j, i, f'{tau_val:.2f}\n{sig_val}',
                     ha='center', va='center', fontsize=7,
                     color='white' if abs(tau_val) > 0.5 else 'black')

# Consistency check
consistent = np.zeros((len(COUNTRIES), len(mk_vars)))
exp_dirs = [None, 1, -1, 1, 1]
for i, country in enumerate(COUNTRIES):
    for j, (col, exp) in enumerate(zip(mk_vars, exp_dirs)):
        if exp is None:
            consistent[i,j] = 0.5
        else:
            tau_val = tau_matrix[i,j]
            sig_val = sig_matrix[i,j]
            if sig_val == 'ns':
                consistent[i,j] = 0.5
            elif np.sign(tau_val) == exp:
                consistent[i,j] = 1.0
            else:
                consistent[i,j] = 0.0

im2 = axes[1].imshow(consistent, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
axes[1].set_xticks(range(len(mk_vars)))
axes[1].set_xticklabels(mk_var_labels, fontsize=8)
axes[1].set_yticks(range(len(COUNTRIES)))
axes[1].set_yticklabels(COUNTRIES, fontsize=10)
axes[1].set_title("Theoretical consistency\n(green=consistent, red=inconsistent, yellow=ns)",
                   fontsize=10)
for j, exp_label in enumerate(expected):
    axes[1].text(j, len(COUNTRIES)+0.1, exp_label, ha='center', va='top',
                 fontsize=7, style='italic')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}fig6_mann_kendall_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Figure 6 saved: Mann-Kendall heatmap")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print(f"""
Output files:
  panel_final.csv              — Clean balanced panel dataset
  fig1_individual_series.png   — Individual indicator trajectories
  fig2_phase_space.png         — Phase-space trajectories (qA vs qK)
  fig3_cross_correlation.png   — Cross-correlograms by country
  fig4_structural_breaks.png   — Chow test p-value heatmap
  fig5_xai_importance.png      — XAI feature importance
  fig6_mann_kendall_heatmap.png — Mann-Kendall τ and consistency

Japan note: interpreted as post-transition comparator throughout.
All XAI results labelled illustrative only (N=110).
""")
