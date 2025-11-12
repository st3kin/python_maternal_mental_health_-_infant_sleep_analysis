import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Loading the data

participant_df = pd.read_csv('CSV_files/participant.csv')
mental_health_df = pd.read_csv('CSV_files/mental_health.csv')

# Do mothers whose infants wake more frequently at night report higher CBTS, HADS, or EPDS scores?

cbts_cols = [c for c in mental_health_df.columns if c.startswith('cbts_')]
epds_cols = [c for c in mental_health_df.columns if c.startswith('epds_')]
hads_cols = [c for c in mental_health_df.columns if c.startswith('hads_')]

nw_df = participant_df[['participant_number', 'infant_wakes_per_night']].merge(
    mental_health_df[['participant_number', *cbts_cols, *epds_cols, *hads_cols]],
    on='participant_number',
    how='left'
)

nw_df['cbts_total'] = nw_df[cbts_cols].sum(axis=1, skipna=True)
nw_df['epds_total'] = nw_df[epds_cols].sum(axis=1, skipna=True)
nw_df['hads_total'] = nw_df[hads_cols].sum(axis=1, skipna=True)

# Visualising by nightly wakes

nw_long = nw_df.melt(
    id_vars='infant_wakes_per_night',
    value_vars=['cbts_total', 'epds_total', 'hads_total'],
    var_name='scale',
    value_name='score'
)

sns.lmplot(
    data=nw_long,
    x='infant_wakes_per_night', y='score',
    hue='scale',
    palette='viridis',
    scatter_kws={'alpha':0.5, 's':30},
    line_kws={'lw':2},
    lowess=True,
    height=5, aspect=1.3
)
plt.xlabel('Number of nightly wakes')
plt.ylabel('Mental health scores (higher = worse symptoms)')
plt.title("Infant nightly wakes vs CBTS, EPDS & HADS scores")
plt.tight_layout()
plt.show()

# Conducting a Spearman Correlation Coefficient

scales = ['cbts_total', 'epds_total', 'hads_total']
for s in scales:
    valid = nw_df[['infant_wakes_per_night', s]].dropna()
    rho, p = stats.spearmanr(valid['infant_wakes_per_night'], valid[s])
    print(f"Spearman: nightly wakes vs {s.upper()} -> ρ={rho:.3f}, p={p:.5f}")
    

'''A Spearman rank correlation showed a small, statistically significant positive relationship 
between infant nightly wakes and maternal depression (EPDS) scores (ρ = 0.11, p = 0.031). No significant 
correlations were observed for postpartum trauma (CBTS) or anxiety (HADS) measures. This suggests that while frequent 
infant waking may contribute modestly to maternal depressive symptoms, its impact on broader mental 
health appears limited in this sample.'''

