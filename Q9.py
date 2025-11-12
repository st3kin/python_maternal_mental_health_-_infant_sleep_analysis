import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Loading the data

participant_df = pd.read_csv('CSV_files/participant.csv')
mental_health_df = pd.read_csv('CSV_files/mental_health.csv')


# Are younger mothers more likely to experience postpartum trauma or depression (higher CBTS & EPDS scores)?

cbts_cols = [c for c in mental_health_df.columns if c.startswith('cbts_')]
epds_cols = [c for c in mental_health_df.columns if c.startswith('epds_')]

agepp_df = participant_df[['participant_number', 'age']].merge(
    mental_health_df[['participant_number', *cbts_cols, *epds_cols]],
    on='participant_number',
    how='left'
)

agepp_df['cbts_total'] = agepp_df[cbts_cols].sum(axis=1, skipna=True)
agepp_df['epds_total'] = agepp_df[epds_cols].sum(axis=1, skipna=True)


# Visualising by age

agepp_long = agepp_df.melt(
    id_vars='age',
    value_vars=['cbts_total', 'epds_total'],
    var_name='scale',
    value_name='score'
)

sns.lmplot(
    data=agepp_long,
    x='age', y='score', hue='scale',
    palette='viridis', height=6, aspect=1.2,
    scatter_kws={'alpha':0.6, 's':30}, ci=None
)
plt.title('Maternal Age vs CBTS and EPDS Scores (with Trendlines)')
plt.xlabel('Age')
plt.ylabel('Score (higher = worse symptoms)')
plt.tight_layout()
plt.show()


# Statistical analysis using Spearman's rank correlation

rho_cbts, p_cbts = stats.spearmanr(agepp_df['age'], agepp_df['cbts_total'])
rho_epds, p_epds = stats.spearmanr(agepp_df['age'], agepp_df['epds_total'])

print(f"Spearman: age vs CBTS -> ρ={rho_cbts:.3f}, p={p_cbts:.4g}")
print(f"Spearman: age vs EPDS -> ρ={rho_epds:.3f}, p={p_epds:.4g}")


'''Both correlation coefficients are negative and small, meaning older mothers show slightly lower trauma 
and depression scores, but the effect is weak and not statistically significant. The near-significant 
EPDS result (p≈0.06) could hint at a subtle pattern; younger mothers might experience somewhat higher depressive 
symptoms, but not strong enough to draw firm conclusions.'''

