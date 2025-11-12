import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
from scipy import stats

# Loading the data

participant_df = pd.read_csv('CSV_files/participant.csv')
mental_health_df = pd.read_csv('CSV_files/mental_health.csv')

# Is there a correlation between mothers with PPD symptoms and distressed/restless infants (high CBTS/EPDS vs high IBQ-R scores)?

cbts_cols = [c for c in mental_health_df.columns if c.startswith('cbts_')]
epds_cols = [c for c in mental_health_df.columns if c.startswith('epds_')]
ibq_cols = [c for c in mental_health_df.columns if c.startswith('ibq_')]

mh_df = mental_health_df[['participant_number', *cbts_cols, *epds_cols, *ibq_cols]].copy()

mh_df['cbts_total'] = mh_df[cbts_cols].sum(axis=1, skipna=True)
mh_df['epds_total'] = mh_df[epds_cols].sum(axis=1, skipna=True)
mh_df['ibq_mean'] = mh_df[ibq_cols].mean(axis=1, skipna=True)
mh_df = mh_df.dropna(subset=['cbts_total', 'epds_total', 'ibq_mean'])

# Visualisation

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

sns.regplot(data=mh_df, x='cbts_total', y='ibq_mean', ax=axes[0],
            scatter_kws={'alpha':0.6, 's':35}, line_kws={'color': 'crimson'}, lowess=True)
axes[0].set_title('CBTS vs IBQ-R mean')
axes[0].set_xlabel('CBTS total score')
axes[0].set_ylabel('Infant distress (IBQ-R mean)')

sns.regplot(data=mh_df, x='epds_total', y='ibq_mean', ax=axes[1],
            scatter_kws={'alpha':0.6, 's':35}, line_kws={'color':'darkgreen'}, lowess=True)
axes[1].set_title('EPDS vs IBQ-R mean')
axes[1].set_xlabel('EPDS total score')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()


# Statistical analysis using Spearman's rank correlation

rho_cbts, p_cbts = stats.spearmanr(mh_df['cbts_total'], mh_df['ibq_mean'])
rho_epds, p_epds = stats.spearmanr(mh_df['epds_total'], mh_df['ibq_mean'])

print(f"Spearman: CBTS vs IBQ-R mean -> ρ={rho_cbts:.3f}, p={p_cbts:.5f}")
print(f"Spearman: EPDS vs IBQ-R mean -> ρ={rho_epds:.3f}, p={p_epds:.5f}")


'''Spearman rank correlations revealed that higher maternal postpartum trauma and depressive symptoms were associated with greater 
infant negative emotionality on the IBQ-R (ρ = .13, p = .008; ρ = .19, p < .001, respectively). These findings suggest that maternal 
mental health difficulties are modestly but significantly linked to infants’ temperament reactivity, underscoring the interplay 
between postpartum adjustment and early emotional development.'''
