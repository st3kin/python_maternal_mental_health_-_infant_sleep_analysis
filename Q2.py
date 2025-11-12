import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
from scipy import stats

# Loading the data

participant_df = pd.read_csv('CSV_files/participant.csv')
mental_health_df = pd.read_csv('CSV_files/mental_health.csv')


# Is there a correlation between a mother's marital status and worse PPD symptoms (high cbts scores)?

'''Testing whether marital status (a proxy for social or emotional support) is associated 
with CBTS scores (postpartum trauma symptoms).'''

cbts_cols = [c for c in mental_health_df.columns if c.startswith('cbts_')]

cbts_df = participant_df[['participant_number', 'marital_status']].merge(
    mental_health_df[['participant_number', *cbts_cols]],
    on='participant_number',
    how='left'
)

cbts_df['cbts_total'] = cbts_df[cbts_cols].sum(axis=1, skipna=True)
cbts_df = cbts_df.dropna(subset=['marital_status', 'cbts_total'])

print(cbts_df['marital_status'].value_counts()) 
print(cbts_df['cbts_total'].describe())

# Visualising by marital status

plt.figure(figsize=(8, 5))
sns.boxplot(data=cbts_df, x='marital_status', y='cbts_total', palette='mako')
sns.stripplot(data=cbts_df, x='marital_status', y='cbts_total', color='black', alpha=0.5, size=3)
plt.title('City Birth Trauma Scale (CBTS) Scores by Marital Status')
plt.xlabel('Marital status')
plt.ylabel('Total CBTS score (higher = worse postpartum trauma)')
plt.tight_layout()
plt.show()

# Statistical analysis

'''Since the “single” and “separated/divorced/widowed” groups contained very few participants (n=14 and n=7), I 
combined them into a single “Unpartnered” group. While this may hinder the granularity of the analysis, it will handle
the sample size imbalance better and provide better stability.'''

cbts_df['marital_group'] = cbts_df['marital_status'].replace({
    'Single': 'Unpartnered',
    'Separated, divorced or widowed': 'Unpartnered',
    'In a relationship': 'Partnered'
})

partnered = cbts_df.loc[cbts_df['marital_group'] == 'Partnered', 'cbts_total']
unpartnered = cbts_df.loc[cbts_df['marital_group'] == 'Unpartnered', 'cbts_total']

marital_U, marital_p = stats.mannwhitneyu(partnered, unpartnered, alternative='two-sided')
marital_r = abs(stats.norm.ppf(marital_p/2)) / (len(cbts_df)**0.5) # rough rank-biserial correlation approximation

print(f"Mann-Whitney U: U={marital_U:.2f}, p={marital_p:.5f}, r={marital_r:.3f}")

'''A Mann–Whitney U test comparing partnered (n = 389) and unpartnered (n = 21) mothers on their City Birth Trauma Scale 
(CBTS) scores found no significant difference (U = 4279, p = 0.714, r = 0.018). The effect size was negligible, suggesting 
that marital status alone was not associated with postpartum trauma severity in this sample. However, considering the severity
of the sample size imbalance, it would be reasonable to conclude that a lack of sufficient unpartnered participants could have
skewed the analysis.'''

plt.figure(figsize=(8, 5))
sns.boxplot(data=cbts_df, x='marital_group', y='cbts_total', palette='mako')
sns.stripplot(data=cbts_df, x='marital_group', y='cbts_total', color='black', alpha=0.5, size=3)
plt.title('City Birth Trauma Scale (CBTS) Scores by Partner Status')
plt.xlabel('Partner Status')
plt.ylabel('Total CBTS Score (higher = worse postpartum trauma)')
plt.tight_layout()
plt.show()