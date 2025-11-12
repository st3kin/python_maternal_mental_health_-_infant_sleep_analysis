import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
from scipy import stats

# Loading the data

participant_df = pd.read_csv('CSV_files/participant.csv')
mental_health_df = pd.read_csv('CSV_files/mental_health.csv')


# Is there a correlation between babies' sleep fragmentation (nightly wakes) and poorer scores on their IBQ-R results?

ibq_cols = ['ibq_3','ibq_4','ibq_9','ibq_10','ibq_16','ibq_17','ibq_28','ibq_29','ibq_32','ibq_33']

ibq_df = participant_df[['participant_number', 'infant_wakes_per_night']].merge(
    mental_health_df[['participant_number', *ibq_cols]],
    on='participant_number', how='inner')

# Using Spearman's rank correlation

correlations = []

for col in ibq_cols:
    x = ibq_df['infant_wakes_per_night']
    y = ibq_df[col]

    valid = (~x.isna()) & (~y.isna())
    if valid.sum() > 1:
        rho, p = stats.spearmanr(x[valid], y[valid])
    else:
        rho, p = (np.nan, np.nan)
    
    correlations.append({'item': col, 'rho': rho, 'p': p})

corr_df = pd.DataFrame(correlations)
corr_df['significant'] = corr_df['p'] < 0.05
print(corr_df.sort_values('rho'))


'''A Spearman’s rank correlation analysis found significant positive associations between the number of infant nightly 
awakenings and several IBQ-R items related to distress and emotional reactivity (ρ = 0.18–0.35, all p < .05). Infants who woke 
more frequently at night tended to show higher levels of crying, anger, and clinginess, suggesting that poorer sleep continuity 
may be associated with greater negative affectivity in infancy.'''

# Visualisation

ibq_df['ibq_mean'] = ibq_df[ibq_cols].mean(axis=1)

plt.figure(figsize=(8, 5))
sns.regplot(
    data=ibq_df,
    x='infant_wakes_per_night',
    y='ibq_mean',
    scatter_kws={'alpha': 0.6, 's': 40},
    line_kws={'color': 'black', 'lw': 1.5},
    color=sns.color_palette('viridis', 10)[3]
)
plt.title('Correlation Between Infant Nightly Wakes and IBQ-R Mean Score')
plt.xlabel('Number of nightly wakes')
plt.ylabel('Mean IBR-R score (1-7 likert scale)')
plt.tight_layout()
plt.show()


# Answers are ordinal (Likert 1–7)

#Ten items from the Infant Behaviour Questionnaire (higher = more frequent behaviour)