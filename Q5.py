import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Loading the data

participant_df = pd.read_csv(
    'CSV_files/participant.csv',
    usecols=['participant_number', 'infant_gestational_age']
)

ibq_cols = ['ibq_3','ibq_4','ibq_9','ibq_10','ibq_16','ibq_17','ibq_28','ibq_29','ibq_32','ibq_33']
mental_health_df = pd.read_csv(
    'CSV_files/mental_health.csv',
    usecols=['participant_number', *ibq_cols]
)

# Is there a correlation between an infant's gestational age at birth and their IBQ-R scores?

ibq_df = participant_df[['participant_number', 'infant_gestational_age']].merge(
    mental_health_df[['participant_number', *ibq_cols]],
    on='participant_number', how='inner')

ibq_df['infant_gestational_age'] = pd.to_numeric(ibq_df['infant_gestational_age'], errors='coerce')
ibq_df[ibq_cols] = ibq_df[ibq_cols].apply(pd.to_numeric, errors='coerce')

ibq_df = ibq_df.dropna(subset=['infant_gestational_age'])

ibq_df['ibq_mean'] = ibq_df[ibq_cols].mean(axis=1)

# Correlation analysis using Spearman's rank correlation

gi_correlations = []
for col in ibq_cols:
    x = ibq_df['infant_gestational_age']
    y = ibq_df[col]

    # Drop any pairwise NaNs before correlation
    valid = (~x.isna()) & (~y.isna())
    if valid.sum() > 1:   # need at least 2 valid pairs
        gi_rho, gi_p = stats.spearmanr(x[valid], y[valid])
    else:
        gi_rho, gi_p = (np.nan, np.nan)

    gi_correlations.append({'item': col, 'rho': gi_rho, 'p': gi_p})

gi_corr_df = pd.DataFrame(gi_correlations)
gi_corr_df['significant'] = gi_corr_df['p'] < 0.05
print(gi_corr_df.sort_values('rho'))


'''A Spearman rank correlation found no significant relationship between infants’ gestational age at birth and their average IBQ temperament 
scores (ρ = –0.03, p = .50). This suggests that gestational maturity at birth was not associated with behavioural or emotional reactivity levels 
at 3–12 months in this sample.'''

# Visualising with a heatmap

print(gi_corr_df.set_index('item')[['rho']].T)


plt.figure(figsize=(6, 1.5))
sns.heatmap(
    gi_corr_df.set_index('item')[['rho']].T,  # <-- fixed
    annot=True, cmap='viridis', center=0,
    cbar=False, fmt=".2f"
)
plt.title('Spearman Correlations: Gestational Age vs IBQ-R Items')
plt.yticks([])
plt.show()


