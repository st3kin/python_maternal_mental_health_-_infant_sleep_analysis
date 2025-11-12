import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Loading the data

participant_df = pd.read_csv('CSV_files/participant.csv')
mental_health_df = pd.read_csv('CSV_files/mental_health.csv')


# Do infants who fall asleep independently (alone in crib) score lower on the IBQ-R negative emotionality dimensions?

ibq_cols = ['ibq_3', 'ibq_4', 'ibq_9', 'ibq_10', 'ibq_16', 'ibq_17', 'ibq_28', 'ibq_29', 'ibq_32', 'ibq_33']

ibq2_df = participant_df[['participant_number', 'infant_sleeping_method']].merge(
    mental_health_df[['participant_number', *ibq_cols]],
    on='participant_number', how='left'
)

ibq2_df[ibq_cols] = ibq2_df[ibq_cols].apply(pd.to_numeric, errors='coerce')
ibq2_df['independent_sleep'] = ibq2_df['infant_sleeping_method'].eq('Alone in the crib')

ibq2_df['ibq_mean'] = ibq2_df[ibq_cols].mean(axis=1)

desc = ibq2_df.groupby('independent_sleep')['ibq_mean'].agg(['count', 'median', 'mean', 'std'])
print(desc)

# Visualising

plt.figure(figsize=(8, 5))
sns.boxplot(data=ibq2_df,
            x='independent_sleep',
            y='ibq_mean',
            palette='viridis')
sns.stripplot(data=ibq2_df,
              x='independent_sleep',
              y='ibq_mean',
              color='black',
              size=3,
              alpha=0.5)
plt.title('Infant Sleep Dependence vs IBQ-R Negative Emotionality Dimension Scores')
plt.xlabel('Infant falls asleep independently')
plt.ylabel('IBQ-R negative emotionality (1-7 Likert)')
plt.xticks([0, 1], ['No', 'Yes'])
plt.tight_layout()
plt.show()

# Conducting a Mann-Whitney U test for statistical analysis

a = ibq2_df.loc[ibq2_df['independent_sleep'], 'ibq_mean'].dropna()
b = ibq2_df.loc[~ibq2_df['independent_sleep'], 'ibq_mean'].dropna()
res = stats.mannwhitneyu(a, b, alternative='two-sided')
U, p = res.statistic, res.pvalue
r_rb = 1 - (2 * U) / (len(a) * len(b))

print(f"Mann-Whitney U: U={U:.2f}, p={p:.5f}, r_rb={r_rb:.3f} (pos = higher in independent sleepers)")


'''A Mann–Whitney U test found that infants who fall asleep independently exhibited significantly lower negative emotionality 
scores on the IBQ-R (U = 14,903, p < .001, r_rb = 0.28). This suggests that independent sleep initiation may be linked to 
calmer or more self-regulated temperament profiles.'''

