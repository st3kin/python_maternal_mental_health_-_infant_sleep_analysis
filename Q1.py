import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
from scipy import stats

# Loading the data

participant_df = pd.read_csv('CSV_files/participant.csv')
mental_health_df = pd.read_csv('CSV_files/mental_health.csv')


# Is there a correlation between infants' number of wakes per night and their method of sleeping?

sleep_df = participant_df[['infant_wakes_per_night', 'infant_sleeping_method']].dropna()
sleep_df['infant_wakes_per_night'] = pd.to_numeric(sleep_df['infant_wakes_per_night'], errors='coerce')

plt.figure(figsize=(8, 5))
sns.boxplot(data=sleep_df,
            x='infant_sleeping_method',
            y='infant_wakes_per_night',
            palette='viridis')
sns.stripplot(data=sleep_df,
              x='infant_sleeping_method',
              y='infant_wakes_per_night',
              color='black',
              size=3,
              alpha=0.5)
plt.title("Number of Nightly Wakes by Infants' Sleeping Method")
plt.xlabel('Sleeping method')
plt.ylabel('Number of nightly wakes')
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# Conducting a Kruskal-Wallis test (non-parametric one-way ANOVA)

sleep_df = participant_df[['infant_sleeping_method', 'infant_wakes_per_night']].copy()
sleep_df['infant_wakes_per_night'] = pd.to_numeric(sleep_df['infant_wakes_per_night'], errors='coerce')

sleep_order = [
    'Alone in the crib',
    'In the crib with parental presence',
    'While being held',
    'While being rocked',
    'While being fed'
]

sleep_df['infant_sleeping_method'] = pd.Categorical(sleep_df['infant_sleeping_method'], categories=sleep_order, ordered=True)

sleep_groups = [g['infant_wakes_per_night'].values for _, g in sleep_df.groupby('infant_sleeping_method', observed=True)]
sleep_H, sleep_p_kw = stats.kruskal(*sleep_groups)

sleep_k = sleep_df['infant_sleeping_method'].nunique()
sleep_n = len(sleep_df)
sleep_eps2 = (sleep_H - sleep_k + 1) / (sleep_n - sleep_k)

print(f"Kruskal-Wallis: H={sleep_H:.3f}, p={sleep_p_kw:.4g}, epsilon^2={sleep_eps2:.6f}")
# Rule of thumb: ~0.01 small, ~0.06 medium, ~0.14 large

sleep_posthoc = pg.pairwise_tests(
    data=sleep_df,
    dv='infant_wakes_per_night',
    between='infant_sleeping_method',
    parametric=False,
    padjust='holm',
    effsize='hedges'
)

print(sleep_posthoc[['A', 'B', 'U-val', 'p-unc', 'p-corr', 'hedges']].sort_values('p-corr'))
print(f"Epsilon2: {sleep_eps2:.6f}")

'''Infants who fall asleep independently (alone in the crib) tend to have fewer nighttime 
awakenings than those who fall asleep while being soothed (fed, rocked, or held).
Statistically significant, but effect size small — suggesting many other factors likely play 
larger roles.'''





