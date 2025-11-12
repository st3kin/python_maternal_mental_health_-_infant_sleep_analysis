import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
from scipy import stats

# Loading the data

participant_df = pd.read_csv('CSV_files/participant.csv')
mental_health_df = pd.read_csv('CSV_files/mental_health.csv')


# Is there a correlation between an infant's sleep duration and their method of sleeping?

sleepdur_df = participant_df[['infant_nightly_sleep_duration', 'infant_sleeping_method']].copy()
sleepdur_df = sleepdur_df[sleepdur_df['infant_nightly_sleep_duration'] != 100.65]


# Visualising

plt.figure(figsize=(9, 5))
sns.violinplot(
    data=sleepdur_df,
    x='infant_sleeping_method',
    y='infant_nightly_sleep_duration',
    inner='quartile',
    cut=0,
    palette='viridis'
)
sns.pointplot(
    data=sleepdur_df,
    x='infant_sleeping_method',
    y='infant_nightly_sleep_duration',
    errorbar=('ci', 95),
    join=False,
    color='black',
    markers='D',
    scale=0.6
)
plt.title('Nightly Sleep Duration by Sleeping Method\n(Kruskal–Wallis: p<0.001, ε²≈0.115)')
plt.xlabel('Infant Sleeping Method')
plt.ylabel('Sleep Duration (hours)')
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# Kruskal-Wallis test

sleepdur_groups = [g['infant_nightly_sleep_duration'].values for _, g in sleepdur_df.groupby('infant_sleeping_method', observed=True)]
sleepdur_H, sleepdur_p_kw = stats.kruskal(*sleepdur_groups)
sleepdur_k = len(sleepdur_groups)
sleepdur_n = len(sleepdur_df)
sleepdur_eps2 = (sleepdur_H - sleepdur_k + 1) / (sleepdur_n - sleepdur_k)
print(f"Kruskal-Wallis: H={sleepdur_H:.3f}, p={sleepdur_p_kw:.4g}, epsilon^2={max(sleepdur_eps2, 0):.6f}")

sleepdur_posthoc = pg.pairwise_tests(
    data=sleepdur_df,
    dv='infant_nightly_sleep_duration',
    between='infant_sleeping_method',
    parametric=False,
    padjust='holm',
    effsize='hedges'
)

print(sleepdur_posthoc[['A', 'B', 'U-val', 'p-unc', 'p-corr', 'hedges']].sort_values('p-corr'))

'''A Kruskal–Wallis test showed a significant effect of sleeping method on nightly sleep duration 
(H(4) = 50.39, p < 0.001, ε² = 0.115). Post-hoc pairwise comparisons indicated that infants who fell asleep 
alone in the crib slept significantly longer than those who fell asleep while being fed, rocked, or held 
(Hedges g = 0.56–0.85, large effects). Sleeping in the crib with parental presence produced intermediate results. 
These findings suggest that independent sleep initiation is strongly associated with longer continuous nighttime sleep.'''