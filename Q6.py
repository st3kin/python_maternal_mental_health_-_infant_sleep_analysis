import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import pingouin as pg
from scipy import stats
from scipy.stats import chi2_contingency

# Loading the data

participant_df = pd.read_csv('CSV_files/participant.csv')
mental_health_df = pd.read_csv('CSV_files/mental_health.csv')


# Is there a relationship between mothers' education level and the method they use to put their babies to sleep?

edusleep_df = participant_df[['education', 'infant_sleeping_method']].dropna()

# Visualising with a 100% stacked horizontal bar chart

grouped = edusleep_df.groupby('education')['infant_sleeping_method'].value_counts(normalize=True).unstack('infant_sleeping_method')

fig, ax = plt.subplots(1, 1, figsize=[12, 6])
grouped.plot.barh(stacked=True, cmap=cm.get_cmap('viridis'), ax=ax)
ax.legend(
    bbox_to_anchor=(0.3, 1.04),
    loc='lower center',
    borderaxespad=0,
    frameon=True,
    ncol=5
)
ax.set_xlabel("Infant's sleep method (percentage)")
ax.set_ylabel("Mother's education level")
plt.tight_layout()
plt.show()

# Statistical analysis

# Chi-square test of independence

contingency = pd.crosstab(edusleep_df['education'], edusleep_df['infant_sleeping_method'])

chi2, p, dof, expected = chi2_contingency(contingency)

print(f"Chi-square test: χ²={chi2:.3f}, df={dof}, p={p:.4g}")

# Cramer's V

n = contingency.to_numpy().sum()
cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape)-1)))
print(f"Cramer's V = {cramers_v:.3f}")

'''A chi-square test of independence found no significant relationship between mothers’ education level and the method they use 
to put their babies to sleep, χ²(16) = 14.40, p = .57. The effect size (Cramér’s V = 0.09) indicates only a weak association, 
suggesting that mothers across different education levels tended to use similar sleep-soothing methods.'''

