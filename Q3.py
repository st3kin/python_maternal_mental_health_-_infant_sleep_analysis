import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
from scipy import stats

# Loading the data

participant_df = pd.read_csv('CSV_files/participant.csv')
mental_health_df = pd.read_csv('CSV_files/mental_health.csv')


# Is there a correlation between infants' age group and the number of times they wake up at night?

age_df = participant_df[['infant_age_category', 'infant_wakes_per_night']].dropna()
age_df['infant_wakes_per_night'] = pd.to_numeric(age_df['infant_wakes_per_night'], errors='coerce')

# Visual exploration

plt.figure(figsize=(8, 5))
sns.barplot(
    data=age_df,
    x='infant_age_category',
    y='infant_wakes_per_night',
    palette='viridis'
)

plt.title("Infant Nightly Wakes by Age Group")
plt.xlabel("Infant Age Group")
plt.ylabel("Number of Nightly Wakes")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# Conducting a Kruskal-Wallis test

age_groups = [g['infant_wakes_per_night'].values for _, g in age_df.groupby('infant_age_category', observed=True)]
age_H, age_p_kw = stats.kruskal(*age_groups)

age_k = len(age_groups)
age_n = len(age_df)
age_eps2 = (age_H - age_k +1) / (age_n - age_k)

print(f"Kruskal-Wallis: H={age_H:.3f}, p={age_p_kw:.4g}, epsilon^2={age_eps2:.6f}")

'''A Kruskal–Wallis test found no significant difference in the number of nightly wakes across infant age groups 
(H = 0.67, p = 0.716, ε² ≈ 0.00). This suggests that, within the 3–12 month range, age was not significantly related to 
how often infants woke during the night.'''