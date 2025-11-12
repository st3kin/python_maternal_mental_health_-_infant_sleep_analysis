import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Loading the data

participant_df = pd.read_csv('CSV_files/participant.csv')
mental_health_df = pd.read_csv('CSV_files/mental_health.csv')


# Is there a correlation between babies' sex and their sleep durations?

ss_df = participant_df[['infant_sex', 'infant_nightly_sleep_duration']].dropna()
ss_df = ss_df[ss_df['infant_nightly_sleep_duration'] != 100.65]

female_summary = ss_df[ss_df['infant_sex'] == 'Female'].describe()
#print(female_summary)

male_summary = ss_df[ss_df['infant_sex'] == 'Male'].describe()
#print(male_summary)


# Visualisation with a box plot

plt.figure(figsize=(6, 8))
sns.boxplot(data=ss_df, x='infant_sex', y='infant_nightly_sleep_duration', palette='mako')
sns.stripplot(data=ss_df, x='infant_sex', y='infant_nightly_sleep_duration', color='black', alpha=0.5, size=2)
plt.title('Infant Nightly Sleep Duration by Sex')
plt.xlabel('Sex')
plt.ylabel('Sleep duration (hours)')
plt.tight_layout()
plt.show()

# Mann-Whitney U test

female = ss_df.loc[ss_df['infant_sex'] == 'Female', 'infant_nightly_sleep_duration']
male = ss_df.loc[ss_df['infant_sex'] == 'Male', 'infant_nightly_sleep_duration']

U, p = stats.mannwhitneyu(female, male, alternative='two-sided')
r = abs(stats.norm.ppf(p/2)) / (len(ss_df)**0.5)

print(f"Mann-Whitney U: U={U:.2f}, p={p:.5f}, r={r:.3f}")

'''A Mann-Whitney U test comparing infants' sexes and their nightly sleep durations found that biological sex is not a counfounding factor on 
how long babies sleep at night. Both males and females sleep roughly similar durations.'''


