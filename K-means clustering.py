import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Loading the data

participant_df = pd.read_csv('CSV_files/participant.csv')
mental_health_df = pd.read_csv('CSV_files/mental_health.csv')

# Can we identify distinct infant ‘sleep–temperament profiles’ (e.g., long sleepers with low distress vs short sleepers with high reactivity)?

ibq_cols = ['ibq_3', 'ibq_4', 'ibq_9', 'ibq_10', 'ibq_16', 'ibq_17', 'ibq_28', 'ibq_29', 'ibq_32', 'ibq_33']

features = participant_df[['participant_number', 'infant_nightly_sleep_duration', 'infant_wakes_per_night']].merge(
    mental_health_df[['participant_number', *ibq_cols]],
    on='participant_number',
    how='inner'
)

features = features.dropna(subset=['infant_nightly_sleep_duration', 'infant_wakes_per_night'] + ibq_cols)

for col in ['infant_nightly_sleep_duration', 'infant_wakes_per_night'] + ibq_cols:
    features[col] = pd.to_numeric(features[col], errors='coerce')

features = features[features['infant_nightly_sleep_duration'] != 100.65]

features = features.dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features[['infant_nightly_sleep_duration', 'infant_wakes_per_night'] + ibq_cols])

# k-Means to find sleep–emotion types

inertias, sils = [], []
K_range = range(2, 6)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    sils.append(silhouette_score(X_scaled, kmeans.labels_))

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(K_range, inertias, marker='o')
ax[0].set_title('Elbow Method')
ax[0].set_xlabel('Number of clusters (k)')
ax[0].set_ylabel('Inertia')

ax[1].plot(K_range, sils, marker='o', color='green')
ax[1].set_title('Silhouette Scores')
ax[1].set_xlabel('Number of clusters (k)')
ax[1].set_ylabel('Silhouette score')
plt.tight_layout()
plt.show()

# Fitting the final KMeans based on the elbow/silhouette methods

optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
features['cluster'] = kmeans.fit_predict(X_scaled)

# Principal component analysis

pca = PCA(n_components=2)
coords = pca.fit_transform(X_scaled)
features['pca1'], features['pca2'] = coords[:, 0], coords[:, 1]

# Visualising the clusters

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=features, x='pca1', y='pca2', hue='cluster',
    palette='viridis', alpha=0.7, s=40
)
plt.title('Infant Sleep-Temperament Clusters (PCA Projection)')
plt.xlabel('Principal component 1: Sleep related variance')
plt.ylabel('Principal component 2: Temperament related variance')
plt.tight_layout()
plt.show()

# Cluster summary

cluster_summary = features.groupby('cluster').agg({
    'infant_nightly_sleep_duration': ['mean', 'std'],
    'infant_wakes_per_night': ['mean', 'std'],
    **{col: ['mean'] for col in ibq_cols}
})

pd.set_option('display.max_columns', None)
print("\nCluster Summary (means ± stds):")
print(cluster_summary.round(2))

# Label interpretation hints

print("\nInterpratation Tip:")
print("- Clusters with long sleep duration and low IBQ means -> calmer, well-regulated infants (teal).")
print("- Clusters with short sleep duration and high IBQ means -> more reactive or distressed infants (purple)")
print("- Clusters in-between -> moderate sleepers/reactivity balance (yellow)")


'''
Cluster 0 (purple): Short sleepers, higher reactivity; infants who sleep less and show higher emotional intensity or distress.
Cluster 1 (teal): Longer sleepers, calmer temperament; infants who sleep longer, wake less, and display lower negative affect.
Cluster 2 (yellow): Moderate sleepers, low reactivity; infants with average sleep but lower reactivity/more adaptable temperament.

According to the k-means algorithm results the clusters are loosely distinct, which suggests meaningful subgroups exist. However there’s 
some overlap, meaning infants vary along a continuum rather than strict categories. The diagonal patterns imply a relationship between 
sleep and temperament.
'''

