# Country Clustering: Categorizing Nations by Socio-Economic & Health Factors
 
This project uses **PCA**, **K-Means**, and **Hierarchical Clustering** to categorize 167 countries into three development groups — **Underdeveloped**, **Developing**, and **Developed** — based on socio-economic and health indicators.
 
## Dataset
 
The dataset contains 167 rows and 10 columns:
 
| Column | Description |
|---|---|
| `country` | Name of the country |
| `child_mort` | Death of children under 5 years of age per 1000 live births |
| `exports` | Exports of goods and services per capita (% of GDP per capita) |
| `health` | Total health spending per capita (% of GDP per capita) |
| `imports` | Imports of goods and services per capita (% of GDP per capita) |
| `income` | Net income per person |
| `inflation` | Annual growth rate of the Total GDP |
| `life_expec` | Average number of years a newborn child would live |
| `total_fer` | Number of children that would be born to each woman |
| `gdpp` | GDP per capita (Total GDP / total population) |
 
---
 
## Workflow
 
1. Data cleaning & outlier check
2. Feature scaling
3. Dimensionality reduction with PCA
4. Clustering with K-Means
5. Clustering with Hierarchical (Agglomerative) methods
6. Comparing both methods
7. Labeling clusters as development categories
---
 
## 1. Imports
 
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
 
import warnings
warnings.filterwarnings('ignore')
```
 
## 2. Load & Inspect Data
 
```python
df = pd.read_csv('Country-data.csv')
df.set_index('country', inplace=True)
 
df.info()
df.describe()
df.isnull().sum()
```
 
`country` is set as the index rather than kept as a feature or dropped entirely, since it's a categorical label with no numeric meaning for distance-based algorithms — but we still want it attached to each row so cluster results can be mapped back to real country names.
 
## 3. Outlier Check
 
Even though this dataset had no missing values, extreme values still needed checking. Boxplots and `describe()` showed wide ranges in `income`, `gdpp`, and `child_mort` — but these reflect **real economic disparities** between countries (e.g. Qatar's high income, or high child mortality in the least developed nations), not data errors. These extremes were **kept**, since separating them is the entire goal of the clustering.
 
```python
plt.figure(figsize=(15, 8))
for i, col in enumerate(df.columns, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(y=df[col])
    plt.title(col)
plt.tight_layout()
plt.show()
```
 
## 4. Feature Scaling
 
```python
scaler = StandardScaler()
scaled_array = scaler.fit_transform(df)
df_scaled = pd.DataFrame(scaled_array, columns=df.columns, index=df.index)
```
 
Standardization was necessary because features are on very different scales (e.g. `gdpp` in tens of thousands vs. `total_fer` between 1–8). Without scaling, high-magnitude features like `income` and `gdpp` would dominate the distance calculations used by PCA and K-Means, even though they aren't inherently "more important."
 
## 5. PCA (Dimensionality Reduction)
 
```python
pca = PCA()
pca.fit(df_scaled)
 
cumulative_var = pca.explained_variance_ratio_.cumsum()
print(cumulative_var)
 
plt.plot(range(1, len(cumulative_var)+1), cumulative_var, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Scree Plot')
plt.axhline(y=0.9, color='r', linestyle='--')
plt.grid(True)
plt.show()
```
 
**Result:** the first 3 components explained ~76% of total variance. 3 components were chosen as a balance between retaining enough information and keeping the clustering interpretable/visualizable.
 
```python
pca_final = PCA(n_components=3)
pca_data = pca_final.fit_transform(df_scaled)
pca_df = pd.DataFrame(pca_data, columns=['PC1','PC2','PC3'], index=df_scaled.index)
```
 
PCA reduces the 9 correlated socio-economic/health features into fewer uncorrelated components that capture the main patterns of variation, making the following clustering steps more efficient and easier to visualize.
 
## 6. K-Means Clustering
 
### 6.1 Choosing k (Elbow + Silhouette)
 
```python
inertias, sil_scores = [], []
K_range = range(2, 8)
 
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(pca_data)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(pca_data, labels))
```
 
| k | Inertia | Silhouette |
|---|---|---|
| 2 | 698.24 | 0.349 |
| **3** | **550.21** | **0.372** |
| 4 | 442.08 | 0.309 |
| 5 | 377.78 | 0.289 |
| 6 | 330.54 | 0.306 |
| 7 | 286.42 | 0.311 |
 
**k=3** gave the highest silhouette score and aligned with the elbow in the inertia curve — it was also the natural choice given the task's three target categories.
 
### 6.2 Fit final K-Means model
 
```python
kmeans_final = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels = kmeans_final.fit_predict(pca_data)
df['kmeans_cluster'] = kmeans_labels
```
 
**Cluster sizes:** Cluster 0 → 70 countries, Cluster 1 → 94 countries, Cluster 2 → 3 countries (Luxembourg, Malta, Singapore — extremely high-income outlier economies).
 
### 6.3 Cluster profiling
 
```python
cluster_profile = df.groupby('kmeans_cluster')[
    ['child_mort','income','gdpp','life_expec','total_fer']
].mean()
print(cluster_profile)
```
 
| | child_mort | income | gdpp | life_expec | total_fer |
|---|---|---|---|---|---|
| Cluster 0 | 74.28 | $4,522 | $2,169 | 62.28 | 4.34 |
| Cluster 1 | 12.54 | $25,048 | $19,580 | 76.37 | 1.96 |
| Cluster 2 | 4.13 | $64,033 | $57,567 | 81.43 | 1.38 |
 
Every indicator moves consistently with development level: child mortality and fertility fall while income, GDP per capita, and life expectancy rise — confirming a coherent and interpretable clustering.
 
## 7. Hierarchical Clustering
 
```python
Z = linkage(pca_data, method='ward')
 
plt.figure(figsize=(15,6))
dendrogram(Z, labels=df_scaled.index.tolist(), leaf_rotation=90, leaf_font_size=8)
plt.title('Hierarchical Clustering Dendrogram')
plt.show()
 
hier_labels = fcluster(Z, t=3, criterion='maxclust')
df['hier_cluster'] = hier_labels
```
 
Ward's linkage minimizes within-cluster variance at each merge step, producing compact, natural groupings. Cutting the dendrogram at 3 clusters gave sizes of 64, 100, and 3 countries — closely matching the K-Means split.
 
## 8. Comparing K-Means vs. Hierarchical
 
```python
pd.crosstab(df['kmeans_cluster'], df['hier_cluster'])
```
 
|  | hier_cluster 1 | hier_cluster 2 | hier_cluster 3 |
|---|---|---|---|
| kmeans 0 | 57 | 0 | 13 |
| kmeans 1 | 7 | 0 | 87 |
| kmeans 2 | 0 | 3 | 0 |
 
**Agreement:** (57 + 87 + 3) / 167 ≈ **88%**. The "Developed" outlier cluster (Luxembourg, Malta, Singapore) matched perfectly between both methods. Most disagreement occurred at the Underdeveloped/Developing border — expected, since development level is a continuum without sharp boundaries.
 
## 9. Labeling & Final Categorization
 
```python
label_map = {0: 'Underdeveloped', 1: 'Developing', 2: 'Developed'}
df['development_status'] = df['kmeans_cluster'].map(label_map)
 
df['development_status'].value_counts()
df[['development_status']]
```
 
## 10. Visualizations
 
**Cluster scatter plot on PCA components:**
```python
plt.figure(figsize=(10,7))
sns.scatterplot(x=pca_data[:,0], y=pca_data[:,1], hue=df['kmeans_cluster'], palette='Set1', s=80)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Countries Clustered on PCA Components')
plt.legend(title='Cluster')
plt.show()
```
 
**Cluster profile bar chart (original units):**
```python
df.groupby('kmeans_cluster')[['child_mort','income','gdpp','life_expec']].mean() \
  .plot(kind='bar', figsize=(10,6))
plt.title('Average Indicators per Cluster')
plt.ylabel('Value')
plt.show()
```
 
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/9ffc5b86-b5d3-433f-89ed-ac88d689d071" width="400"/></td>
    <td><img src="https://github.com/user-attachments/assets/049c8594-7aea-4377-8fcc-15b957f68274" width="400"/></td>
  </tr>
</table>
 
## Conclusions
 
- PCA reduced 9 features to 3 components, retaining ~76% of variance while making clustering computationally efficient and visualizable.
- K-Means with **k=3** produced the best-separated clusters (silhouette score 0.372), matching the task's three target categories.
- Hierarchical clustering validated the K-Means result with ~88% agreement, confirming the groupings are robust.
- The three resulting groups map cleanly onto real-world development categories:
  - **Underdeveloped** — high child mortality, low income/GDP, low life expectancy
  - **Developing** — moderate values across all indicators
  - **Developed** — low child mortality, very high income/GDP, high life expectancy (e.g. Luxembourg, Malta, Singapore)
## Tech Stack
 
- Python (pandas, numpy)
- scikit-learn (`StandardScaler`, `PCA`, `KMeans`, `silhouette_score`)
- scipy (`linkage`, `fcluster`, `dendrogram`)
- matplotlib, seaborn
 
