import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

file_path = 'dataset/gpt-4/writing_prompts_train_subset_normalized_trimmed.csv'
data = pd.read_csv(file_path)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

pca = PCA()
pca_result = pca.fit_transform(scaled_data)

explained_variance = pca.explained_variance_ratio_

cumulative_variance = np.cumsum(explained_variance)

plt.figure(figsize=(8, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.5, align='center', label='Individual Explained Variance')
plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='Cumulative Explained Variance')
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Components')
plt.legend(loc='best')
plt.title('GPT-4: PCA Explained Variance')
plt.show()

pca_components = pd.DataFrame(pca.components_, columns=data.columns, index=[f'PC{i+1}' for i in range(len(data.columns))])
print("PCA Components (Loadings):")
print(pca_components)
