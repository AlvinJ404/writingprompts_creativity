import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

gpt_3_df = pd.read_csv('dataset/gpt-3.5-turbo/writing_prompts_train_subset_normalized_trimmed.csv')
gpt_4_df = pd.read_csv('dataset/gpt-4/writing_prompts_train_subset_normalized_trimmed.csv')

gpt_3_correlation_matrix = gpt_3_df.corr()

gpt_4_correlation_matrix = gpt_4_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(gpt_3_correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("GPT-3.5-turbo: Correlation Matrix")
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(gpt_4_correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("GPT-4: Correlation Matrix")
plt.show()