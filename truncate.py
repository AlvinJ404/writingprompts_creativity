#! Keep columns before 'completion'

import pandas as pd

df = pd.read_csv('dataset/gpt-4/writing_prompts_train_subset.csv')

columns_to_keep = df.columns[:df.columns.get_loc('completion')]
df = df[columns_to_keep]

df.to_csv('dataset/gpt-4/writing_prompts_train_subset.csv', index=False)

print(df)
