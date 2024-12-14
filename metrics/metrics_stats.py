import pandas as pd

path_to_dataset_file = 'dataset/gpt-4/writing_prompts_train_subset_normalized.csv'

df = pd.read_csv(path_to_dataset_file)

subset_df = df.iloc[:, 4:]

avg = subset_df['elaboration(len_char)'].mean()
std_dev = subset_df['elaboration(len_word)'].std()
print(f"Column: | Average: {avg:.10f} | Standard Deviation: {std_dev:.10f}")