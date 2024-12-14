import pandas as pd

model = 'gpt-4'
path_to_dataset_file = 'dataset/gpt-4/writing_prompts_train_subset'
df = pd.read_csv(f'{path_to_dataset_file}.csv')

min_len_word = df['len_word'].min()
min_len_char = df['len_char'].min()

max_min_diff_word = df['len_word'].max() - df['len_word'].min()
max_min_diff_char = df['len_char'].max() - df['len_char'].min()


df['novelty'] = 1 - df['cosine']
df['surprise'] = df['ppl']/100
df['orig_and_flex'] = 1-df['n-gram_overlap']*100
df['fluency'] = df['n-gram_transition']*100
df['elaboration(len_word)'] = (df['len_word'] - min_len_word) / max_min_diff_word
df['elaboration(len_char)'] = (df['len_char'] - min_len_char) / max_min_diff_char
df['elaboration(self-bleu)'] = df['self-bleu']/100
print(df.head())

df.to_csv(f'dataset/{model}/writing_prompts_train_subset_normalized.csv')