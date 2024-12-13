import pandas as pd

df = df = pd.read_csv('dataset/gpt-3.5-turbo/writing_prompts_train_subset.csv')

import pandas as pd

# Ensure the 'prompt' column exists in the dataframe
if 'prompt' in df.columns:
    # Count the number of characters in each entry of the 'prompt' column
    df['char_count'] = df['prompt'].apply(lambda x: len(str(x)))
    
    # Compute the average number of characters
    average_char_count = df['char_count'].mean()
    
    print(f"The average number of characters in the 'prompt' column is: {average_char_count:.2f}")
else:
    print("The DataFrame does not have a 'prompt' column.")
