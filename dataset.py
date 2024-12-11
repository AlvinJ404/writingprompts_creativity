#! WritingPrompts
#! len(train) 272600
#! len(test) 15138
#! len(val) 15620

from datasets import load_dataset
import random
import pandas as pd

dataset = load_dataset("euclaise/writingprompts")

#! Training Data Subset

random_train_indices = random.sample(range(len(dataset['train'])), 2726)
random_train_entries = [dataset['train'][i] for i in random_train_indices]

data = [
    {
        "row_id": idx,
        "prompt": entry["prompt"],
        "story": entry["story"],
        "completion": ""
    }
    for idx, entry in zip(random_train_indices, random_train_entries)
]

df = pd.DataFrame(data)

output_file = 'writing_prompts_train_subset.csv'

df.to_csv(output_file, index=False)

print(f"Saved {len(df)} entries to {output_file}")

#! Testing Data Subset

random_test_indices = random.sample(range(len(dataset['test'])), 1500)
random_test_entries = [dataset['test'][i] for i in random_test_indices]

data = [
    {
        "row_id": idx,
        "prompt": entry["prompt"],
        "story": entry["story"],
        "completion": ""
    }
    for idx, entry in zip(random_test_indices, random_test_entries)
]

df = pd.DataFrame(data)

output_file = 'writing_prompts_test_subset.csv'

df.to_csv(output_file, index=False)

print(f"Saved {len(df)} entries to {output_file}")