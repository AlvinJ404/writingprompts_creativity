from datasets import load_dataset
dataset = load_dataset("euclaise/writingprompts")

import csv

# len(train) 272600
# len(test) 15138
# len(val) 15620

mode = 'train'

# Save 1k prompts with corresponding stories to a csv file

with open(f'{mode}.csv', mode='a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    
    for entry in dataset[f"{mode}"]:        
        writer.writerow([entry["prompt"], entry["story"]])