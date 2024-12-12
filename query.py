import openai
import pandas as pd
import os
import time

#! Richie Cotton's guide on querying the GPT API, found at:
#! https://www.datacamp.com/tutorial/using-gpt-models-via-the-openai-api-in-python

openai.api_key = os.environ.get('OPENAI_DHH_KEY')

# File name
input_file = "dataset/writing_prompts_test_subset.csv"

def generate_completion(prompt):
    """Generate a story completion using GPT-3.5-turbo."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for completing creative writing prompts."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Error generating completion: {e}")
        return None

def main():
    # Load the CSV file
    df = pd.read_csv(input_file)

    # Ensure the 'completion' column exists
    if 'completion' not in df.columns:
        df['completion'] = ''

    # Iterate through the rows
    for index, row in df.iterrows():
        if not pd.isna(row['completion']) and row['completion'] != '':
            continue  # Skip rows that already have a completion

        prompt = row['prompt']
        print(f"Processing row {index + 1}: {prompt}")

        # Generate the completion
        completion = generate_completion(prompt)

        # If successful, add it to the dataframe
        if completion:
            df.at[index, 'completion'] = completion

        # Save progress periodically to avoid data loss
        if (index + 1) % 10 == 0:
            df.to_csv(input_file, index=False)
            print(f"Saved progress at row {index + 1}")

        # Rate limit to comply with API usage policy
        time.sleep(1.5)

    # Save the final completed file
    df.to_csv(input_file, index=False)
    print(f"Completed and saved to {input_file}")

if __name__ == "__main__":
    main()