import pandas as pd
import numpy as np
import torch
from scipy.spatial.distance import cosine
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
from collections import Counter
from collections import defaultdict


path_to_dataset_file = 'dataset/gpt-4/writing_prompts_train_subset.csv'

df = pd.read_csv(path_to_dataset_file)
df['prompt'] = df['prompt'].astype(str)
df['completion'] = df['completion'].astype(str)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_cosine_similarity(text1, text2):
    embeddings1 = model.encode(text1, convert_to_tensor=True).cpu()
    embeddings2 = model.encode(text2, convert_to_tensor=True).cpu()
    return 1 - cosine(embeddings1, embeddings2)

model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)

def compute_perplexity(text):
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    with torch.no_grad():
        output = model(input_ids, labels=input_ids)
        loss = output[0]
        perplexity = torch.exp(loss)
    return perplexity.item()

def compute_n_gram_overlap(text1, text2, n=3):
    tokens1 = text1.split()
    tokens2 = text2.split()
    
    if len(tokens1) < n or len(tokens2) < n:
        return 0.0  
    
    counter1 = Counter(zip(*[tokens1[i:] for i in range(n)]))
    counter2 = Counter(zip(*[tokens2[i:] for i in range(n)]))
    
    intersection = sum((counter1 & counter2).values())
    union = sum((counter1 | counter2).values())
    
    if union == 0:
        return 0.0
        
    return intersection / union

def compute_n_gram_transition_probs(text, n=3):
    tokens = text.split()
    if len(tokens) < n:
        return {}
    
    transitions = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    transition_counts = defaultdict(int)
    
    for transition in transitions:
        transition_counts[transition] += 1
    
    total_transitions = len(transitions)
    transition_probs = {transition: count / total_transitions for transition, count in transition_counts.items()}
    return transition_probs

def average_transition_probability(text, n=3):
    probs = compute_n_gram_transition_probs(text, n)
    return sum(probs.values()) / len(probs) if probs else 0

df['cosine'] = df.apply(lambda row: compute_cosine_similarity(row['prompt'], row['completion']), axis=1)
df['ppl'] = df['completion'].apply(compute_perplexity)
df['n-gram_overlap'] = df.apply(lambda row: compute_n_gram_overlap(row['prompt'], row['completion']), axis=1)
df['n-gram_transition'] = df['completion'].apply(average_transition_probability)
df['len_word'] = df['completion'].str.split().str.len()
df['len_char'] = df['completion'].str.len()
df.to_csv(path_to_dataset_file, index=False)