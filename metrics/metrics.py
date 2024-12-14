"""
This script processes a dataset of writing prompts and completions to compute various linguistic and semantic metrics.
It includes functionalities for evaluating the quality and similarity of text data using state-of-the-art NLP models and metrics.

Key functionalities:
- Compute cosine similarity between prompts and completions using sentence embeddings.
- Compute perplexity of completions using GPT-2 language model.
- Calculate n-gram overlap between prompts and completions.
- Generate n-gram transition probabilities and compute average transition probability.
- Calculate self-BLEU scores for completions to evaluate diversity.
- Compute basic text statistics like word and character counts.

Dependencies:
- pandas
- torch
- scipy
- transformers
- sentence-transformers
- nltk
"""
import pandas as pd
import torch
from scipy.spatial.distance import cosine
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
from collections import Counter, defaultdict
from typing import List, Tuple, Dict
from nltk.util import ngrams

path_to_dataset_file = ''

df = pd.read_csv(path_to_dataset_file)
df['prompt'] = df['prompt'].astype(str)
df['completion'] = df['completion'].astype(str)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_cosine_similarity(text1: str, text2: str) -> float:
    """
    Compute cosine similarity between two text strings using sentence embeddings.

    Args:
        text1 (str): The first text string.
        text2 (str): The second text string.

    Returns:
        float: Cosine similarity between the embeddings of the two text strings.
    """
    embeddings1 = model.encode(text1, convert_to_tensor=True).cpu()
    embeddings2 = model.encode(text2, convert_to_tensor=True).cpu()
    return 1 - cosine(embeddings1, embeddings2)

model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)

def compute_perplexity(text: str) -> float:
    """
    Compute perplexity of a text using GPT-2 language model.

    Args:
        text (str): Input text.

    Returns:
        float: Perplexity score of the input text.
    """
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    with torch.no_grad():
        output = model(input_ids, labels=input_ids)
        loss = output[0]
        perplexity = torch.exp(loss)
    return perplexity.item()

def compute_n_gram_overlap(text1: str, text2: str, n: int = 3) -> float:
    """
    Compute n-gram overlap ratio between two texts.

    Args:
        text1 (str): The first text.
        text2 (str): The second text.
        n (int): Order of n-grams to consider.

    Returns:
        float: Overlap ratio of n-grams between the two texts.
    """
    tokens1 = text1.split()
    tokens2 = text2.split()
    
    if len(tokens1) < n or len(tokens2) < n:
        return 0.0  
    
    counter1 = Counter(zip(*[tokens1[i:] for i in range(n)]))
    counter2 = Counter(zip(*[tokens2[i:] for i in range(n)]))
    
    intersection = sum((counter1 & counter2).values())
    union = sum((counter1 | counter2).values())
    
    return intersection / union if union != 0 else 0.0

def compute_n_gram_transition_probs(text: str, n: int = 3) -> Dict[Tuple[str], float]:
    """
    Compute transition probabilities for n-grams in a text.

    Args:
        text (str): Input text.
        n (int): Order of n-grams to consider.

    Returns:
        Dict[Tuple[str], float]: A dictionary of n-grams and their transition probabilities.
    """
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

def average_transition_probability(text: str, n: int = 3) -> float:
    """
    Compute the average transition probability of n-grams in a text.

    Args:
        text (str): Input text.
        n (int): Order of n-grams to consider.

    Returns:
        float: Average transition probability of n-grams.
    """
    probs = compute_n_gram_transition_probs(text, n)
    return sum(probs.values()) / len(probs) if probs else 0

def compute_self_bleu(text: List[str], ngram_weights: List[float] = [0.25, 0.25, 0.25, 0.25]) -> float:
    """
    Compute self-BLEU score for a list of sentences using GPU acceleration.
    
    Args:
        text (List[str]): List of sentences to compute self-BLEU for
        ngram_weights (List[float]): Weights for different n-gram orders (default: equal weights for 1-4 grams)
    
    Returns:
        float: Self-BLEU score
    """
    if not torch.cuda.is_available():
        raise RuntimeError("GPU not available. Please ensure CUDA is installed and a GPU is accessible.")
    
    device = torch.cuda.current_device()
    
    def get_ngrams(sentence: str, n: int) -> Counter:
        """Generate n-grams from a sentence."""
        tokens = sentence.lower().split()
        return Counter(tuple(gram) for gram in ngrams(tokens, n))
    
    def prepare_ngram_matches(candidates: List[str], n: int) -> Tuple[torch.Tensor, Dict]:
        """
        Prepare n-gram match tensors for GPU computation.
        Returns tensor of shape (num_sentences, unique_ngrams) and ngram mapping.
        """
        all_ngrams = set()
        for sent in candidates:
            all_ngrams.update(get_ngrams(sent, n).keys())
        
        ngram_to_idx = {gram: idx for idx, gram in enumerate(all_ngrams)}
        ngram_counts = torch.zeros((len(candidates), len(ngram_to_idx)), dtype=torch.float32, device=device)
        
        for i, sent in enumerate(candidates):
            sent_ngrams = get_ngrams(sent, n)
            for gram, count in sent_ngrams.items():
                if gram in ngram_to_idx:
                    ngram_counts[i, ngram_to_idx[gram]] = count
        
        return ngram_counts, ngram_to_idx

    def calculate_bleu_stats_gpu(ref_ngrams: torch.Tensor, hyp_ngrams: torch.Tensor) -> Tuple[float, float]:
        """Calculate BLEU statistics using GPU tensor operations."""
        matches = torch.minimum(ref_ngrams, hyp_ngrams).sum().item()
        total = hyp_ngrams.sum().item()
        
        return matches, total

    scores = []
    
    for n, weight in enumerate(ngram_weights, start=1):
        if weight == 0:
            continue
            
        total_precision = 0
        ngram_counts, ngram_map = prepare_ngram_matches(text, n)
        
        for i in range(len(text)):
            hypothesis = ngram_counts[i:i+1]
            references = torch.cat([ngram_counts[:i], ngram_counts[i+1:]], dim=0)
            
            matches, total = calculate_bleu_stats_gpu(references, hypothesis)
            
            if total > 0:
                precision = matches / total
                total_precision += precision
        
        if len(text) > 1:
            avg_precision = total_precision / (len(text))
            scores.append((weight, avg_precision))
    
    final_score = 0
    sum_weights = sum(weight for weight, _ in scores)
    
    if sum_weights > 0:
        final_score = sum(weight * score for weight, score in scores) / sum_weights
        
    torch.cuda.empty_cache()
    return final_score

df['cosine'] = df.apply(lambda row: compute_cosine_similarity(row['prompt'], row['completion']), axis=1)
df['ppl'] = df['completion'].apply(compute_perplexity)
df['n-gram_overlap'] = df.apply(lambda row: compute_n_gram_overlap(row['prompt'], row['completion']), axis=1)
df['n-gram_transition'] = df['completion'].apply(average_transition_probability)
df['self-bleu'] = df['completion'].apply(compute_self_bleu)
df['len_word'] = df['completion'].str.split().str.len()
df['len_char'] = df['completion'].str.len()
df.to_csv(path_to_dataset_file, index=False)
