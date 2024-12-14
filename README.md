# writingprompts_creativity

This repository contains a derived subset of the [WritingPrompts](https://paperswithcode.com/dataset/writingprompts) dataset, designed to explore the creative potential of prompts and their completions using advanced natural language processing (NLP) techniques. The repository features data preprocessing scripts, analytical tools, and metrics to study creativity in text generation.

---

## Table of Contents
- [Introduction](#introduction)
- [Dataset Details](#dataset-details)
- [Features](#features)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Introduction
Prompt Creativity focuses on analyzing and enhancing creativity in Language Models (LMs) using a subset of the WritingPrompts dataset. The goal is to examine linguistic features, semantic coherence, and diversity in generated completions, with applications in creative writing and AI storytelling.

---

## Dataset Details
The original dataset, [WritingPrompts](https://paperswithcode.com/dataset/writingprompts), is a publicly accessible collection of creative writing prompts and user-generated completions. It was sourced from Reddit's r/WritingPrompts community and adapted for machine learning applications.

The dataset used in this repository is a **derived subset** that has been preprocessed for use in NLP tasks, including:
- Semantic similarity analysis
- N-gram overlap metrics
- Self-BLEU diversity evaluation
- Perplexity scoring

Original WritingPrompts dataset links:
- [Papers with Code](https://paperswithcode.com/dataset/writingprompts)
- [Hugging Face Hub](https://huggingface.co/datasets/euclaise/writingprompts)

---

## Features
This repository includes:
- **Derived Dataset**: A processed subset of WritingPrompts focused on creativity exploration.
- **Evaluation Metrics**: Tools for analyzing text quality, including:
  - Cosine similarity
  - Perplexity
  - N-gram overlap
  - Self-BLEU

---

## Usage
### Prerequisites
Ensure you have the following installed:
- Python 3.8 or higher

Clone the repository:
```bash
git clone https://github.com/AlvinJ404/writingprompts_creativity.git
cd writingprompts_creativity
```

Install dependencies:
```bash
pip install -r requirements.txt
```
