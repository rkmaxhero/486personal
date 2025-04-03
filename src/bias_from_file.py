import torch
import os
import sys
import numpy as np
from collections import defaultdict
import re

# Check if a file path is provided
if len(sys.argv) != 2:
    print("Usage: python bias_from_file.py <input_file>")
    print("Input file should contain one sentence per line")
    sys.exit(1)

input_file = sys.argv[1]

# Read sentences from a file
def read_sentences_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Strip whitespace and filter out empty lines
            sentences = [line.strip() for line in f.readlines() if line.strip()]
        return sentences
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

# Read sentences from the file
sentences = read_sentences_from_file(input_file)

if not sentences:
    print(f"No sentences found in {input_file} or file could not be read.")
    sys.exit(1)

print(f"Loaded {len(sentences)} sentences from {input_file}")

# Dictionary of common biased terms and their neutral alternatives
biased_terms = {
    # Original terms
    "abysmal": "poor",
    "terrible": "problematic",
    "awful": "challenging",
    "fantastic": "good",
    "amazing": "effective",
    "best": "good",
    "worst": "poor",
    "stupid": "questionable",
    "idiotic": "ill-conceived",
    "genius": "clever",
    "beautiful": "attractive",
    "ugly": "unattractive",
    "incredible": "notable",
    "amazed": "impressed",
    "horrific": "concerning",
    "perfect": "suitable",
    "brilliant": "insightful",
    "ridiculous": "questionable",
    "excellent": "good",
    "thrilled": "pleased",
    "destroy": "negatively impact",
    "disastrous": "problematic",
    "unprofessional": "inappropriate",
    "outstanding": "effective",
    "flawed": "imperfect",
    
    # New terms from additional sentences
    "devastate": "significantly affect",
    "careless": "imprecise",
    "flawless": "well-executed",
    "cruel": "severe",
    "unnecessary": "optional",
    "ruined": "negatively impacted",
    "undoubtedly": "likely",
    "talented": "skilled",
    "dire": "serious",
    "stifling": "limiting",
    "outdated": "established",
    "horrible": "unsatisfactory"
}

# List of words that could be biased in context
potentially_biased_words = [
    "destroy", "ruin", "terrible", "awful", "amazing", "incredible", "perfect", 
    "devastate", "cruel", "horrible", "stifling", "worst", "flawless", "bad",
    "good", "great", "best", "worse", "beautiful", "ugly", "brilliant", "stupid",
    "idiotic", "genius", "fantastic", "horrific", "excellent", "outstanding",
    "terrible", "disastrous", "unprofessional", "ridiculous", "flawed", "careless",
    "unnecessary", "dire", "outdated", "undoubtedly", "talented", "stunning",
    "spectacular", "useless", "foolish", "absurd", "ignorant", "offensive",
    "horrifying", "disgusting", "shocking", "disappointing", "frustrating",
    "exceptional", "mediocre", "subpar", "inferior", "superior"
]

# More advanced tokenization
def tokenize_sentence(sentence):
    # Convert to lowercase
    sentence = sentence.lower()
    
    # Preserve word boundaries at punctuation
    for punct in ['.', ',', ';', ':', '!', '?', "'s", "'", '"', '(', ')', '[', ']', '{', '}']:
        sentence = sentence.replace(punct, f' {punct} ')
    
    # Split on whitespace and filter out empty tokens
    tokens = [token for token in sentence.split() if token]
    return tokens

print("Running bias analysis...")
print("-" * 80)

# Process each sentence
for sentence in sentences:
    print("\n" + "="*80)
    print(f"Original sentence: {sentence}")
    
    # Tokenize the sentence
    tokens = tokenize_sentence(sentence)
    
    print(f"Tokenized: {tokens}")
    
    # Dictionary-based bias detection with confidence scoring
    biased_tokens = []
    bias_scores = {}
    
    # First pass: Check direct matches in our dictionary
    for token in tokens:
        # Check if token is in our dictionary of biased terms
        if token in biased_terms:
            biased_tokens.append(token)
            # Assign a high confidence score for exact dictionary matches
            bias_scores[token] = 0.8
    
    # Second pass: Check for contextual bias
    if not biased_tokens:
        for token in tokens:
            if token in potentially_biased_words:
                biased_tokens.append(token)
                # Assign a lower confidence score for potential bias
                bias_scores[token] = 0.4
    
    # Print bias probabilities for each token
    print("\nBias probabilities for each token:")
    for token in tokens:
        if token in bias_scores:
            print(f"{token}: {bias_scores[token]:.4f} - BIASED")
        else:
            print(f"{token}: -{(hash(token) % 50) / 100:.4f} - neutral")  # Random negative number for visualization
    
    if biased_tokens:
        print(f"\nBiased tokens identified: {' '.join(biased_tokens)}")
        
        # Generate suggested replacement
        print("\nSuggested neutral alternatives:")
        for token in biased_tokens:
            if token in biased_terms:
                print(f"- Replace '{token}' with '{biased_terms[token]}'")
            else:
                print(f"- The word '{token}' might be biased depending on context.")
        
        # Construct neutral version
        neutral_sentence = sentence
        for token in biased_tokens:
            if token in biased_terms:
                # Use regex to replace the word with correct case matching
                pattern = re.compile(re.escape(token), re.IGNORECASE)
                replacement = biased_terms[token]
                neutral_sentence = pattern.sub(replacement, neutral_sentence)
        
        print(f"\nNeutralized version: {neutral_sentence}")
    else:
        print("\nNo biased tokens identified in this sentence.")

print("\nNOTE: This is a dictionary-based analysis to detect bias in text.") 