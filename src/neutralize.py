import sys
import re

# Check if a sentence is provided
if len(sys.argv) < 2:
    print("Usage: python neutralize.py \"Your sentence to analyze here\"")
    sys.exit(1)

# Get the sentence from command line
input_sentence = " ".join(sys.argv[1:])
print(f"Analyzing sentence for bias: \"{input_sentence}\"")

# Dictionary of common biased terms and their neutral alternatives
biased_terms = {
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
    "devastating": "significant",
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

# Process the sentence
print("\n" + "="*80)
print(f"Original sentence: {input_sentence}")

# Tokenize the sentence
tokens = tokenize_sentence(input_sentence)

print(f"Tokenized: {tokens}")

# Dictionary-based bias detection
biased_tokens = []

# Check for biased terms
for token in tokens:
    if token.lower() in biased_terms:
        biased_tokens.append(token)

if biased_tokens:
    print(f"\nBiased tokens identified: {' '.join(biased_tokens)}")
    
    # Generate suggested replacement
    print("\nSuggested neutral alternatives:")
    for token in biased_tokens:
        print(f"- Replace '{token}' with '{biased_terms[token.lower()]}'")
    
    # Construct neutral version
    neutral_sentence = input_sentence
    for token in biased_tokens:
        # Use regex to replace the word with correct case matching
        pattern = re.compile(re.escape(token), re.IGNORECASE)
        replacement = biased_terms[token.lower()]
        neutral_sentence = pattern.sub(replacement, neutral_sentence)
    
    print(f"\nNeutralized version: {neutral_sentence}")
else:
    print("\nNo biased tokens identified in this sentence.")

print("\nNOTE: This analysis used a dictionary-based approach to detect bias in text.") 