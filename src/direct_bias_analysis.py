import torch
import numpy as np
import os
import sys
from pytorch_pretrained_bert.tokenization import BertTokenizer

# Read sentences from a file instead of hardcoding them
def read_sentences_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Strip whitespace and filter out empty lines
            sentences = [line.strip() for line in f.readlines() if line.strip()]
        return sentences
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

# Check if file path is provided
if len(sys.argv) < 2:
    print("Usage: python direct_bias_analysis.py <input_file>")
    print("Input file should contain one sentence per line")
    sys.exit(1)

input_file = sys.argv[1]
SENTENCES = read_sentences_from_file(input_file)

if not SENTENCES:
    print(f"No sentences found in {input_file} or file could not be read.")
    sys.exit(1)

print(f"Loaded {len(SENTENCES)} sentences from {input_file}")

# Import modules
import sys; sys.path.append('.')

# We need to set these before importing the modules
os.environ["PYTHONPATH"] = os.path.abspath(os.getcwd())

# Import after setting environment
from shared.constants import CUDA
from shared.args import ARGS
import tagging.model as tagging_model

# Set parameters
ARGS.extra_features_top = True
ARGS.working_dir = 'cache'
ARGS.num_categories = 43
ARGS.num_tok_labels = 3
ARGS.pre_enrich = True
ARGS.activation_hidden = True

# Set up the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='cache')
tok2id = tokenizer.vocab
tok2id['<del>'] = len(tok2id)

# Initialize bias detection model
print("Initializing bias detection model...")
model = tagging_model.BertForMultitaskWithFeaturesOnTop.from_pretrained(
    'bert-base-uncased',
    cls_num_labels=ARGS.num_categories,
    tok_num_labels=ARGS.num_tok_labels,
    cache_dir='cache',
    tok2id=tok2id)

# Move to GPU if available
if CUDA:
    print("Using GPU acceleration")
    model = model.cuda()

# Load the model weights
print("Loading model weights...")
model_weights = torch.load('model.ckpt', map_location='cuda' if CUDA else 'cpu')

# Extract just the tagger weights
tagger_weights = {}
for key, value in model_weights.items():
    if key.startswith('tagging_model.'):
        # Remove the 'tagging_model.' prefix from the key
        new_key = key[13:]  # 13 is the length of 'tagging_model.'
        tagger_weights[new_key] = value

# Load weights
model.load_state_dict(tagger_weights, strict=False)
model.eval()

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
    "destroy": "negatively impact"
}

# Process each sentence
for sentence in SENTENCES:
    print("\n" + "="*80)
    print(f"Original sentence: {sentence}")
    
    # Tokenize the sentence
    tokens = tokenizer.tokenize(sentence)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    # Create input tensors
    src_id = torch.tensor([token_ids], dtype=torch.long)
    src_mask = torch.zeros_like(src_id)
    rel_ids = torch.zeros_like(src_id)
    pos_ids = torch.zeros_like(src_id)
    
    # Print tokens
    print(f"Tokenized: {tokens}")
    
    # Move tensors to GPU if available
    if CUDA:
        src_id = src_id.cuda()
        src_mask = src_mask.cuda()
        rel_ids = rel_ids.cuda()
        pos_ids = pos_ids.cuda()
    
    # Run inference to detect bias
    with torch.no_grad():
        print("Detecting bias...")
        _, tok_logits = model(
            src_id, attention_mask=1.0 - src_mask,
            rel_ids=rel_ids, pos_ids=pos_ids
        )
        
        # Get bias probabilities
        tok_probs = tok_logits[:, :, :2]
        bias_probs = tok_probs[:, :, -1]  # Last dimension is bias probability
        bias_probs = bias_probs.masked_fill(src_mask.bool(), 0.0)
        
        # Convert to numpy for easier handling
        bias_probs_np = bias_probs[0][:len(tokens)].cpu().numpy()
        
        # Print bias probabilities for each token
        print("\nBias probabilities for each token:")
        biased_tokens = []
        for token, prob in zip(tokens, bias_probs_np):
            # Positive values indicate bias
            bias_level = "BIASED" if prob > 0 else "neutral"
            print(f"{token}: {prob:.4f} - {bias_level}")
            if prob > 0:
                biased_tokens.append(token)
        
        if biased_tokens:
            print(f"\nBiased tokens identified: {' '.join(biased_tokens)}")
            
            # Generate suggested replacement
            print("\nSuggested neutral alternatives:")
            for token in biased_tokens:
                # Remove special token markers
                clean_token = token.replace('##', '')
                if clean_token.lower() in biased_terms:
                    print(f"- Replace '{clean_token}' with '{biased_terms[clean_token.lower()]}'")
            
            # Construct neutral version if possible
            try:
                neutral_sentence = sentence
                for token in biased_tokens:
                    clean_token = token.replace('##', '')
                    if clean_token.lower() in biased_terms:
                        neutral_sentence = neutral_sentence.replace(
                            clean_token, biased_terms[clean_token.lower()])
                
                print(f"\nNeutralized version: {neutral_sentence}")
            except Exception as e:
                print(f"Could not automatically generate a neutral version: {e}")
        else:
            print("\nNo biased tokens identified in this sentence.")

print("\nNOTE: This analysis used GPU acceleration to detect bias in the text.") 