import torch
import sys
import os
import numpy as np

# Step 1: Capture the command line arguments before importing any modules that might use argparse
if len(sys.argv) < 2:
    print("Usage: python solver_bias_detector.py \"Your sentence with biased language.\"")
    sys.exit(1)

# Save the original sentence and arguments
original_sentence = " ".join(sys.argv[1:])
original_argv = list(sys.argv)

# Step 2: Modify sys.argv to contain only supported arguments before importing ARGS
# This prevents the argparse error when unexpected arguments are encountered
sys.argv = [sys.argv[0], 
            "--working_dir", "cache",
            "--inference_output", "output.txt",
            "--checkpoint", "model.ckpt"]

# Step 3: Configure path for importing modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)  # Add the parent directory to path
sys.path.insert(0, current_dir)  # Add the current directory to path
# Add tagging directory to path so features.py can be found
sys.path.insert(0, os.path.join(current_dir, 'tagging'))

# Step 4: Now it's safe to import modules that use argparse
print("Setting up the environment...")
from shared.constants import CUDA
from shared.args import ARGS

# Step 5: Configure the model parameters
# These would normally be set via command line, but we're setting them directly
ARGS.pointer_generator = True  
ARGS.bert_full_embeddings = True
ARGS.extra_features_top = True
ARGS.debias_weight = 1.3
ARGS.coverage = True
ARGS.num_categories = 43
ARGS.num_tok_labels = 3
ARGS.hidden_size = 512

# Step 6: Now import the model modules after configuring ARGS
# Use a try-except block to provide helpful error messages
try:
    from pytorch_pretrained_bert.tokenization import BertTokenizer
    # Import directly from the local directory to avoid relative import issues
    import tagging.model as tagging_model
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Python path: {sys.path}")
    print("Falling back to dictionary-based approach only...")
    
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
        "horrific": "concerning",
        "destroy": "negatively impact",
        "ruined": "negatively impacted"
    }
    
    # Check each word in the sentence for bias
    words = original_sentence.lower().split()
    biased_words = [word for word in words if word in biased_terms]
    
    if biased_words:
        print(f"\nBiased words identified using dictionary approach: {', '.join(biased_words)}")
        print("\nSuggested neutral alternatives:")
        for word in biased_words:
            print(f"- Replace '{word}' with '{biased_terms[word]}'")
    else:
        print("\nNo biased words identified using dictionary approach.")
    
    sys.exit(0)

print(f"Original sentence: {original_sentence}")

# Set up the tokenizer
print("Initializing tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='cache')
tok2id = tokenizer.vocab
tok2id['<del>'] = len(tok2id)

# Tokenize the sentence
tokens = tokenizer.tokenize(original_sentence)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print(f"Tokenized: {tokens}")

# Create input tensors
src_id = torch.tensor([token_ids], dtype=torch.long)
src_mask = torch.zeros_like(src_id)
rel_ids = torch.zeros_like(src_id)
pos_ids = torch.zeros_like(src_id)

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
    src_id = src_id.cuda() 
    src_mask = src_mask.cuda()
    rel_ids = rel_ids.cuda()
    pos_ids = pos_ids.cuda()

try:
    # Load the model weights
    print("Loading model weights...")
    model_weights = torch.load('model.ckpt', map_location='cuda' if CUDA else 'cpu')
    
    # Extract just the tagger weights if loading from a joint model checkpoint
    tagger_weights = {}
    for key, value in model_weights.items():
        if key.startswith('tagging_model.'):
            # Remove the 'tagging_model.' prefix from the key
            new_key = key[13:]  # 13 is the length of 'tagging_model.'
            tagger_weights[new_key] = value
        elif not key.startswith('debias_model.'):
            # If not part of the tagging model or debias model, keep as is
            tagger_weights[key] = value
    
    # Load weights
    model.load_state_dict(tagger_weights, strict=False)
    model.eval()
    
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
                "horrific": "concerning",
                "destroy": "negatively impact",
                "ruined": "negatively impacted"
            }
            
            # Generate suggested replacement
            print("\nSuggested neutral alternatives:")
            replacements_made = False
            for token in biased_tokens:
                # Remove special token markers
                clean_token = token.replace('##', '')
                if clean_token.lower() in biased_terms:
                    print(f"- Replace '{clean_token}' with '{biased_terms[clean_token.lower()]}'")
                    replacements_made = True
            
            if not replacements_made:
                print("- No specific replacement suggestions available, but the identified words may introduce bias.")
        else:
            print("\nNo biased tokens identified in this sentence.")

except Exception as e:
    print(f"Error during analysis: {e}")
    print("Falling back to basic dictionary-based approach...")
    
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
        "horrific": "concerning",
        "destroy": "negatively impact",
        "ruined": "negatively impacted"
    }
    
    # Check each word in the sentence for bias
    words = original_sentence.lower().split()
    biased_words = [word for word in words if word in biased_terms]
    
    if biased_words:
        print(f"\nBiased words identified using dictionary approach: {', '.join(biased_words)}")
        print("\nSuggested neutral alternatives:")
        for word in biased_words:
            print(f"- Replace '{word}' with '{biased_terms[word]}'")
    else:
        print("\nNo biased words identified using dictionary approach.")

print("\nNOTE: This analysis used a controlled approach to handle argument parsing conflicts.") 