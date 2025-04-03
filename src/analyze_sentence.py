import torch
import os
import sys
import numpy as np
import re

# Check if a sentence is provided
if len(sys.argv) < 2:
    print("Usage: python analyze_sentence.py \"Your sentence to analyze here\"")
    sys.exit(1)

# Get the sentence from command line
input_sentence = " ".join(sys.argv[1:])
print(f"Analyzing sentence for bias: \"{input_sentence}\"")

# Dictionary of common biased terms and their neutral alternatives (fallback approach)
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

# Try to use the model-based approach first
try:
    # Setup the Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    sys.path.insert(0, os.path.join(parent_dir, 'src'))
    os.environ["PYTHONPATH"] = parent_dir

    # Import the necessary modules
    from pytorch_pretrained_bert.tokenization import BertTokenizer
    from shared.constants import CUDA
    from shared.args import ARGS
    import tagging.model as tagging_model

    print("Using model-based bias detection...")

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

    # Process the sentence
    print("\n" + "="*80)
    print(f"Original sentence: {input_sentence}")
    
    # Tokenize the sentence
    tokens = tokenizer.tokenize(input_sentence)
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
            replacements_made = False
            for token in biased_tokens:
                # Remove special token markers
                clean_token = token.replace('##', '')
                if clean_token.lower() in biased_terms:
                    print(f"- Replace '{clean_token}' with '{biased_terms[clean_token.lower()]}'")
                    replacements_made = True
            
            if not replacements_made:
                print("- No predefined replacements available for the identified biased tokens")
            
            # Construct neutral version if possible
            try:
                neutral_sentence = input_sentence
                changes_made = False
                for token in biased_tokens:
                    clean_token = token.replace('##', '')
                    if clean_token.lower() in biased_terms:
                        pattern = re.compile(re.escape(clean_token), re.IGNORECASE)
                        replacement = biased_terms[clean_token.lower()]
                        neutral_sentence = pattern.sub(replacement, neutral_sentence)
                        changes_made = True
                
                if changes_made:
                    print(f"\nNeutralized version: {neutral_sentence}")
                else:
                    print("\nCould not generate a neutral version: no replacements available in dictionary")
            except Exception as e:
                print(f"\nCould not automatically generate a neutral version: {e}")
        else:
            print("\nNo biased tokens identified in this sentence.")

    print("\nNOTE: This analysis used model-based bias detection with GPU acceleration.")

except Exception as e:
    print(f"Error using model-based approach: {e}")
    print("Falling back to dictionary-based approach...")
    
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
    
    # Process the sentence with dictionary approach
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
        print("\nNo biased tokens identified in this sentence using dictionary lookup.")
    
    print("\nNOTE: This analysis used a dictionary-based approach to detect bias in text.") 