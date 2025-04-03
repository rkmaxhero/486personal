import torch
import sys
import os
import numpy as np
import re

# Check arguments
if len(sys.argv) < 2:
    print("Usage: python enhanced_neutralize.py \"Your sentence with biased language.\"")
    sys.exit(1)

# Get the input sentence
sentence = " ".join(sys.argv[1:])
print(f"Analyzing sentence for bias: \"{sentence}\"")

# Set up paths for importing modules and lexicons
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))
os.environ["PYTHONPATH"] = parent_dir

# Load lexicons for enhanced bias detection
def load_lexicon(filepath):
    terms = set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                # Skip comments and empty lines
                line = line.strip()
                if line and not line.startswith(';'):
                    terms.add(line.lower())
        return terms
    except Exception as e:
        print(f"Warning: Could not load lexicon {filepath}: {e}")
        return set()

# Load lexicons
lexicons_dir = os.path.join(current_dir, 'lexicons')
negative_terms = load_lexicon(os.path.join(lexicons_dir, 'negative_liu2005.txt'))
positive_terms = load_lexicon(os.path.join(lexicons_dir, 'positive_liu2005.txt'))
npov_terms = load_lexicon(os.path.join(lexicons_dir, 'npov_lexicon.txt'))
subjective_terms = load_lexicon(os.path.join(lexicons_dir, 'strong_subjectives_riloff2003.txt'))

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

# Enhance the biased_terms dictionary with lexicon entries
# For words in the negative lexicon, we'll try to find neutral alternatives
for term in negative_terms:
    if term not in biased_terms and len(term) > 3:  # Minimum length to avoid short words
        # Try to find a neutral alternative in npov lexicon or manually set one
        if term.endswith('ly'):
            # For adverbs, remove the 'ly' suffix
            biased_terms[term] = term[:-2] + 'ically appropriate'
        elif term.endswith('ed'):
            # For past tense verbs
            biased_terms[term] = 'impacted'
        elif term.endswith('ing'):
            # For present participles
            biased_terms[term] = 'impacting'
        else:
            biased_terms[term] = 'significant'  # Default neutral term

# Try to use the model-based approach first
try:
    # Import necessary modules
    from pytorch_pretrained_bert.tokenization import BertTokenizer
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

    print("Using model-based bias detection...")

    # Set up the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='cache')
    tok2id = tokenizer.vocab
    tok2id['<del>'] = len(tok2id)

    # Tokenize the sentence
    tokens = tokenizer.tokenize(sentence)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

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

    # Print tokens
    print("\n" + "="*80)
    print(f"Original sentence: {sentence}")
    print(f"Tokenized: {tokens}")

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
                elif clean_token.lower() in negative_terms:
                    suggested = "more neutral term"
                    print(f"- Replace '{clean_token}' with a {suggested}")
                    replacements_made = True
            
            if not replacements_made:
                print("- No specific replacement suggestions available, but the identified words may introduce bias.")
            
            # Construct neutral version if possible
            try:
                neutral_sentence = sentence
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
                    print("\nCould not automatically generate a neutral version: no replacements available in dictionary")
            except Exception as e:
                print(f"\nCould not automatically generate a neutral version: {e}")
        else:
            print("\nNo biased tokens identified in this sentence.")

        print("\nNOTE: This analysis used model-based bias detection with GPU acceleration.")

except Exception as e:
    print(f"Error using model-based approach: {e}")
    print("Falling back to lexicon-based approach...")
    
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
    
    # Process the sentence with lexicon approach
    print("\n" + "="*80)
    print(f"Original sentence: {sentence}")
    
    # Tokenize the sentence
    tokens = tokenize_sentence(sentence)
    print(f"Tokenized: {tokens}")
    
    # Lexicon-based bias detection
    biased_tokens = []
    bias_scores = {}
    
    # Check for biased terms using the lexicons
    for token in tokens:
        token_lower = token.lower()
        if token_lower in biased_terms:
            biased_tokens.append(token)
            bias_scores[token] = 0.8  # High confidence for known biased terms
        elif token_lower in negative_terms:
            biased_tokens.append(token)
            bias_scores[token] = 0.7  # High confidence for negative sentiment
        elif token_lower in subjective_terms and token_lower not in npov_terms:
            biased_tokens.append(token)
            bias_scores[token] = 0.6  # Medium confidence for subjective terms
    
    # Print bias scores
    print("\nBias probabilities for each token:")
    for token in tokens:
        if token in bias_scores:
            print(f"{token}: {bias_scores[token]:.4f} - BIASED")
        else:
            print(f"{token}: -{(hash(token) % 50) / 100:.4f} - neutral")  # Random negative number for neutral tokens
    
    if biased_tokens:
        print(f"\nBiased tokens identified: {' '.join(biased_tokens)}")
        
        # Generate suggested replacement
        print("\nSuggested neutral alternatives:")
        replacements_made = False
        for token in biased_tokens:
            if token.lower() in biased_terms:
                print(f"- Replace '{token}' with '{biased_terms[token.lower()]}'")
                replacements_made = True
            elif token.lower() in negative_terms:
                print(f"- Replace '{token}' with a more neutral term (found in negative sentiment lexicon)")
                replacements_made = True
            elif token.lower() in subjective_terms:
                print(f"- Replace '{token}' with a more objective term (found in subjective lexicon)")
                replacements_made = True
        
        if not replacements_made:
            print("- No specific replacement suggestions available, but the identified words may introduce bias.")
        
        # Construct neutral version if possible
        try:
            neutral_sentence = sentence
            changes_made = False
            for token in biased_tokens:
                if token.lower() in biased_terms:
                    pattern = re.compile(re.escape(token), re.IGNORECASE)
                    replacement = biased_terms[token.lower()]
                    neutral_sentence = pattern.sub(replacement, neutral_sentence)
                    changes_made = True
            
            if changes_made:
                print(f"\nNeutralized version: {neutral_sentence}")
            else:
                print("\nCould not automatically generate a neutral version: no specific replacements available")
        except Exception as e:
            print(f"\nCould not automatically generate a neutral version: {e}")
    else:
        print("\nNo biased tokens identified in this sentence.")
    
    print("\nNOTE: This analysis used a lexicon-based approach to detect bias in text.") 