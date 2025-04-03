import torch
import sys
import os
import numpy as np

# Step 1: Capture the command line arguments before importing any modules that might use argparse
if len(sys.argv) < 2:
    print("Usage: python advanced_neutralizer.py \"Your sentence with biased language.\"")
    sys.exit(1)

# Save the original sentence and arguments
original_sentence = " ".join(sys.argv[1:])
original_argv = list(sys.argv)

# Step 2: Modify sys.argv to contain only supported arguments before importing ARGS
# This prevents the argparse error when unexpected arguments are encountered
sys.argv = [sys.argv[0], 
            "--working_dir", "cache",
            "--inference_output", "output.txt",
            "--checkpoint", "model.ckpt",
            "--pointer_generator",
            "--coverage",
            "--bert_full_embeddings",
            "--extra_features_top",
            "--debias_weight", "1.3"]

# Step 3: Configure path for importing modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

# Create a symlink from lexicons to src/lexicons if needed
lexicons_source = os.path.join(current_dir, 'lexicons')
lexicons_target = os.path.join(parent_dir, 'lexicons')

# Check if we need to create the symlink
if not os.path.exists(lexicons_target) and os.path.exists(lexicons_source):
    try:
        os.symlink(lexicons_source, lexicons_target)
        print(f"Created symlink from {lexicons_source} to {lexicons_target}")
    except Exception as e:
        print(f"Could not create symlink: {e}")

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
ARGS.bert_model = 'bert-base-uncased'

# Step 6: Now import the model modules after configuring ARGS
try:
    from pytorch_pretrained_bert.tokenization import BertTokenizer
    import tagging.model as tagging_model
    import seq2seq.model as seq2seq_model
    import joint.model as joint_model
    
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
    src_len = torch.tensor([len(token_ids)], dtype=torch.long)
    post_start_id = tok2id['[CLS]']
    rel_ids = torch.zeros_like(src_id)
    pos_ids = torch.zeros_like(src_id)
    
    # Initialize models
    print("Initializing joint model with both detection and neutralization capabilities...")
    
    # Initialize seq2seq model for neutralization
    debias_model = seq2seq_model.PointerSeq2Seq(
        vocab_size=len(tok2id), hidden_size=ARGS.hidden_size,
        emb_dim=768, dropout=0.2, tok2id=tok2id)
    
    # Initialize tagging model for bias detection
    tagger_model = tagging_model.BertForMultitaskWithFeaturesOnTop.from_pretrained(
        'bert-base-uncased',
        cls_num_labels=ARGS.num_categories,
        tok_num_labels=ARGS.num_tok_labels,
        cache_dir='cache',
        tok2id=tok2id)
    
    # Create the joint model that combines both models
    model = joint_model.JointModel(debias_model=debias_model, tagging_model=tagger_model)
    
    # Move to GPU if available
    if CUDA:
        print("Using GPU acceleration")
        model = model.cuda()
        src_id = src_id.cuda()
        src_mask = src_mask.cuda()
        rel_ids = rel_ids.cuda()
        pos_ids = pos_ids.cuda()
        # Keep src_len on CPU for pack_padded_sequence compatibility
    
    # Load the pretrained checkpoint
    print("Loading model checkpoint...")
    model_weights = torch.load('model.ckpt', map_location='cuda' if CUDA else 'cpu')
    model.load_state_dict(model_weights, strict=False)
    model.eval()
    
    try:
        # Run inference for both bias detection and neutralization
        with torch.no_grad():
            print("Running joint inference for bias detection and neutralization...")
            output_ids, bias_probs = model.inference_forward_greedy(
                src_id, post_start_id, src_mask, src_len, 
                max_len=50, tok_dist=None,
                rel_ids=rel_ids, pos_ids=pos_ids
            )
    
        # Convert output IDs back to tokens
        output_tokens = []
        for id_val in output_ids[0]:
            if id_val in tokenizer.ids_to_tokens:
                output_tokens.append(tokenizer.ids_to_tokens[id_val])
            else:
                output_tokens.append('[UNK]')
    
        # Join tokens into a sentence
        output_sentence = ' '.join(output_tokens).replace(' ##', '').replace(' [CLS]', '').replace('[SEP]', '')
        print(f"\nNeutralized: {output_sentence}")
    
        # Print the bias probabilities for each token
        print("\nBias probabilities for each token:")
        biased_tokens = []
        for token, prob in zip(tokens, bias_probs[0][:len(tokens)]):
            bias_level = "BIASED" if prob > 0 else "neutral"
            print(f"{token}: {prob:.4f} - {bias_level}")
            if prob > 0:
                biased_tokens.append(token)
                
        if biased_tokens:
            print(f"\nBiased tokens identified: {' '.join(biased_tokens)}")
        else:
            print("\nNo biased tokens identified in this sentence.")
            
    except Exception as e:
        print(f"Error during joint inference: {e}")
        print("Trying alternative approach with tagger only...")
        
        # Use just the tagging model to identify bias
        with torch.no_grad():
            _, tok_logits = model.tagging_model(
                src_id, attention_mask=1.0 - src_mask, 
                rel_ids=rel_ids, pos_ids=pos_ids
            )
            
            # Get bias probabilities
            tok_probs = tok_logits[:, :, :2]
            bias_probs = tok_probs[:, :, -1]
            bias_probs = bias_probs.masked_fill(src_mask.bool(), 0.0)
            
            print("\nBias probabilities for each token:")
            biased_tokens = []
            for token, prob in zip(tokens, bias_probs[0][:len(tokens)].cpu().numpy()):
                bias_level = "BIASED" if prob > 0 else "neutral"
                print(f"{token}: {prob:.4f} - {bias_level}")
                if prob > 0:
                    biased_tokens.append(token)
            
            if biased_tokens:
                print(f"\nBiased tokens identified: {' '.join(biased_tokens)}")
                print("\nUnable to generate a neutralized version due to model error.")
                print("The model was able to identify bias but couldn't complete the neutralization process.")
            else:
                print("\nNo biased tokens identified in this sentence.")
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Python path: {sys.path}")
    sys.exit(1)

print("\nModel-based bias analysis and neutralization complete.")
