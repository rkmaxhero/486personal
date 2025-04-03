import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer
import numpy as np
import sys

# The script takes a sentence as a command line argument
if len(sys.argv) < 2:
    print("Usage: python neutralize_any_sentence.py \"Your sentence with biased language.\"")
    sys.exit(1)

# Get the sentence from command line
sentence = sys.argv[1]

# Import required modules
sys.path.append('.')
from shared.constants import CUDA
from shared.args import ARGS
import joint.model as joint_model
import seq2seq.model as seq2seq_model
import tagging.model as tagging_model

# Configure the model
ARGS.pointer_generator = True
ARGS.bert_full_embeddings = True
ARGS.extra_features_top = True
ARGS.working_dir = 'cache'
ARGS.coverage = True

print(f"Original sentence: {sentence}")

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
src_len = torch.tensor([len(token_ids)], dtype=torch.long)
post_start_id = tok2id['[CLS]']

# Create rel_ids and pos_ids (needed by the model)
rel_ids = torch.zeros_like(src_id)
pos_ids = torch.zeros_like(src_id)

# Print tokens for debugging
print(f"Tokenized: {tokens}")

# Initialize models
print("Initializing models...")
debias_model = seq2seq_model.PointerSeq2Seq(
    vocab_size=len(tok2id), hidden_size=ARGS.hidden_size,
    emb_dim=768, dropout=0.2, tok2id=tok2id)

tagging_model = tagging_model.BertForMultitaskWithFeaturesOnTop.from_pretrained(
    'bert-base-uncased',
    cls_num_labels=ARGS.num_categories,
    tok_num_labels=ARGS.num_tok_labels,
    cache_dir='cache',
    tok2id=tok2id)

# Create the joint model
model = joint_model.JointModel(debias_model=debias_model, tagging_model=tagging_model)

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
    # Run inference
    with torch.no_grad():
        print("Running inference...")
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
    print(f"Neutralized: {output_sentence}")

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
    print(f"Error during inference: {e}")
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
            print("\nYou may want to replace these biased words with more neutral alternatives.")
        else:
            print("\nNo biased tokens identified in this sentence.") 