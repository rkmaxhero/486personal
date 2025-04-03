# File: /home/rounaq/486personal/src/joint/utils.py

from tqdm import tqdm
import torch
import torch.nn as nn
import sys; sys.path.append(".")
from shared.args import ARGS
from shared.constants import CUDA
import seq2seq.utils as seq2seq_utils
from simplediff import diff
from shared.data import get_tok_labels

def detokenize_wordpieces(tokens):
    """
    Merge BERT wordpiece tokens into a human‑readable string.
    e.g. ['the', 'ho', '##rre', '##ndo', '##us', 'rest', '##ura', '##unt', ...]
    becomes "the horrendous restaurant ..."
    """
    words = []
    for token in tokens:
        if token.startswith("##") and words:
            # Append to the previous word
            words[-1] += token[2:]
        else:
            words.append(token)
    return " ".join(words)

def train_for_epoch(model, dataloader, optimizer, debias_loss_fn, tagging_loss_fn=None, ignore_tagger=False, coverage=False):
    model.train()
    losses = []
    for step, batch in enumerate(tqdm(dataloader)):
        if ARGS.debug_skip and step > 2:
            continue

        if CUDA: 
            batch = tuple(x.cuda() for x in batch)
        (
            pre_id, pre_mask, pre_len, 
            post_in_id, post_out_id, 
            pre_tok_label_id, post_tok_label_id,
            rel_ids, pos_ids, categories
        ) = batch      
        post_log_probs, post_probs, tok_probs, tok_logits, attns, coverages = model(
            pre_id, post_in_id, pre_mask, pre_len, pre_tok_label_id,
            rel_ids=rel_ids, pos_ids=pos_ids, categories=categories, ignore_tagger=ignore_tagger)

        loss = debias_loss_fn(post_log_probs, post_out_id, post_tok_label_id)
        
        if tagging_loss_fn is not None and ARGS.tag_loss_mixing_prob > 0:
            tok_loss = tagging_loss_fn(tok_logits, pre_tok_label_id, apply_mask=pre_tok_label_id)
            loss = loss + (ARGS.tag_loss_mixing_prob * tok_loss)

        if coverage:
            cov_loss = seq2seq_utils.coverage_loss(attns, coverages)
            loss = loss + ARGS.coverage_lambda * cov_loss

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        optimizer.step()
        model.zero_grad()
        
        losses.append(loss.detach().cpu().numpy())

    return losses

def run_eval(model, dataloader, tok2id, out_file_path, max_seq_len, beam_width=1):
    id2tok = {x: tok for (tok, x) in tok2id.items()}

    out_file = open(out_file_path, 'w')

    losses = []
    hits = []
    preds, golds, srcs = [], [], []
    for step, batch in enumerate(tqdm(dataloader)):
        if ARGS.debug_skip and step > 2:
            continue

        if CUDA:
            batch = tuple(x.cuda() for x in batch)
        (
            pre_id, pre_mask, pre_len, 
            post_in_id, post_out_id, 
            pre_tok_label_id, _,
            rel_ids, pos_ids, categories
        ) = batch

        post_start_id = tok2id['行']
        max_len = min(max_seq_len, pre_len[0].detach().cpu().numpy() + 10)

        with torch.no_grad():
            predicted_toks, predicted_probs = model.inference_forward(
                pre_id, post_start_id, pre_mask, pre_len, max_len, pre_tok_label_id,
                rel_ids=rel_ids, pos_ids=pos_ids, categories=categories,
                beam_width=beam_width)
            
        # Decode full gold and predicted sequences (using id2tok)
        gold_seq_full = [id2tok[x.item()] for x in post_out_id[0]]
        pred_seq_full = [id2tok[x] for x in predicted_toks[0]]

        # Define trimming functions
        def trim_gold_sequence(seq):
            if seq and seq[0] == '行':
                seq = seq[1:]
            while seq and seq[-1] in ['止', '[PAD]']:
                seq = seq[:-1]
            return seq

        def trim_pred_sequence(seq):
            if seq and seq[0] == '行':
                seq = seq[1:]
            if '止' in seq:
                seq = seq[:seq.index('止')]
            while seq and seq[-1] == '[PAD]':
                seq = seq[:-1]
            return seq

        # Trim the sequences
        gold_seq = trim_gold_sequence(gold_seq_full)
        pred_seq = trim_pred_sequence(pred_seq_full)

        print("Trimmed Gold tokens:", gold_seq)
        print("Trimmed Predicted tokens:", pred_seq)
        
        # Compute diff and extract token-level labels
        tok_diff = diff(gold_seq, pred_seq)
        pre_labels, post_labels = get_tok_labels(tok_diff)
        print("Gold token labels:", pre_labels)
        print("Predicted token labels:", post_labels)
        
        new_hits, new_preds, new_golds, new_srcs = seq2seq_utils.dump_outputs(
            pre_id.detach().cpu().numpy(), 
            post_out_id.detach().cpu().numpy(), 
            predicted_toks, 
            pre_tok_label_id.detach().cpu().numpy(), 
            id2tok, out_file,
            pred_dists=predicted_probs)
        hits += new_hits
        preds += new_preds
        golds += new_golds
        srcs += new_srcs
    out_file.close()

    return hits, preds, golds, srcs
