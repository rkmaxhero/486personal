#!/usr/bin/env python3
import sys
import os
import spacy
from pytorch_pretrained_bert.tokenization import BertTokenizer

def create_test_file(sentence, output_file="bias_data/WNC/rk.test"):
    """
    Formats a single sentence for bias neutralization and writes to test file.
    Aligns the BERT tokenization with the POS/dependency tags from spaCy.
    """
    # Load BERT tokenizer and spaCy model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    
    # Align tokenizations: for each spaCy token, get BERT wordpieces and replicate tags.
    b_tokens = []          # BERT tokens list
    pos_tags_aligned = []  # aligned POS tags
    dep_tags_aligned = []  # aligned dependency tags
    
    for token in doc:
        wordpieces = tokenizer.tokenize(token.text)
        b_tokens.extend(wordpieces)
        pos_tags_aligned.extend([token.pos_] * len(wordpieces))
        dep_tags_aligned.extend([token.dep_] * len(wordpieces))
    
    # Join lists into strings
    tokenized_text = " ".join(b_tokens)
    pos_tags_text = " ".join(pos_tags_aligned)
    dep_tags_text = " ".join(dep_tags_aligned)
    
    # Debug prints to check alignment:
    print("Tokenized text:", tokenized_text)
    print("Token count:", len(b_tokens))
    print("POS tags:", pos_tags_text)
    print("POS tag count:", len(pos_tags_aligned))
    print("Dependency tags:", dep_tags_text)
    print("Dependency tag count:", len(dep_tags_aligned))
    
    # Format the TSV line with exactly 7 fields:
    # ID, tokenized source, tokenized target, raw source, raw target, POS tags, dependency tags
    id_num = "12345678"
    tsv_line = f"{id_num}\t{tokenized_text}\t{tokenized_text}\t{sentence}\t{sentence}\t{pos_tags_text}\t{dep_tags_text}"
    
    

    # Ensure directory exists and write the file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        f.write(tsv_line)
    
    print(f"Sentence formatted and written to {output_file}")
    print("Run the model with: sh runner.sh")
    return output_file

if __name__ == "__main__":
    if len(sys.argv) > 1:
        sentence = " ".join(sys.argv[1:])
    else:
        sentence = input("Enter a sentence to neutralize: ")
    create_test_file(sentence)