#!/usr/bin/env python3
import sys
import spacy
import os

def create_test_file(sentence, output_file="bias_data/WNC/rk.test"):
    """
    Formats a single sentence for bias neutralization and writes to test file.
    """
    # Load spaCy English model for POS tagging and dependency parsing
    nlp = spacy.load("en_core_web_sm")
    
    # Process the sentence
    doc = nlp(sentence)
    
    # Create tokenized version (lowercase with spaces around punctuation)
    tokenized = []
    for token in doc:
        if token.is_punct:
            tokenized.append(token.text)
        else:
            tokenized.append(token.text.lower())
    
    tokenized_text = " ".join(tokenized)
    
    # Create POS tags
    pos_tags = [token.pos_ for token in doc]
    pos_tags_text = " ".join(pos_tags)
    
    # Create dependency parse tags
    dep_tags = [token.dep_ for token in doc]
    dep_tags_text = " ".join(dep_tags)
    
    # Format the TSV line
    id_num = "12345678"  # Arbitrary ID number
    tsv_line = f"{id_num}\t{tokenized_text}\t{tokenized_text}\t{sentence}\t{sentence}\t{pos_tags_text}\t{dep_tags_text}"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Write to the test file
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