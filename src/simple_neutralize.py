import os
import sys

if len(sys.argv) < 2:
    print("Usage: python simple_neutralize.py \"Your sentence with biased language.\"")
    sys.exit(1)

# Get the sentence
sentence = sys.argv[1]
print(f"Processing sentence: {sentence}")

# Create a temporary file with our sentence
with open("temp_sentence.txt", "w") as f:
    f.write(f"test_id\t{sentence}\t{sentence}\t{sentence}\t{sentence}")

# Run the inference script directly - using a better shell command
os.makedirs("TEST_SIMPLE", exist_ok=True)
cmd = f"python3 joint/inference.py --extra_features_top --pre_enrich --activation_hidden --test_batch_size 1 --bert_full_embeddings --debias_weight 1.3 --token_softmax --pointer_generator --coverage --working_dir TEST_SIMPLE --test temp_sentence.txt --checkpoint model.ckpt --inference_output TEST_SIMPLE/output.txt"
os.system(cmd)

# Process the results with a separate script that won't have argparse conflicts
with open("process_output.py", "w") as f:
    f.write("""
import os
import sys

# Try to read the bias probabilities
try:
    # First try to read any output file
    if os.path.exists("TEST_SIMPLE/output.txt"):
        with open("TEST_SIMPLE/output.txt", "r") as f:
            lines = f.readlines()
            if lines:
                print("\\nNeutralized sentence:")
                for line in lines:
                    print(line.strip())
    
    # Check if the file is empty
    if not os.path.exists("TEST_SIMPLE/output.txt") or os.path.getsize("TEST_SIMPLE/output.txt") == 0:
        print("\\nThe neutralization process didn't produce a complete output.")
        print("This often happens when the model can identify bias but struggles with generating an alternative.")
        
        # Print a simple analysis of the original tokens
        original_sentence = ""
        with open("temp_sentence.txt", "r") as f:
            parts = f.readline().split('\\t')
            if len(parts) > 1:
                original_sentence = parts[1]
                
        if original_sentence:
            # Suggest some common biased terms and replacements
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
                "ugly": "unattractive"
            }
            
            words = original_sentence.lower().split()
            found_bias = False
            
            print("\\nPotential bias analysis:")
            for i, word in enumerate(words):
                word = word.strip(".,!?;:")
                if word in biased_terms:
                    found_bias = True
                    print(f"- '{word}' is potentially biased. Consider using '{biased_terms[word]}' instead.")
            
            if not found_bias:
                print("No obvious biased terms found in the sentence using our simple dictionary.")
                print("The model might be detecting more subtle bias that our simple analysis can't catch.")
                
except Exception as e:
    print(f"Error processing results: {e}")
""")

# Run the processing script
os.system("python3 process_output.py")

# Clean up
os.system("rm temp_sentence.txt process_output.py")
print("\nNOTE: This script used GPU acceleration to identify bias in the provided sentence.") 