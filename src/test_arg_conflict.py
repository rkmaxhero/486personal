import sys
import argparse

# Print original sys.argv
print(f"Original sys.argv: {sys.argv}")

# Create an argparse parser to simulate the imported modules
parser = argparse.ArgumentParser(description='Test argparse conflict')
parser.add_argument('--debias_weight', type=float, default=1.0, help='Debiasing weight parameter')
parser.add_argument('--checkpoint', type=str, default='model.ckpt', help='Model checkpoint')
parser.add_argument('--working_dir', type=str, default='cache', help='Working directory')

# Try to parse args - this would fail if unexpected arguments are present
try:
    args = parser.parse_args()
    print(f"Parsed args: {args}")
except SystemExit as e:
    print(f"Parser triggered SystemExit with code {e.code}")

# Now try again but with allow_unknown=True
parser = argparse.ArgumentParser(description='Test with unknown args')
parser.add_argument('--debias_weight', type=float, default=1.0)
parser.add_argument('--checkpoint', type=str, default='model.ckpt')

try:
    # Parse known args only
    known_args, unknown_args = parser.parse_known_args()
    print(f"Known args: {known_args}")
    print(f"Unknown args: {unknown_args}")
    
    # Try to manually extract the sentence
    # Find where our input sentence might be
    if len(unknown_args) > 0:
        sentence = " ".join([arg for arg in unknown_args if not arg.startswith('--')])
        print(f"Extracted sentence: {sentence}")
except Exception as e:
    print(f"Error with parse_known_args: {e}")

# Try a completely manual approach
manual_sentence = ""
i = 1
while i < len(sys.argv):
    arg = sys.argv[i]
    if arg.startswith('--'):
        i += 2  # Skip this arg and its value
    else:
        manual_sentence += arg + " "
        i += 1

if manual_sentence:
    print(f"Manually extracted sentence: {manual_sentence.strip()}")
else:
    print("No sentence found with manual extraction") 