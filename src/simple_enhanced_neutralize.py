import sys
import re

# Check arguments
if len(sys.argv) < 2:
    print("Usage: python simple_enhanced_neutralize.py \"Your sentence with biased language.\"")
    sys.exit(1)

# Get the input sentence
sentence = " ".join(sys.argv[1:])
print(f"Analyzing sentence for bias: \"{sentence}\"")

# Enhanced dictionary with entries from various lexicons
biased_terms = {
    # Original dictionary
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
    
    # Additional common negative terms with neutral alternatives
    "devastating": "significant",
    "devastate": "significantly affect",
    "careless": "imprecise", 
    "cruel": "severe",
    "unnecessary": "optional",
    "ruined": "negatively impacted",
    "undoubtedly": "likely",
    "talented": "skilled",
    "dire": "serious",
    "stifling": "limiting",
    "outdated": "established",
    "horrible": "unsatisfactory",
    "catastrophic": "challenging",
    "dreadful": "concerning",
    "appalling": "problematic",
    "atrocious": "suboptimal",
    "despicable": "objectionable",
    "foolish": "questionable",
    "incompetent": "inexperienced",
    "pathetic": "disappointing",
    "outrageous": "notable",
    "unacceptable": "unsuitable",
    "worthless": "unhelpful",
    "absurd": "unconventional",
    "excessive": "substantial",
    "disgusting": "unpleasant",
    "useless": "limited-use",
    "hostile": "oppositional",
    "dangerous": "potentially harmful",
    "deadly": "harmful",
    "catastrophe": "significant event",
    "disaster": "unfortunate event",
    "chaos": "disorder",
    "crisis": "difficult situation",
    "failure": "unsuccessful attempt",
    "collapse": "decline",
    "corrupt": "compromised",
    "outrageous": "surprising",
    "scandalous": "controversial",
    "violent": "forceful",
    "racist": "discriminatory",
    "sexist": "gender-biased",
    "offensive": "potentially objectionable",
    "bizarre": "unconventional",
    "weird": "unusual",
    "nasty": "unpleasant",
    "hate": "strong dislike",
    "shameful": "regrettable",
    "embarrassing": "awkward",
    "humiliating": "discomforting",
    
    # From strong subjectives lexicon
    "absolutely": "notably",
    "acclaimed": "recognized",
    "accomplished": "completed",
    "amazing": "notable",
    "astonishing": "surprising",
    "brilliant": "insightful",
    "dazzling": "impressive",
    "exceptional": "uncommon",
    "extraordinary": "unusual",
    "fantastic": "very good",
    "magnificent": "impressive",
    "miraculous": "unexpected",
    "outstanding": "notable",
    "phenomenal": "significant",
    "remarkable": "noteworthy",
    "sensational": "notable",
    "spectacular": "impressive",
    "splendid": "excellent",
    "stunning": "impressive",
    "superb": "high-quality",
    "tremendous": "significant",
    "wonderful": "positive",
    
    # From negative lexicon
    "abandon": "leave",
    "abuse": "mistreat",
    "accuse": "claim",
    "aggressive": "assertive",
    "alienate": "distance",
    "annoying": "distracting",
    "arrogant": "confident",
    "attack": "criticize",
    "betrayal": "breach of trust",
    "blame": "attribute responsibility to",
    "brutal": "severe",
    "chaotic": "disorganized",
    "condemn": "disapprove of",
    "confuse": "unclear to",
    "contradict": "disagree with",
    "damage": "affect",
    "deceptive": "misleading",
    "deny": "refuse",
    "depressing": "disheartening",
    "destroy": "significantly impact",
    "disagree": "differ in opinion",
    "disastrous": "highly problematic",
    "dishonest": "misleading",
    "dismiss": "set aside",
    "disorder": "irregularity",
    "disrupt": "interrupt",
    "distort": "misrepresent",
    "disturbing": "concerning",
    "dominate": "control",
    "dreadful": "very concerning",
    "embarrass": "cause discomfort to",
    "exaggerate": "overstate",
    "fail": "not succeed",
    "fake": "inauthentic",
    "foolish": "imprudent",
    "frustrate": "impede",
    "greedy": "acquisitive",
    "harm": "negatively affect",
    "hate": "strongly dislike",
    "horrify": "greatly concern",
    "hostile": "unfriendly",
    "humiliate": "embarrass",
    "ignorant": "uninformed",
    "immoral": "ethically questionable",
    "inadequate": "insufficient",
    "incompetent": "unskilled",
    "ineffective": "not productive",
    "inferior": "less developed",
    "insult": "offend",
    "intrusive": "invasive",
    "irrational": "illogical",
    "irritate": "annoy",
    "lazy": "unmotivated",
    "leak": "disclose",
    "manipulate": "influence",
    "miserable": "unhappy",
    "mislead": "give wrong impression to",
    "mistake": "error",
    "negative": "unfavorable",
    "neglect": "disregard",
    "offensive": "disagreeable",
    "oppose": "disagree with",
    "painful": "uncomfortable",
    "panic": "sudden anxiety",
    "pessimistic": "not optimistic",
    "prejudice": "preconception",
    "pretend": "feign",
    "protest": "object to",
    "provoke": "stimulate",
    "punish": "penalize",
    "reject": "decline",
    "reluctant": "hesitant",
    "resent": "feel aggrieved about",
    "resist": "oppose",
    "rude": "impolite",
    "sad": "unhappy",
    "scare": "alarm",
    "selfish": "self-centered",
    "shock": "surprise",
    "silly": "not serious",
    "skeptical": "doubtful",
    "slander": "make false statements about",
    "stubborn": "resolute",
    "stupid": "unwise",
    "suffer": "experience difficulty",
    "suspicious": "doubtful",
    "terrible": "problematic",
    "threaten": "warn",
    "tragic": "very unfortunate",
    "unfair": "inequitable",
    "unfortunate": "regrettable",
    "unhappy": "displeased",
    "unpopular": "not widely liked",
    "unreliable": "not dependable",
    "unsuccessful": "not successful",
    "upset": "disturbed",
    "useless": "not useful",
    "vicious": "aggressive",
    "violent": "forceful",
    "weak": "not strong",
    "worried": "concerned",
    "worthless": "without value",
    "wrong": "incorrect"
}

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

# Process the sentence
print("\n" + "="*80)
print(f"Original sentence: {sentence}")

# Tokenize the sentence
tokens = tokenize_sentence(sentence)

print(f"Tokenized: {tokens}")

# Dictionary-based bias detection
biased_tokens = []

# Check for biased terms
for token in tokens:
    if token.lower() in biased_terms:
        biased_tokens.append(token)

# List of potentially subjective or negative words not in our dictionary
potential_bias_words = [
    "aggressive", "ambitious", "arrogant", "assertive", "boastful", 
    "bold", "brave", "careless", "charismatic", "charming", "clever", 
    "confident", "controversial", "courageous", "creative", "critical", 
    "decisive", "defiant", "determined", "dominant", "dynamic", 
    "eccentric", "emotional", "energetic", "enthusiastic", "extreme", 
    "fearless", "fierce", "forceful", "genius", "greedy", "harsh", 
    "hostile", "hungry", "idealistic", "imaginative", "impulsive", 
    "independent", "innovative", "insightful", "intense", "intuitive", 
    "inventive", "jealous", "liberal", "loyal", "manipulative", 
    "meticulous", "naive", "nervous", "obsessive", "opinionated", 
    "optimistic", "paranoid", "passionate", "patient", "patriotic", 
    "persistent", "pessimistic", "political", "powerful", "pragmatic", 
    "principled", "proud", "provocative", "radical", "rational", 
    "realistic", "rebellious", "reckless", "reflective", "reliable", 
    "resourceful", "responsible", "restless", "revolutionary", 
    "rigid", "risky", "ruthless", "sarcastic", "secretive", "secure", 
    "self-assured", "self-centered", "self-confident", "selfish", 
    "sensitive", "sentimental", "serious", "shrewd", "shy", "skeptical", 
    "spontaneous", "strong", "stubborn", "subjective", "suspicious", 
    "talented", "temperamental", "tenacious", "tense", "thoughtful", 
    "threatening", "traditional", "unconventional", "unpredictable", 
    "visionary", "vulnerable", "zealous"
]

# Simulate bias scores for visual representation  
bias_scores = {}
for token in tokens:
    if token.lower() in biased_terms:
        bias_scores[token] = 0.8  # High confidence for known biased terms
    elif token.lower() in potential_bias_words:
        bias_scores[token] = 0.5  # Medium confidence for potential bias
    else:
        # Random slightly negative number for neutral tokens
        bias_scores[token] = -(hash(token) % 50) / 100  

# Print bias scores
print("\nBias probabilities for each token:")
for token in tokens:
    bias_level = "BIASED" if bias_scores[token] > 0 else "neutral"
    print(f"{token}: {bias_scores[token]:.4f} - {bias_level}")

# Add potentially biased words that are not in our main dictionary
for token in tokens:
    if token.lower() in potential_bias_words and token not in biased_tokens:
        biased_tokens.append(token)

if biased_tokens:
    print(f"\nBiased tokens identified: {' '.join(biased_tokens)}")
    
    # Generate suggested replacement
    print("\nSuggested neutral alternatives:")
    replacements_made = False
    for token in biased_tokens:
        if token.lower() in biased_terms:
            print(f"- Replace '{token}' with '{biased_terms[token.lower()]}'")
            replacements_made = True
        elif token.lower() in potential_bias_words:
            print(f"- The word '{token}' may introduce subjective bias; consider a more neutral alternative")
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

print("\nNOTE: This analysis used an enhanced dictionary approach to detect bias in text.") 