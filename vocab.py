import pandas as pd
import re

# Step 1: Load the word list from the text file into a set for efficient lookups
def load_word_set(file_path):
    with open(file_path, 'r') as file:
        return {line.strip().lower() for line in file}

# Assume the text file is named 'wordlist.txt' and is located in the same directory
word_set = load_word_set('words.txt')

# Step 2: Define the function to process your quotes and update the vocab list
def get_vocab(data, vocab, word_set):
    pattern = r"[^\w\s]"
    for quote in data:
        text = re.sub(pattern, " ", quote)
        words = text.lower().split()
        for word in words:
            word = word.strip()
            if word in word_set and word not in vocab:
                vocab.append(word)
    return sorted(vocab)

# Assuming your DataFrame is already loaded
file = pd.read_csv("clean_dataset.csv")

file["Q10"].fillna(" ", inplace=True)

# Initialize an empty vocab list
vocab = []

# Step 3: Update vocab by processing the quotes from the DataFrame's specified column
vocab = get_vocab(file["Q10"].values, vocab, word_set)

# Now vocab contains a sorted list of unique words from your quotes that are also in the text file
print(len(vocab))
