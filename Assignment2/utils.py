import os
import random
import re
import nltk
import pandas as pd

# from transformers import AutoTokenizer
from nltk import word_tokenize
from nltk.corpus import stopwords

# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

combined_text_file = "combined_text.txt"


def tokenize(text):
    text = text.strip().lower()
    tokens = word_tokenize(text)
    # tokens = tokenizer.tokenize(text)
    # tokens = [word for word in tokens if word.lower() not in stop_words]
    return tokens


def combine_text_files(input_dir, output_file=combined_text_file):
    with open(os.path.join(input_dir, output_file), "w", encoding="utf-8") as outfile:
        for filename in os.listdir(input_dir):
            if not filename.endswith(".txt"):
                continue  # Skip non-text files
            filepath = os.path.join(input_dir, filename)
            with open(filepath, "r", encoding="utf-8") as infile:
                outfile.write(infile.read())


def combine_all_text_files(path="Lyrics"):
    for genre in os.listdir(path):
        combine_text_files(f"{path}/{genre}")


def get_data(file_name):
    with open(file_name, "r") as rfile:
        data = rfile.read().splitlines()
    return data


def split_data(data_file, train_ratio=0.9):
    """Splits text data into training and validation sets.

    Args:
      data_file: The path to the file containing the combined text data.
      train_ratio: The proportion of data to use for training (default: 0.9).

    Returns:
      A tuple containing the training data and validation data (both as strings).
    """
    with open(data_file, "r", encoding="utf-8") as f:
        data = f.read()

    random.seed(42)
    data_lines = data.splitlines()
    random.shuffle(data_lines)  # Shuffle data for randomness

    split_index = int(len(data_lines) * train_ratio)
    training_data = data_lines[:split_index]
    validation_data = data_lines[split_index:]

    return training_data, validation_data


def add_space_between_capital_letters(input_file, output_file):
    with open(input_file, 'r') as f:
        text = f.read()

    # Use regular expression to find occurrences where a capital letter follows a non-capital letter
    pattern = re.compile(r'(?<=[a-z])(?=[A-Z])')

    # Insert a space at each identified position
    modified_text = re.sub(pattern, ' ', text)

    with open(output_file, 'w') as f:
        f.write(modified_text)


def evaluate_model(model):
    # prepare data
    combine_all_text_files()
    add_space_between_capital_letters('test.tsv', 'test2.tsv')

    df = pd.read_csv('test2.tsv', sep='\t', header=0)
    songs = df['Text']
    labels = df['Genre']

    print("Evaluating model:")
    for genre in os.listdir("Lyrics"):
        data = get_data(f"Lyrics/{genre}/{combined_text_file}")
        model.train_for_genre(genre, data)
    predicted_labels = model.predict(songs)

    for i in range(len(df)):
        print(f"Song_id:{i+1}, true_label: {labels[i]}, predicted_label: {predicted_labels[i]}")

    correct = sum(1 for true, pred in zip(labels, predicted_labels) if true == pred)
    print(f"Accuracy: {correct}/{len(labels)} = {(correct/len(labels)):.3}%")

