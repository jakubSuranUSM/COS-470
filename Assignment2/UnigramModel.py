import os
import math

import nltk
from transformers import AutoTokenizer
from nltk.corpus import stopwords
from collections import defaultdict

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))


def read_files(file_list):
    # key: tokens value: their frequency in all songs belonging to a genre
    dic_term_frequency = defaultdict(int)

    for file in file_list:
        with open(file, 'r') as rfile:
            for line in rfile:
                tokens = tokenize(line)
                for token in tokens:
                    dic_term_frequency[token] += 1

    return dic_term_frequency


def tokenize(text):
    text = text.strip()
    text = text.lower()
    tokens = tokenizer.tokenize(text)
    tokens = [word for word in tokens if word.lower() not in stop_words]
    return tokens


def freq_to_prob(dic_term_frequency):
    dic_term_prob = {}

    total_terms = sum(dic_term_frequency.values())
    for term in dic_term_frequency:
        dic_term_prob[term] = 1 + (dic_term_frequency[term] * 1.0 / total_terms)

    return dic_term_prob


class UnigramModel:
    def __init__(self):
        self.model = {}

    def fit(self, file_dic):
        for genre in file_dic:
            dic_term_frequency = read_files(file_dic[genre])
            self.model[genre] = freq_to_prob(dic_term_frequency)

    def calculate_probability(self, genre, input_text):
        prob = 0.0
        input_text = tokenize(input_text)
        for term in input_text:
            prob += math.log10(self.model[genre].get(term, 1))

        return prob

    def predict(self, input_text):
        probabilities = []
        for genre in self.model:
            prob = self.calculate_probability(genre, input_text)
            probabilities.append((genre, prob))

        return probabilities
        # return max(probabilities, key=lambda x: x[1])


def main():
    text = """You used to call me on my cell phone
Late night when you need my love
Call me on my cell phone"""

    file_dic = {}
    for genre in os.listdir("Lyrics"):
        file_dic[genre] = []
        for song in os.listdir(f"Lyrics/{genre}/"):
            file_dic[genre].append(f"Lyrics/{genre}/{song}")

    model = UnigramModel()
    model.fit(file_dic)
    print(model.predict(text))

    return


if __name__ == "__main__":
    main()
