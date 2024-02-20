import math
import os

from collections import defaultdict
from UnigramModel import tokenize


def read_files(file_list):
    # key: tokens value: their frequency in all songs belonging to a genre
    dic_term_frequency = defaultdict(int)
    dic_pairs_frequency = defaultdict(int)
    for file in file_list:
        with open(file, 'r') as rfile:
            tokens = []
            for line in rfile:
                tokens.extend(tokenize(line))

            for token in tokens:
                dic_term_frequency[token] += 1

            for curr_token, next_token in zip(tokens, tokens[1:]):
                dic_pairs_frequency[(curr_token, next_token)] += 1

    return dic_term_frequency, dic_pairs_frequency


def freq_to_prob(dic_term_frequency, dic_pairs_frequency):
    dic_term_prob = {}

    total_terms = sum(dic_term_frequency.values())
    for pair in dic_pairs_frequency:
        curr_term, _ = pair
        dic_term_prob[pair] = (dic_pairs_frequency[pair] + 1) / (dic_term_frequency[curr_term] + total_terms)
        dic_term_prob[pair] += 1

    return dic_term_prob


class BigramModel:
    def __init__(self):
        self.model = {}

    def fit(self, file_dic):
        for genre in file_dic:
            dic_term_frequency, dic_pair_frequency = read_files(file_dic[genre])
            self.model[genre] = freq_to_prob(dic_term_frequency, dic_pair_frequency)

    def calculate_probability(self, genre, input_text):
        prob = 0.0
        input_text = tokenize(input_text)
        for pair in zip(input_text, input_text[1:]):
            prob += math.log10(self.model[genre].get(pair, 1))

        return prob

    def predict(self, input_text):
        probabilities = []
        for genre in self.model:
            prob = self.calculate_probability(genre, input_text)
            probabilities.append((genre, prob))

        return max(probabilities, key=lambda x: x[1])


def main():
    text = """You used to call me on my cell phone
    Late night when you need my love
    Call me on my cell phone"""

    file_dic = {}
    for genre in os.listdir("Lyrics"):
        file_dic[genre] = []
        for song in os.listdir(f"Lyrics/{genre}/"):
            file_dic[genre].append(f"Lyrics/{genre}/{song}")

    print("model")
    model = BigramModel()
    model.fit(file_dic)
    pred = model.predict(text)
    print(pred)

    return


if __name__ == "__main__":
    main()
