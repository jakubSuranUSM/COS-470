import math

from collections import Counter
from utils import tokenize, evaluate_model


# returns frequencies of the terms in the input text
def get_freq(data):
    # key: tokens value: their frequency in all songs belonging to a genre
    dic_term_frequency = Counter()
    dic_pairs_frequency = Counter()

    for line in data:
        tokens = tokenize(line)
        # Update unigram frequencies
        dic_term_frequency.update(tokens)

        # Create bigrams with ('<START>', first_token) and (last_token, '<END>')
        if len(tokens) >= 1:
            bigrams = [('<START>', tokens[0])] + list(zip(tokens[:-1], tokens[1:])) + [(tokens[-1], '<END>')]
            dic_pairs_frequency.update(bigrams)

    return dic_term_frequency, dic_pairs_frequency


# transforms term frequencies into probabilities
def freq_to_prob(dic_term_frequency, dic_pairs_frequency):
    total_terms = sum(dic_term_frequency.values())
    dic_term_prob = {pair: (f + 1) / (dic_term_frequency[pair[0]] + total_terms) + 1
                     for pair, f in dic_pairs_frequency.items()}
    return dic_term_prob


class BigramModel:
    def __init__(self):
        self.model = {}

    # trains the model for a specific genre
    def train_for_genre(self, genre, data):
        dic_term_frequency, dic_pair_frequency = get_freq(data)
        self.model[genre] = freq_to_prob(dic_term_frequency, dic_pair_frequency)

    # calculates probability for a specified genre
    def calculate_probability(self, genre, input_text):
        prob = 0.0
        input_text = tokenize(input_text)
        for pair in zip(input_text, input_text[1:]):
            prob += math.log10(self.model[genre].get(pair, 1))

        return prob

    # makes predictions for an array of inputs
    def predict(self, input_text_array):
        results = []
        for input_text in input_text_array:
            probabilities = []
            for genre in self.model:
                prob = self.calculate_probability(genre, input_text)
                probabilities.append((genre, prob))
            results.append(max(probabilities, key=lambda x: x[1])[0])

        return results


# evaluates the model
def main():
    model = BigramModel()
    evaluate_model(model)
    return


if __name__ == "__main__":
    main()
