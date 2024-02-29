import math

from collections import Counter
from utils import tokenize, evaluate_model


# returns frequencies of the terms in the input text
def get_freq(data):
    # key: tokens value: their frequency in all songs belonging to a genre
    dic_term_frequency = Counter()

    for line in data:
        tokens = tokenize(line)
        dic_term_frequency.update(tokens)

    return dic_term_frequency


# transforms term frequencies into probabilities
def freq_to_prob(dic_term_frequency):
    total_tokens = sum(dic_term_frequency.values())
    # add 1 to make sure that the logs return positive value
    dic_term_prob = {term: (f + 1.0) / (total_tokens + len(dic_term_frequency)) + 1
                     for term, f in dic_term_frequency.items()}
    return dic_term_prob


class UnigramModel:
    def __init__(self):
        self.model = {}

    # trains the model for a specific genre
    def train_for_genre(self, genre, data):
        dic_term_frequency = get_freq(data)
        self.model[genre] = freq_to_prob(dic_term_frequency)

    # calculates probability for a specified genre
    def calculate_probability(self, genre, input_text):
        prob = 0.0
        input_text = tokenize(input_text)
        for term in input_text:
            prob += math.log10(self.model[genre].get(term, 1))

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
    model = UnigramModel()
    evaluate_model(model)
    return


if __name__ == "__main__":
    main()
