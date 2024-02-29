import os
import numpy as np

from BigramModel import BigramModel
from UnigramModel import UnigramModel
from utils import split_data, combined_text_file, evaluate_model

best_lambda = 0.0


class MixedModel:
    def __init__(self, model_lambda=best_lambda):
        self.unigram_model = UnigramModel()
        self.bigram_model = BigramModel()
        self.model_lambda = model_lambda

    def change_lambda(self, lambda_value):
        self.model_lambda = lambda_value

    def train_for_genre(self, genre, data):
        self.unigram_model.train_for_genre(genre, data)
        self.bigram_model.train_for_genre(genre, data)

    def predict(self, input_text):
        results = []
        for text in input_text:
            probabilities = []
            for genre in self.unigram_model.model:
                unigram_prob = self.unigram_model.calculate_probability(genre, text)
                bigram_prob = self.bigram_model.calculate_probability(genre, text)

                prob = self.model_lambda * unigram_prob + (1 - self.model_lambda) * bigram_prob
                probabilities.append((genre, prob))

            results.append(max(probabilities, key=lambda x: x[1])[0])
        return results


def prepare_labels(dic_val):
    lyrics = []
    labels = []
    for genre, data in dic_val.items():
        for chunk in np.array_split(data, 5):
            lyrics.append(" ".join(chunk))
            labels.append(genre)

    return lyrics, labels


def find_lambda(model, dic_val):
    top_lambda = None
    top_accuracy = 0
    songs, labels = prepare_labels(dic_val)
    for i in np.linspace(0.00, 1, 21):
        model_lambda = i
        model.change_lambda(model_lambda)
        predicted_labels = model.predict(songs)
        accuracy = sum(1 for true, pred in zip(labels, predicted_labels) if true == pred) / len(labels)
        print(f"lambda: {model_lambda}, accuracy: {accuracy}")
        if accuracy > top_accuracy:
            top_accuracy = accuracy
            top_lambda = model_lambda

    return top_lambda


def main():
    model = MixedModel()
    dic_val = {}
    print("Finding best lambda:")
    for genre in os.listdir("Lyrics"):
        train, val = split_data(f"Lyrics/{genre}/{combined_text_file}")
        model.train_for_genre(genre, train)
        dic_val[genre] = val

    best_lambda = find_lambda(model, dic_val)
    print(f"Best lambda: {best_lambda}\n")

    model = MixedModel(best_lambda)
    evaluate_model(model)


if __name__ == "__main__":
    main()
