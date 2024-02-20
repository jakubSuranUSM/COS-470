import os
import random
import numpy as np

from BigramModel import BigramModel
from UnigramModel import UnigramModel


class MixedModel:
    def __init__(self, model_lambda):
        self.unigram_model = UnigramModel()
        self.bigram_model = BigramModel()
        self.model_lambda = model_lambda

    def predict(self, input_text):
        probabilities = []
        for genre in self.unigram_model.model:
            unigram_prob = self.unigram_model.calculate_probability(genre, input_text)
            bigram_prob = self.bigram_model.calculate_probability(genre, input_text)

            prob = self.model_lambda * unigram_prob + (1 - self.model_lambda) * bigram_prob
            probabilities.append((genre, prob))

        return max(probabilities, key=lambda x: x[1])

    def fit(self, file_dic):
        self.unigram_model.fit(file_dic)
        self.bigram_model.fit(file_dic)


def train_val_split(dir_path):
    # generate two random positions
    rand_pos = []
    rand_pos.append(random.randint(0, 20))
    i = 1
    while i < 2:
        rand_int = random.randint(0, 20)
        if rand_int not in rand_pos:
            rand_pos.append(rand_int)
            i += 1

    train = {}
    val = {}

    for genre in os.listdir(dir_path):
        train[genre] = []
        val[genre] = []
        for j, song in enumerate(os.listdir(f"{dir_path}/{genre}/")):
            if j in rand_pos:
                val[genre].append(f"{dir_path}/{genre}/{song}")
            else:
                train[genre].append(f"{dir_path}/{genre}/{song}")

    return train, val


def prepare_labels(file_dic):
    lyrics = []
    labels = []
    for genre in file_dic:
        for song in file_dic[genre]:
            with open(song, 'r') as rfile:
                i = 1
                text = ""
                for line in rfile:
                    text += line
                    if i >= 3:
                        text = text.replace("\n", " ")
                        lyrics.append(text)
                        labels.append(genre)
                        text = ""
                        i = 0
                    i += 1

    return lyrics, labels


def get_accuracy(model, songs, labels):
    correct_pred = 0
    for song, label in zip(songs, labels):
        pred, prob = model.predict(song)
        if pred == label:
            correct_pred += 1

    return correct_pred/len(songs)


def find_lambda(train, val, steps):
    top_lambda = None
    top_accuracy = 0
    songs, labels = prepare_labels(val)
    for model_lambda in np.linspace(0, 1, steps):
        model = MixedModel(model_lambda)
        model.fit(train)
        accuracy = get_accuracy(model, songs, labels)
        print(f"lambda: {model_lambda}, accuracy: {accuracy}")
        if accuracy > top_accuracy:
            top_accuracy = accuracy
            top_lambda = model_lambda

    return top_lambda


def main():
    train, val = train_val_split("Lyrics")
    l = find_lambda(train, val, 11)
    print(l)


if __name__ == "__main__":
    main()
