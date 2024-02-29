import os

import pandas as pd

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from mlxtend.evaluate import mcnemar_table, mcnemar

from UnigramModel import UnigramModel
from BigramModel import BigramModel
from MixedModel import MixedModel
from utils import combine_all_text_files, get_data, combined_text_file, add_space_between_capital_letters

# prepare data
combine_all_text_files()
add_space_between_capital_letters('test.tsv', 'test2.tsv')


df = pd.read_csv('test2.tsv', sep='\t', header=0)
songs = df['Text']
labels = df['Genre']

# define models
model_lambda = 0

unigram_model = UnigramModel()
bigram_model = BigramModel()
mixed_model = MixedModel(model_lambda)

models = {
            "unigram": unigram_model,
            "bigram": bigram_model,
            "mixed": mixed_model
          }

model_predictions = {}

# create label encoder
label_encoder = LabelEncoder()
true_labels = label_encoder.fit_transform(labels)


for model_name, model in models.items():
    for genre in os.listdir("Lyrics"):
        data = get_data(f"Lyrics/{genre}/{combined_text_file}")
        model.train_for_genre(genre, data)
    predicted_labels = model.predict(songs)
    model_predictions[model_name] = label_encoder.transform(predicted_labels)

for model_name, predicted_labels in model_predictions.items():
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels,
                                                               average='macro', zero_division=0.0)
    print(f"{model_name}: precision: {precision:.4}, recall: {recall:.4}, f1: {f1:.4}")

print("\nConfusion matrices:")
for model_name, predicted_labels in model_predictions.items():
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    print(f"{model_name}:")
    print(conf_matrix)


# significant testing
print("\nMcNemar's test for unigram and bigram models")
mcnemar_tb = mcnemar_table(y_target=true_labels,
                           y_model1=model_predictions["unigram"],
                           y_model2=model_predictions["bigram"])

print("Contingency table:")
print(mcnemar_tb)

chi2, p = mcnemar(ary=mcnemar_tb, corrected=True)
print('chi-squared:', chi2)
print('p-value:', p)

