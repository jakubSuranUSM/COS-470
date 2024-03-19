import os
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from Dataset import SongFeaturesDataset
from GenreClassifier import GenreClassifier
from feature_extraction import get_songs, analyze_songs, show_correlation_heatmap

df_filename = "df_features.csv"

if not os.path.exists(df_filename):
    song_lyrics = get_songs()
    song_features_df = analyze_songs(song_lyrics)
    song_features_df.to_csv(df_filename, index=False)

df = pd.read_csv(df_filename)

show_correlation_heatmap(df)

dataset = SongFeaturesDataset(df, "Genre")
model = GenreClassifier()

batch_size = 4
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# num_of_batches = sum([1 for X, label in train_dataloader])
# for X, label in train_dataloader:
#     out = model(X)
#     print(out)
#     break

num_epochs = 20
learning_rate = 0.001
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
loss_func = torch.nn.CrossEntropyLoss()
loss_epoch = {}
loss_values = []

for epoch in tqdm(range(num_epochs)):
    for X, y in train_dataloader:
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        pred = model(X)
        loss = loss_func(pred, y)
        loss_values.append(loss.item())
        loss.backward()
        optimizer.step()
    loss_epoch[epoch] = sum(loss_values) / len(loss_values)


plt.plot(loss_epoch.keys(), loss_epoch.values(), 'r--')
plt.legend(['Training Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
