import os
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import optim
from SongFeaturesDataset import SongFeaturesDataset
from GenreClassifier import GenreClassifier, get_train_val_dataloaders, train, evaluate
from feature_extraction import get_lyrics_from_files, analyze_lyrics, show_correlation_heatmap

torch.manual_seed(42)


def main():
    df_filename = "df_features.csv"

    if not os.path.exists(df_filename):
        df_lyrics = get_lyrics_from_files()
        df_song_features = analyze_lyrics(df_lyrics)
        df_song_features.to_csv(df_filename, index=False)

    df = pd.read_csv(df_filename)

    # Take care of imbalanced dataset
    samples_per_genre = df['Genre'].value_counts().min()
    balanced_df = df.groupby('Genre').apply(lambda x: x.sample(samples_per_genre)).reset_index(drop=True)

    print(balanced_df['Genre'].value_counts())

    # show correlation heatmap of the song lyric features
    show_correlation_heatmap(df)

    dataset = SongFeaturesDataset(balanced_df, "Genre")
    model = GenreClassifier()

    if TRAIN:
        batch_size = 2
        num_epochs = 20
        learning_rate = 0.001
        loss_func = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        train_loader, val_loader = get_train_val_dataloaders(dataset, batch_size, val_ratio=0.2, shuffle=False)

        loss_train, loss_valid = train(model, train_loader, val_loader, loss_func, optimizer, num_epochs)

        torch.save(model.state_dict(), "./song_genre_class_model.pt")

        plot_train_val_loss(loss_train, loss_valid)
    else:
        model.load_state_dict(torch.load("./song_genre_class_model.pt"))
        print("Loading weights complete...")

    df_test_filename = "df_test_features.csv"

    if not os.path.exists(df_test_filename):
        df_lyrics = get_lyrics_from_files("./Test Songs")
        df_song_features = analyze_lyrics(df_lyrics)
        df_song_features.to_csv(df_test_filename, index=False)

    df = pd.read_csv(df_test_filename)
    dataset = SongFeaturesDataset(df, "Genre")

    f1, f1_per_genre = evaluate(model, dataset.features, dataset.labels)
    print(f"\nF1 score of the model: {round(f1.item(), 3)}")
    print(f"F1 score for each class:")

    genres = ["Blues", "Country", "Metal", "Pop", "Rap", "Rock"]
    for genre, score in zip(genres, list(f1_per_genre)):
        print(f"{genre}: {round(score.item(), 3)}")


def find_batch_size(dataset, loss_func, num_epochs, learning_rate):
    """
    Used to find the optimal batch size
    """
    batch_sizes = [1, 2, 4, 8, 16]

    t, v = [], []
    best_train = (0, 1000)
    best_valid = (0, 1000)

    for batch_size in batch_sizes:
        model = GenreClassifier()
        train_loader, val_loader = get_train_val_dataloaders(dataset, batch_size, val_ratio=0.2, shuffle=False)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        loss_train, loss_valid = train(model, train_loader, val_loader, loss_func, optimizer, num_epochs)
        t.append(loss_train[-1])
        v.append(loss_valid[-1])

        if best_train[1] > loss_train[-1]:
            best_train = (batch_size, loss_train[-1])

        if best_valid[1] > loss_valid[-1]:
            best_valid = (batch_size, loss_valid[-1])

    print(f"Best train batch size: {best_train}")
    print(f"Best valid batch size: {best_valid}")


def find_learning_rate(dataset, loss_func, num_epochs, batch_size):
    """
    Used to find the optimal learning rate
    """
    learning_rates = [10, 1, 0.1, 0.01, 0.001, 0.0001]

    t, v = [], []
    best_train = (0, 1000)
    best_valid = (0, 1000)

    for learning_rate in learning_rates:
        model = GenreClassifier()
        train_loader, val_loader = get_train_val_dataloaders(dataset, batch_size, val_ratio=0.2, shuffle=False)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        loss_train, loss_valid = train(model, train_loader, val_loader, loss_func, optimizer, num_epochs)
        t.append(loss_train[-1])
        v.append(loss_valid[-1])

        if best_train[1] > loss_train[-1]:
            best_train = (batch_size, loss_train[-1])

        if best_valid[1] > loss_valid[-1]:
            best_valid = (batch_size, loss_valid[-1])

    print(f"Best train learning rate: {best_train}")
    print(f"Best valid learning rate: {best_valid}")


def plot_train_val_loss(loss_train, loss_valid):
    plt.plot(loss_train, 'r--', label="Training")
    plt.plot(loss_valid, 'b--', label="Validation")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('train_val_loss.png')
    plt.show()


# set this to true if you want to train the model yourself and save it
TRAIN = True


if __name__ == "__main__":
    main()
