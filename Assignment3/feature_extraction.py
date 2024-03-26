import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from feature_utils import (get_number_of_words, get_vocabulary_size, get_average_word_length, get_pos_distribution,
                           get_stop_word_ratio, get_named_entities, get_rhyme_density, get_line_length_variation,
                           get_sentiment_score, download_resources, get_average_line_length)


def get_lyrics_from_files(path="./Training Songs"):
    lyrics = []
    genres = []
    titles = []
    for genre in os.listdir(path):
        for song in tqdm(os.listdir(f"{path}/{genre}"), f"Gathering songs for {genre}: "):
            with open(f"{path}/{genre}/{song}", 'r', encoding='utf-8') as rf:
                lyric = rf.read()
                lyrics.append(lyric)
                titles.append(song)
                genres.append(genre)

    dic_lyrics = {
        "lyrics": lyrics,
        "title": titles,
        "genre": genres,
    }
    return pd.DataFrame(dic_lyrics)


def get_lyrics_from_csv(path="./lyrics.csv", sep=','):
    df_lyrics = pd.read_csv(path, sep=sep)
    return df_lyrics


def analyze_lyrics(df):
    """
    Takes a song lyrics dictionary and returns a DataFrame with extracted features for each song.
    """
    download_resources()
    pos_tags = ["VERB", "NOUN", "PRON", "ADJ", "ADV", "ADP",
                "CONJ", "DET", "NUM", "PRT", "X", "."]

    genres = []
    number_of_words = []
    vocabulary_size = []
    average_line_length = []
    average_word_length = []
    stop_word_ratio = []
    # named_entities = []
    rhyme_density = []
    line_length_variation = []
    sentiment_score = []
    pos_distribution_dic = {tag: [] for tag in pos_tags}

    for i, row in tqdm(df.iterrows(), "Analyzing songs: ", total=len(df)):
        song_lyrics, genre = row["lyrics"], row["genre"]
        genres.append(genre)
        number_of_words.append(get_number_of_words(song_lyrics))
        vocabulary_size.append(get_vocabulary_size(song_lyrics))
        average_line_length.append(get_average_line_length(song_lyrics))
        average_word_length.append(get_average_word_length(song_lyrics))
        stop_word_ratio.append(get_stop_word_ratio(song_lyrics))
        # named_entities.append(get_named_entities(song_lyrics))
        rhyme_density.append(get_rhyme_density(song_lyrics))
        line_length_variation.append(get_line_length_variation(song_lyrics))
        sentiment_score.append(get_sentiment_score(song_lyrics))
        pos_distribution = get_pos_distribution(song_lyrics)
        for key in pos_distribution_dic.keys():
            pos_distribution_dic[key].append(pos_distribution.get(key, 0))

    genre_encodings = {"Blues": 0,
                       "Country": 1,
                       "Metal": 2,
                       "Pop": 3,
                       "Rap": 4,
                       "Rock": 5
                       }
    genres = [genre_encodings[genre] for genre in genres]

    song_data = {
        "Genre": genres,
        "Number of Words": number_of_words,
        "Vocabulary Size": vocabulary_size,
        "Average Line Length": average_line_length,
        "Average Word Length": average_word_length,
        "Stop Word Ratio": stop_word_ratio,
        # "Named Entities": named_entities,
        "Rhyme density": rhyme_density,
        "Line Length Variation": line_length_variation,
        "Sentiment Score": sentiment_score,
    }

    for key, value in pos_distribution_dic.items():
        song_data[key] = value

    df = pd.DataFrame(song_data)

    return df


def show_correlation_heatmap(df):
    corr = df.corr()
    plt.figure(figsize=(24, 10))
    heatmap = sns.heatmap(corr, vmin=-1, vmax=1, cmap='BrBG', annot=True)
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 20}, pad=12)
    plt.xticks(fontsize=14, color='black')
    plt.yticks(fontsize=14, color='black')
    plt.savefig("correlation_heatmap.png")
    plt.show()

