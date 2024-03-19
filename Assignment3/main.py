import os
import pandas as pd
from feature_utils import (get_number_of_words, get_vocabulary_size, get_average_word_length, get_pos_distribution,
                           get_stop_word_ratio, get_named_entities, get_rhyme_density, get_line_length_variation,
                           get_sentiment_score)


def get_songs(path="./Test Songs"):
    songs = {}
    for genre in os.listdir(path):
        songs[genre] = []
        for song in os.listdir(f"{path}/{genre}"):
            rf = open(f"{path}/{genre}/{song}", 'r')
            lyric = rf.read()
            songs[genre].append(lyric)

    return songs


def analyze_songs(songs):
    """
    Takes a song lyrics dictionary and returns a DataFrame with extracted features for each song.
    """
    pos_tags = ["VERB", "NOUN", "PRON", "ADJ", "ADV", "ADP",
                "CONJ", "DET", "NUM", "PRT", "X", "."]

    genres = []
    number_of_words = []
    vocabulary_size = []
    average_word_length = []
    stop_word_ratio = []
    # named_entities = []
    rhyme_density = []
    line_length_variation = []
    sentiment_score = []
    pos_distribution_dic = {tag: [] for tag in pos_tags}


    for genre in songs.keys():
        for song_lyrics in songs[genre]:
            genres.append(genre)
            number_of_words.append(get_number_of_words(song_lyrics))
            vocabulary_size.append(get_vocabulary_size(song_lyrics))
            average_word_length.append(get_average_word_length(song_lyrics))
            stop_word_ratio.append(get_stop_word_ratio(song_lyrics))
            # named_entities.append(get_named_entities(song_lyrics))
            rhyme_density.append(get_rhyme_density(song_lyrics))
            line_length_variation.append(get_line_length_variation(song_lyrics))
            sentiment_score.append(get_sentiment_score(song_lyrics))
            pos_distribution = get_pos_distribution(song_lyrics)
            for key in pos_distribution_dic.keys():
                pos_distribution_dic[key].append(pos_distribution.get(key, 0))




    song_data = {
        "Number of Words": number_of_words,
        "Vocabulary Size": vocabulary_size,
        "Average Word Length": average_word_length,
        "Stop Word Ratio": stop_word_ratio,
        # "Named Entities": named_entities,
        "Rhyme density": rhyme_density,
        "Line Length Variation": line_length_variation,
        "Sentiment Score": sentiment_score,
        "Genre": genres
    }

    for key, value in pos_distribution_dic.items():
        song_data[key] = value

    df = pd.DataFrame(song_data)

    return df


song_lyrics = get_songs()
song_features_df = analyze_songs(song_lyrics)

print(song_features_df)
