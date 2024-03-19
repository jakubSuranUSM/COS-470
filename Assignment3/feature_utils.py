import nltk
import spacy
import pronouncing

from statistics import stdev
from nltk.tag import pos_tag
from nltk.corpus import stopwords
import nltk.sentiment.vader as vader
from collections import Counter

nltk.download('universal_tagset')
nltk.download('vader_lexicon')
stop_words_set = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm")


def extract_words(lyrics):
    lyrics = lyrics.lower()
    lyrics = "".join(char for char in lyrics if char.isalnum() or char.isspace())
    words = lyrics.split()
    return words


def get_number_of_words(lyrics):
    """
    Takes song lyrics as a string and returns number of  words.
    """
    return len(extract_words(lyrics))


def get_vocabulary_size(lyrics):
    """
    Takes song lyrics as a string and returns the vocabulary size (number of unique words).
    """
    words = extract_words(lyrics)
    vocabulary_size = len(set(words))

    return vocabulary_size


def get_average_word_length(lyrics):
    """
    Takes song lyrics as a string and returns the average word length.
    """
    words = extract_words(lyrics)

    if not words:
        return 0

    total_length = sum(len(word) for word in words)
    word_count = len(words)

    average_length = total_length / word_count

    return average_length


def get_pos_distribution(lyrics):
    """
    Takes song lyrics as a string and returns the part-of-speech distribution.
    """
    words = extract_words(lyrics)
    pos_tags = pos_tag(words, tagset="universal")

    tags = [value for key, value in pos_tags]
    pos_distribution = Counter(tags)

    return pos_distribution


def get_stop_word_ratio(lyrics):
    """
    Takes song lyrics as a string and returns the stop word ratio.
    """
    words = extract_words(lyrics)

    total_words = len(words)
    stop_words_count = 0
    for word in words:
        if word in stop_words_set:
            stop_words_count += 1

    if total_words == 0:
        return 0

    stop_word_ratio = stop_words_count / total_words

    return stop_word_ratio


def get_named_entities(lyrics):
    """
    Takes song lyrics as a string and returns a dictionary containing named entity information.
    """
    lyrics = lyrics.lower()
    doc = nlp(lyrics)

    named_entities = [ent.label_ for ent in doc.ents]

    return Counter(named_entities)


# TODO: could use some upgrades
def get_rhyme_density(lyrics):
    """
    Takes song lyrics as a string and returns the rhyme density, focusing on last words of lines.
    """
    lyrics = lyrics.lower()
    lyrics = "".join(char for char in lyrics if char.isalnum() or char.isspace())
    lines = lyrics.splitlines()

    total_rhyme_pairs = 0
    total_lines = len(lines)

    if total_lines == 0:
        return 0

    for i in range(len(lines) - 1):
        current_word = lines[i].split()[-1]
        next_word = lines[i + 1].split()[-1]
        rhymes = pronouncing.rhymes(current_word)

        if next_word in rhymes:
            total_rhyme_pairs += 1

    rhyme_density = total_rhyme_pairs / total_lines

    return rhyme_density


def get_line_length_variation(lyrics):
    """
    Takes song lyrics as a string and returns the line length variation.
    Uses standard deviation of line lengths (measures variation).
    """
    lyrics = lyrics.splitlines()
    line_lengths = []

    for line in lyrics:
        line_lengths.append(len(line.split()))

    if len(line_lengths) == 0:
        return 0

    return stdev(line_lengths)


def get_sentiment_score(lyrics):
    """
    Takes song lyrics as a string and estimates the overall sentiment
    score using NLTK Vader sentiment analyzer.
    """
    lyrics = lyrics.lower()

    sentiment_analyzer = vader.SentimentIntensityAnalyzer()
    sentiment_analysis = sentiment_analyzer.polarity_scores(lyrics)

    return sentiment_analysis['compound']
