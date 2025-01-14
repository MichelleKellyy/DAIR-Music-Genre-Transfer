import numpy as py
import pandas as pd
import matplotlib.pyplot as plt
import pypianoroll
import os

# Remove unnecessary data columns
df = pd.read_json("./datasets/train.json", lines=True)
df = df.drop(['caption', 'mood', 'mood_prob',
       'key', 'time_signature', 'tempo', 'tempo_word', 'duration',
       'duration_word', 'chord_summary', 'chord_summary_occurence',
       'instrument_summary', 'instrument_numbers_sorted', 'all_chords',
       'all_chords_timestamps', 'test_set'], axis = 1)

df['genre'] = df['genre'].str[0]
df['genre_prob'] = df['genre_prob'].str[0]

# Plot the genre distribution of the MIDI-Caps dataset
def plot_genres(df):
    genre_count = df['genre'].value_counts()

    genre_count.plot(kind = 'bar')
    plt.xlabel('Genre')
    plt.ylabel('Frequency')
    plt.show()

# TODO: Plot genre distribution according to the first genre
# TODO: Plot genre distribution according to the second genre
# TODO: Plot genre distribution according to both genres

# Change the data to be represented by the Million Song Dataset
tagtraum_path = "./datasets/msd_tagtraum_cd2c.cls"
unmatched_genres = pd.read_csv(tagtraum_path, sep = '\t', comment = '#', names = ['trackId', 'genre'])

# Plot genre distribution according to the Million Song Dataset
plot_genres(unmatched_genres)

# TODO: Also print genre occurences for each

# TODO: Plot intrument distribution