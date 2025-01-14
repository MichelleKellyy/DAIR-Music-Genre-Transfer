import numpy as py
import pandas as pd
import matplotlib.pyplot as plt
import pypianoroll
import os

# Remove unnecessary data columns
df = pd.read_json("./datasets/train.json", lines=True)
df = df.drop(['caption', 'genre_prob', 'mood', 'mood_prob',
       'key', 'time_signature', 'tempo', 'tempo_word', 'duration',
       'duration_word', 'chord_summary', 'chord_summary_occurence',
       'instrument_numbers_sorted', 'all_chords',
       'all_chords_timestamps', 'test_set'], axis = 1)

# Plot the genre distribution of the MIDI-Caps dataset
def plot_genres(df, title):
    genre_count = df['genre'].value_counts()
    print(genre_count)

    genre_count.plot(kind = 'bar')
    plt.title(title)
    plt.xlabel('Genre')
    plt.ylabel('Frequency')
    plt.show()

# Plot genre distribution according to the first genre
df_first = df.copy()
df_first['genre'] = df['genre'].str[0]
plot_genres(df_first, "Genre Distribution With First Predicted Genre")

# Plot genre distribution according to the second genre
df_second = df.copy()
df_second['genre'] = df['genre'].str[1]
plot_genres(df_second, "Genre Distribution With Second Predicted Genre")

# Plot genre distribution according to both genres
df_all = df.copy()
df_all = pd.concat([pd.DataFrame(df_first), pd.DataFrame(df_second)])
plot_genres(df_all, "Genre Distribution With All Predicted Genres")

# Plot the instrument distribution of the MIDI-Caps dataset
def plot_instruments(df, title):
    instrument_count = df['instrument_summary'].value_counts()
    print(instrument_count)

    instrument_count.plot(kind = 'bar')
    plt.title(title)
    plt.xlabel('Instrument')
    plt.ylabel('Frequency')
    plt.show()

# Plot intrument distribution
pd.set_option('display.max_rows', None)
df_instruments = df.explode('instrument_summary')
plot_instruments(df_instruments, "Instrument Distribution of MIDI-Caps Dataset")

# Change the data to be represented by the Million Song Dataset
# tagtraum_path = "./datasets/msd_tagtraum_cd2c.cls"
# unmatched_genres = pd.read_csv(tagtraum_path, sep = '\t', comment = '#', names = ['trackId', 'genre'])

# Plot genre distribution of the Million Song Dataset
# plot_genres(unmatched_genres, "Genre Distribution Of Million Song Dataset")