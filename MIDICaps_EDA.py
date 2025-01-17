import numpy as py
import pandas as pd
import matplotlib.pyplot as plt
import pypianoroll
import os

# Remove unnecessary data columns
df = pd.read_json("./datasets/train.json", lines=True)
df = df.drop(['caption', 'genre_prob', 'mood', 'mood_prob', 'key',
    'time_signature', 'tempo', 'tempo_word', 'duration', 'duration_word',
    'chord_summary', 'chord_summary_occurence', 'instrument_numbers_sorted',
    'all_chords', 'all_chords_timestamps', 'test_set'], axis = 1)

# Drop genres under 10k
df = df[df['genre'].str[0].isin(['electronic', 'pop', 'classical', 'rock'])]

# Method to plot the genre distribution of the MIDI-Caps dataset
def plot_genres(df, title):
    genre_count = df['genre'].value_counts()
    print(genre_count)

    genre_count.plot(kind = 'bar')
    plt.title(title)
    plt.xlabel('Genre')
    plt.ylabel('Frequency')
    plt.show()

# Plot genre distribution according to the first predicted genre
df_first = df.copy()
df_first['genre'] = df['genre'].str[0]
plot_genres(df_first,
    "MIDI-Caps Genre Distribution With First Predicted Genre")

# Method to plot the instrument distribution of the MIDI-Caps dataset
def plot_instruments(df, title):
    instrument_count = df['instrument_summary'].value_counts()
    print(instrument_count)

    instrument_count.plot(kind = 'bar')
    plt.title(title)
    plt.xlabel('Instrument')
    plt.ylabel('Frequency')
    plt.show()

# Plot instrument distribution
pd.set_option('display.max_rows', None)
df_instruments = df.explode('instrument_summary')
plot_instruments(df_instruments,
    "MIDI-Caps Instrument Distribution")

# Combine instruments into fewer categories
df_instruments_condensed = df_instruments.copy()
df_instruments_condensed['instrument_summary'].replace(
    ['Electric Piano', 'Honky-tonk Piano',
    'Harpsichord', 'Clavinet', 'Celesta'],
    'Piano', inplace=True)
df_instruments_condensed['instrument_summary'].replace(
    ['Drums', 'Orchestra Hit', 'Synth Drum',
    'Taiko Drum', 'Reverse Cymbal'],
    'Unpitched Percussion', inplace=True)
df_instruments_condensed['instrument_summary'].replace(
    ['String Ensemble', 'Synth Strings', 'Tremolo Strings',
    'Viola', 'Violin', 'Cello', 'Fiddle'],
    'Strings', inplace=True)
df_instruments_condensed['instrument_summary'].replace(
    ['Banjo', 'Dulcimer'],
    'Acoustic Guitar', inplace=True)
df_instruments_condensed['instrument_summary'].replace(
    ['Electric Bass', 'Acoustic Bass', 'Fretless Bass',
    'Synth Bass', 'Slap Bass', 'Contrabass'],
    'Bass', inplace=True)
df_instruments_condensed['instrument_summary'].replace(
    ['Electric Guitar', 'Guitar Harmonics', 'Guitar Fret Noise'],
    'Clean Electric Guitar', inplace=True)
df_instruments_condensed['instrument_summary'].replace(
    ['Overdriven Guitar'],
    'Distortion Guitar', inplace=True)
df_instruments_condensed['instrument_summary'].replace(
    ['Choir Aahs', 'Synth Voice', 'Voice Oohs'],
    'Choir', inplace=True)
df_instruments_condensed['instrument_summary'].replace(
    ['Flute', 'Pan Flute', 'Shakuhachi', 'Clarinet', 'Bassoon',
    'Ocarina', 'Oboe', 'Recorder', 'Piccolo', 'Whistle'],
    'Woodwinds', inplace=True)
df_instruments_condensed['instrument_summary'].replace(
    ['Brass Section', 'Synth Brass', 'Brass Lead', 'Trumpet', 'Muted Trumpet',
    'Trombone', 'Alto Saxophone', 'Tenor Saxophone', 'Soprano Saxophone',
    'Baritone Saxophone', 'French Horn', 'Tuba', 'English Horn'],
    'Brass', inplace=True)
df_instruments_condensed['instrument_summary'].replace(
    ['Synth Effects', 'Charang Lead', 'Chiffer Lead'],
    'Synth Lead', inplace=True)
df_instruments_condensed['instrument_summary'].replace(
    ['Rock Organ', 'Hammond Organ', 'Percussive Organ',
    'Church Organ', 'Reed Organ'],
    'Organ', inplace=True)
df_instruments_condensed['instrument_summary'].replace(
    ['Orchestral Harp', 'Pizzicato Strings', 'Koto', 'Sitar', 'Shamisen'],
    'Plucked Strings', inplace=True)
df_instruments_condensed['instrument_summary'].replace(
    ['Accordion', 'Tango Accordion', 'Harmonica', 'Calliope Lead'],
    'Many Woodwinds', inplace=True)
df_instruments_condensed['instrument_summary'].replace(
    ['Glockenspiel', 'Vibraphone', 'Marimba', 'Steel Drums'],
    'Xylophone', inplace=True)
df_instruments_condensed['instrument_summary'].replace(
    ['Tubular Bells', 'Tinkle Bell', 'Agogo'],
    'Bells', inplace=True)
df_instruments_condensed['instrument_summary'].replace(
    ['Music box'],
    'Kalimba', inplace=True)
df_instruments_condensed['instrument_summary'].replace(
    ['Melodic Tom', 'Timpani', 'Woodblock'],
    'Pitched Percussion', inplace=True)
df_instruments_condensed['instrument_summary'].replace(
    ['Seashore', 'Applause', 'Gunshot', 'Telephone Ring',
    'Bird Tweet', 'Breath Noise', 'Helicopter'],
    'Sound Effects', inplace=True)
df_instruments_condensed['instrument_summary'].replace(
    ['Shana', 'Bottle Blow', 'Bagpipe'],
    'Other', inplace=True)
plot_instruments(df_instruments_condensed,
    "MIDI-Caps Condensed Instrument Distribution")