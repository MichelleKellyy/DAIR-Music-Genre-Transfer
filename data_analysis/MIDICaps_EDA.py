import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil

pd.set_option('display.max_rows', None)

# Remove unnecessary data columns
df = pd.read_json("./full_datasets/train.json", lines=True)
df = df.drop(['caption', 'genre_prob', 'mood', 'mood_prob', 'tempo_word',
    'duration_word', 'chord_summary_occurence', 'instrument_numbers_sorted',
    'all_chords', 'all_chords_timestamps', 'test_set'], axis = 1)

# Drop entries with empty or null values
df = df.dropna()
df = df[df['instrument_summary'].apply(lambda x: len(x) > 0)]
df = df[df['chord_summary'].apply(lambda x: len(x) > 0)]

# Drop genres under 10k
df['genre'] = df['genre'].str[0]
df = df[df['genre'].isin(['electronic', 'pop', 'classical', 'rock'])]

# Drop time signatures under 10k
df = df[df['time_signature'] == '4/4']

# Drop songs under 10 seconds
df = df[df['duration'] >= 10]

# Combine instruments into fewer categories
df['instrument_summary'] = df['instrument_summary'].apply(lambda x:
    ['Piano' if item in ['Electric Piano', 'Honky-tonk Piano', 'Harpsichord',
    'Clavinet', 'Celesta'] else item for item in x])
df['instrument_summary'] = df['instrument_summary'].apply(lambda x:
    ['Drums' if item in ['Orchestra Hit', 'Synth Drum', 'Taiko Drum',
    'Reverse Cymbal', 'Melodic Tom', 'Timpani', 'Woodblock']
    else item for item in x])
df['instrument_summary'] = df['instrument_summary'].apply(lambda x:
    ['Strings' if item in ['String Ensemble', 'Synth Strings',
    'Tremolo Strings', 'Viola', 'Violin', 'Cello', 'Fiddle',
    'Orchestral Harp', 'Pizzicato Strings', 'Koto', 'Sitar', 'Shamisen']
    else item for item in x])
df['instrument_summary'] = df['instrument_summary'].apply(lambda x:
    ['Acoustic Guitar' if item in ['Banjo', 'Dulcimer']
    else item for item in x])
df['instrument_summary'] = df['instrument_summary'].apply(lambda x:
    ['Bass' if item in ['Electric Bass', 'Acoustic Bass', 'Fretless Bass',
    'Synth Bass', 'Slap Bass', 'Contrabass']
    else item for item in x])
df['instrument_summary'] = df['instrument_summary'].apply(lambda x:
    ['Electric Guitar' if item in ['Guitar Harmonics', 'Guitar Fret Noise',
    'Distortion Guitar', 'Overdriven Guitar', 'Clean Electric Guitar']
    else item for item in x])
df['instrument_summary'] = df['instrument_summary'].apply(lambda x:
    ['Choir' if item in ['Choir Aahs', 'Synth Voice', 'Voice Oohs']
    else item for item in x])
df['instrument_summary'] = df['instrument_summary'].apply(lambda x:
    ['Woodwinds' if item in ['Flute', 'Pan Flute', 'Shakuhachi', 'Ocarina',
    'Recorder', 'Piccolo', 'Whistle', 'Clarinet', 'Bassoon', 'Oboe',
    'Bagpipe'] else item for item in x])
df['instrument_summary'] = df['instrument_summary'].apply(lambda x:
    ['Brass' if item in ['Brass Section', 'Synth Brass', 'Brass Lead',
    'Trumpet', 'Muted Trumpet', 'Trombone', 'Alto Saxophone',
    'Tenor Saxophone', 'Soprano Saxophone', 'Baritone Saxophone',
    'French Horn', 'Tuba', 'English Horn']
    else item for item in x])
df['instrument_summary'] = df['instrument_summary'].apply(lambda x:
    ['Synths' if item in ['Synth Pad', 'Synth Effects', 'Charang Lead',
    'Chiffer Lead', 'Synth Lead']
    else item for item in x])
df['instrument_summary'] = df['instrument_summary'].apply(lambda x:
    ['Organ' if item in ['Rock Organ', 'Hammond Organ', 'Percussive Organ',
    'Church Organ', 'Reed Organ', 'Accordion', 'Tango Accordion', 'Harmonica',
    'Calliope Lead'] else item for item in x])
df['instrument_summary'] = df['instrument_summary'].apply(lambda x:
    ['Xylophone' if item in ['Glockenspiel', 'Vibraphone', 'Marimba',
    'Steel Drums', 'Tubular Bells', 'Tinkle Bell', 'Agogo', 'Music box',
    'Kalimba'] else item for item in x])
df['instrument_summary'] = df['instrument_summary'].apply(lambda x:
    ['Other' if item in ['Seashore', 'Applause', 'Gunshot', 'Telephone Ring',
    'Bird Tweet', 'Breath Noise', 'Helicopter', 'Bottle Blow', 'Shana']
    else item for item in x])

# Drop instruments under 10k
df = df[df['instrument_summary'].apply(lambda x: 'Other' not in x)]

# Method to plot distributions in the MIDI-Caps dataset
def plot_distributions(category, title, label):
    count = category.value_counts()
    print(count)

    count.plot(kind = 'bar')
    plt.title(title)
    plt.xlabel(label)
    plt.ylabel('Frequency')
    plt.show()

# Plot genre distribution
plot_distributions(df['genre'], "MIDI-Caps Genre Distribution", "Genre")

# Plot instrument distribution
df_instruments = df.explode('instrument_summary')
plot_distributions(df_instruments['instrument_summary'],
    "MIDI-Caps Instrument Distribution", "Instruments")

# Create new json file based on cleaned dataframe
df.to_json('dataset.json', orient='records', lines=True)

# Create new dataset based on remaining filepaths
full_datasets_dir = 'full_datasets'
dataset_dir = 'dataset'
for index, row in df.iterrows():
    file_path = os.path.join(full_datasets_dir, row['location'])
    if os.path.isfile(file_path):
        file_name = os.path.basename(file_path)
        destination_path = os.path.join(dataset_dir, file_name)
        shutil.copy(file_path, destination_path)