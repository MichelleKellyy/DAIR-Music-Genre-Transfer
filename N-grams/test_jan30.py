import os
import pandas as pd
import music21
import pretty_midi
from music21 import converter, note, chord
from pretty_midi import PrettyMIDI
from nltk import ngrams
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_fscore_support

    ## don't use velocity!
def extract_ngram(midi_file, n):
    try:
        score = music21.converter.parse(midi_file)
        notes = [str(n.pitch) for n in score.flat.notesAndRests if isinstance(n, music21.note.Note)]
        return Counter(ngrams(notes, n))
    except Exception as e:
        print(f"Error processing {midi_file}: {str(e)}")
        return Counter()
    
def is_valid_midi(file_path):
    try:
        midi = music21.midi.MidiFile()
        midi.open(file_path)
        midi.read()
        midi.close()
        return True
    except Exception as e:
        print(f"Invalid MIDI file {file_path}: {str(e)}")
        return False

def genre_database(genre_files, n):
    genre_database = {}
    for genre, files in genre_files.items():
        print(f"Processing genre: {genre}")
        genre_ngrams = Counter()
        valid_files = 0
        for file in files:
            if is_valid_midi(file):
                file_ngrams = extract_ngram(file, n)
                genre_ngrams += file_ngrams
                valid_files += 1
                print(f"  Processed {file}: {len(file_ngrams)} n-grams")
            else:
                print(f"  Skipped invalid file: {file}")
        if valid_files > 0:
            genre_database[genre] = {ngram: count/valid_files for ngram, count in genre_ngrams.items()}
        else:
            print(f"  No valid files for genre {genre}")
    return genre_database

def similar_genre(input_ngrams,genre_database):
    similar={}
    for genre, genre_ngrams in genre_database.items():
        matching_ngrams=0
        total_input_ngrams=sum(input_ngrams.values())
        for ngram, input_freq in input_ngrams.items():
            if ngram in genre_ngrams:
                matching_ngrams +=min(input_freq,genre_ngrams[ngram]) #counting the matching ngrams using min freq
        similar[genre]=matching_ngrams/total_input_ngrams if total_input_ngrams > 0 else 0
    return similar

## Example
n = 2  # n-gram size

# Base directory
base_dir = "/Users/brooklynarseneau/Documents/QMIND/symbolic-music-genre-classification/lmd_matched"

# Read the CSV file
df = pd.read_csv('/Users/brooklynarseneau/Documents/QMIND/train.csv', header=None, names=['File', 'Genre'])

# Create the genre_files dictionary
genre_files = {}

# Group the DataFrame by genre
for genre, group in df.groupby('Genre'):
    # Select the first 10 files for the genre
    group = group.head(10)
    
    for _, row in group.iterrows():
        file_name = row['File']
        file_name_fixed = file_name.replace("\\", "/")
        
        # Construct the full file path
        file_path = os.path.join(base_dir, file_name_fixed)
        
        # Check if the file exists
        if os.path.exists(file_path):
            # If the genre is not in the dictionary, add it with an empty list
            if genre not in genre_files:
                genre_files[genre] = []
            
            # Append the full file path to the appropriate genre list
            genre_files[genre].append(file_path)
        #else:
            #print(f"File not found: {file_path}")

# Print the genre_files dictionary
# #print(genre_files)


# Create genre database
genre_database = genre_database(genre_files, n)

# Input symbolic music file
input_file = "/Users/brooklynarseneau/Downloads/jazz_sample2.mid"
input_ngrams = extract_ngram(input_file, n)

# Compare similarity
similarities = similar_genre(input_ngrams, genre_database)

# Print results
print("Genre similarities:")
for genre, similarity in similarities.items():
    print(f"{genre}: {similarity:.4f}")

print(f"\nMost similar genre: {max(similarities, key=similarities.get)}")





