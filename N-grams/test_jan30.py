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

#added more features to attempt to improve accuracy --> made a counter list instead of only looking at pitch
def extract_ngram(midi_file, n):
    try:
        score = music21.converter.parse(midi_file) #parsing the midi file
        
        notes = [] 
        durations = []
        chords = []
        onset_times = []
        prev_onset = 0 #starting time to rest when looking at notes/ rests in chords
        
        for element in score.flat.notesAndRests:
            onset_time = element.offset
            onset_diff = onset_time - prev_onset # finding the notes and rests durations
            prev_onset = onset_time
            
            if isinstance(element, music21.note.Note): #if note is present
                notes.append(str(element.pitch))
                durations.append(str(element.quarterLength)) #adding quarter length to durations
                onset_times.append(str(onset_diff))
            elif isinstance(element, music21.note.Rest): # if rest is present
                durations.append(str(element.quarterLength))
                onset_times.append(str(onset_diff))
            elif isinstance(element, music21.chord.Chord): #what chord is being played
                chord_symbol = element.pitchedCommonName
                chords.append(chord_symbol)
                durations.append(str(element.quarterLength))
                onset_times.append(str(onset_diff))
        
        note_ngrams = Counter(ngrams(notes, n)) #ngram for notes
        duration_ngrams = Counter(ngrams(durations, n)) #ngram for durations
        onset_ngrams = Counter(ngrams(onset_times, n)) #ngram for onset = time that note starts to be played 
        chord_ngrams = Counter(ngrams(chords, n)) #ngram for chord patterns
        
        return {
            'notes': note_ngrams,
            'durations': duration_ngrams,
            'onsets': onset_ngrams,
            'chords': chord_ngrams
        }
    except Exception as e:
        print(f"Error processing {midi_file}: {str(e)}")
        return {'notes': Counter(), 'durations': Counter(), 'onsets': Counter(), 'chords': Counter()} #returning counter for each feature to compare later

    
def is_valid_midi(file_path): #validating if midi file is not corrupt --> not processing it if is is not readable
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
        print(f"Processing genre: {genre}") #prinitng whihc file is being processed
        genre_ngrams = {'notes': Counter(), 'durations': Counter(), 'onsets': Counter(), 'chords': Counter()}
        valid_files = 0 #accounting for files that are being processed

        for file in files:
            if is_valid_midi(file):
                file_ngrams = extract_ngram(file, n)
                for feature in genre_ngrams:
                    genre_ngrams[feature] += file_ngrams[feature]
                valid_files += 1 #adding to proceesed file counter
                print(f"  Processed {file}: {sum(len(v) for v in file_ngrams.values())} n-grams")
            else:
                print(f"  Skipped invalid file: {file}")

        if valid_files > 0:
            genre_database[genre] = {
                feature: {ngram: count / valid_files for ngram, count in genre_ngrams[feature].items()}
                for feature in genre_ngrams
            }
        else:
            print(f"  No valid files for genre {genre}") #if there are no valid files in the genre it will skip it 

    return genre_database

def similar_genre(input_ngrams, genre_database, weights={'notes': 0.4, 'durations': 0.2, 'onsets': 0.2, 'chords': 0.2}):
    similar = {}
    
    for genre, genre_ngrams in genre_database.items():
        total_score = 0
        
        for feature_type, weight in weights.items():
            input_feature_ngrams = input_ngrams.get(feature_type, Counter())
            genre_feature_ngrams = genre_ngrams.get(feature_type, Counter())
            
            matching_ngrams = sum(min(input_feature_ngrams.get(ngram, 0), genre_feature_ngrams.get(ngram, 0)) for ngram in input_feature_ngrams)
            total_input_ngrams = sum(input_feature_ngrams.values())
            
            feature_similarity = (matching_ngrams / total_input_ngrams) if total_input_ngrams > 0 else 0
            total_score += feature_similarity * weight
        
        similar[genre] = total_score
    
    return similar


## Example
n = 3  # n-gram size increase to 3 since the dataset is larger --> size of n-gram is proportional to the size of the dataset

# Base directory
base_dir = "/Users/brooklynarseneau/Documents/QMIND/symbolic-music-genre-classification/lmd_matched"

# Read the CSV file
df = pd.read_csv('/Users/brooklynarseneau/Documents/QMIND/train.csv', header=None, names=['File', 'Genre'])

# Create the genre_files dictionary
genre_files = {}

# Group the DataFrame by genre 
for genre, group in df.groupby('Genre'):
    # Select the first 10 files for each genre
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
            
            # Append the full file path to the each genre list
            genre_files[genre].append(file_path)
        #else:
            #print(f"File not found: {file_path}")

# Create genre database
genre_database = genre_database(genre_files, n)

# Input symbolic music file
#jazz= "/Users/brooklynarseneau/Downloads/jazz_sample2.mid" (matched correctly)
#rock="/Users/brooklynarseneau/Downloads/rock_sample1.mid" (matched correctly)
#pop="/Users/brooklynarseneau/Downloads/pop_sample2.mid" (matched correctly)
#rnb= "/Users/brooklynarseneau/Downloads/rnb_sample2.mid" (matched correctly)

input_file = "/Users/brooklynarseneau/Downloads/rnb_sample2.mid"

input_ngrams = extract_ngram(input_file, n)

# Compare similarity
similarities = similar_genre(input_ngrams, genre_database)

# Print results
print("Genre similarities:")
for genre, similarity in similarities.items():
    print(f"{genre}: {similarity:.4f}")

print(f"\nMost similar genre: {max(similarities, key=similarities.get)}")
