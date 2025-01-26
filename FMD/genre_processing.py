import os
import json
import numpy as np
import pretty_midi
import matplotlib.pyplot as plt
from collections import defaultdict

# Gaussian function
def gaussian(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Compute temporal and pitch distributions for a single MIDI file
def compute_distributions(midi_file_path, time_steps=1000, sigma=1):
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file_path)
    except Exception as e:
        print(f"Error processing {midi_file_path}: {e}")
        return None, None

    # Temporal Gaussian distribution
    time_grid = np.linspace(0, midi_data.get_end_time(), time_steps)
    time_gaussian = np.zeros_like(time_grid)

    # Pitch Gaussian distribution
    pitch_grid = np.arange(0, 128)  # MIDI pitch range
    pitch_gaussian = np.zeros_like(pitch_grid, dtype=float)

    # Process notes in the MIDI file
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            # Temporal Gaussian
            center_time = (note.start + note.end) / 2
            time_gaussian += gaussian(time_grid, center_time, sigma)

            # Pitch Gaussian
            pitch_gaussian += gaussian(pitch_grid, note.pitch, sigma)

    return time_gaussian, pitch_gaussian

# Process all files for one genre and compute average distributions
def process_genre_files(file_list, base_folder, time_steps=1000, sigma=1):
    all_time_gaussians = []
    all_pitch_gaussians = []

    for entry in file_list:
        file_path = os.path.join(base_folder, entry["location"])
        time_gaussian, pitch_gaussian = compute_distributions(file_path, time_steps, sigma)
        if time_gaussian is not None and pitch_gaussian is not None:
            all_time_gaussians.append(time_gaussian)
            all_pitch_gaussians.append(pitch_gaussian)

    if all_time_gaussians and all_pitch_gaussians:
        avg_time_gaussian = np.mean(all_time_gaussians, axis=0)
        avg_pitch_gaussian = np.mean(all_pitch_gaussians, axis=0)
        return avg_time_gaussian, avg_pitch_gaussian
    else:
        return None, None

# Visualization function
def plot_distributions(time_grid, avg_time_gaussian, pitch_grid, avg_pitch_gaussian, genre):
    """
    Plot average temporal and pitch distributions for a genre.
    """
    # Temporal distribution
    plt.figure(figsize=(10, 4))
    plt.plot(time_grid, avg_time_gaussian, label=f'{genre} Temporal Distribution')
    plt.xlabel('Time (s)')
    plt.ylabel('Density')
    plt.title(f'Average Temporal Distribution for {genre}')
    plt.legend()
    plt.show()

    # Pitch distribution
    plt.figure(figsize=(10, 4))
    plt.plot(pitch_grid, avg_pitch_gaussian, label=f'{genre} Pitch Distribution')
    plt.xlabel('Pitch')
    plt.ylabel('Density')
    plt.title(f'Average Pitch Distribution for {genre}')
    plt.legend()
    plt.show()

# Main function for one genre
if __name__ == "__main__":
    
    # Path to the folder containing all MIDI files
    midi_folder_path = "F:/dataset"  # Replace with the actual path

    genres = ["electronic", "pop", "rock", "classical"]
    for genre in genres:
        # Path to the JSON mapping file for one genre
        genre_mapping_file = "data/" + genre + "_first_100.json"

        # Load the file list for the selected genre
        try:
            with open(genre_mapping_file, "r") as file:
                file_list = json.load(file)  # List of dictionaries for the genre
        except FileNotFoundError:
            print(f"File not found: {genre_mapping_file}")
            exit()

        # Process files and compute average distributions
        print(f"Processing genre: " + genre + " with " + {len(file_list)} + " files.")
        avg_time_gaussian, avg_pitch_gaussian = process_genre_files(file_list, midi_folder_path)

        if avg_time_gaussian is not None and avg_pitch_gaussian is not None:
            # Save the results
            time_grid = np.linspace(0, 60, 1000)  # Assuming 60 seconds max duration
            pitch_grid = np.arange(0, 128)

            plot_distributions(time_grid, avg_time_gaussian, pitch_grid, avg_pitch_gaussian, genre)
            # Save distributions for future use
            np.save("genre_gaussian_data\\" + genre + "_avg_time.npy", avg_time_gaussian)
            np.save("genre_gaussian_data\\" + genre + "_avg_pitch.npy", avg_pitch_gaussian)

            print("Average distributions saved for genre: electronic.")
        else:
            print("No valid data processed for genre: electronic.")
