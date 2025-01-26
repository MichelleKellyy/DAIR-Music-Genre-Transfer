import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
import json
from collections import defaultdict

# Gaussian function
def gaussian(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Compute temporal and pitch distributions for a single MIDI file
def compute_distributions(midi_file_path, time_steps=1000, sigma=2):
    """
    Compute temporal and pitch distributions for a MIDI file.
    """
    midi_data = pretty_midi.PrettyMIDI(midi_file_path)

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

    return time_grid, time_gaussian, pitch_grid, pitch_gaussian

# Generate average distributions for each genre
def generate_average_distributions(genre_files, time_steps=1000, sigma=2):
    """
    Generate average temporal and pitch distributions for a genre.
    """
    all_time_gaussians = []
    all_pitch_gaussians = []

    for file in genre_files:
        time_grid, time_gaussian, pitch_grid, pitch_gaussian = compute_distributions(file, time_steps, sigma)
        all_time_gaussians.append(time_gaussian)
        all_pitch_gaussians.append(pitch_gaussian)

    # Average distributions
    avg_time_gaussian = np.mean(all_time_gaussians, axis=0)
    avg_pitch_gaussian = np.mean(all_pitch_gaussians, axis=0)

    return time_grid, avg_time_gaussian, pitch_grid, avg_pitch_gaussian

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

# Main process
if __name__ == "__main__":
    # Path to your list of genres and their files (replace with your actual files)
    genre_file_mapping_path = "genre_files.json"  # JSON file with genres mapped to lists of files

    # Load genre files
    with open(genre_file_mapping_path, "r") as file:
        genre_files = json.load(file)  # { "genre1": ["file1.mid", "file2.mid"], "genre2": [...] }

    # Process each genre
    for genre, files in genre_files.items():
        print(f"Processing genre: {genre} with {len(files)} files.")
        time_grid, avg_time_gaussian, pitch_grid, avg_pitch_gaussian = generate_average_distributions(files)

        # Plot and save results
        plot_distributions(time_grid, avg_time_gaussian, pitch_grid, avg_pitch_gaussian, genre)

        # Save averaged distributions for future use
        np.save(f"{genre}_avg_time.npy", avg_time_gaussian)
        np.save(f"{genre}_avg_pitch.npy", avg_pitch_gaussian)
