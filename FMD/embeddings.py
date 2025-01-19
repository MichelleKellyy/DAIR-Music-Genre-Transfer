import pretty_midi
import numpy as np
import matplotlib.pyplot as plt

# Gaussian function
def gaussian(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Compute temporal and pitch distributions for a single MIDI file
def extract_distributions(midi_file_path, time_steps=1000, sigma=1):
   
    midi_data = pretty_midi.PrettyMIDI(midi_file_path)

    # Temporal Gaussian distribution
    time_grid = np.linspace(0, midi_data.get_end_time(), time_steps)
    time_gaussian = np.zeros_like(time_grid)

    # Pitch Gaussian distribution
    pitch_grid = np.arange(0, 128)  # MIDI pitch range
    pitch_gaussian = np.zeros_like(pitch_grid, dtype=float)

    # Process each note in the MIDI file
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            # Temporal Gaussian
            center_time = (note.start + note.end) / 2
            time_gaussian += gaussian(time_grid, center_time, sigma)

            # Pitch Gaussian
            pitch_gaussian += gaussian(pitch_grid, note.pitch, sigma)

    return time_grid, time_gaussian, pitch_grid, pitch_gaussian

# Visualization function for distributions
def plot_distributions(time_grid, time_gaussian, pitch_grid, pitch_gaussian, file_name):
    # Plot temporal distribution
    plt.figure(figsize=(10, 4))
    plt.plot(time_grid, time_gaussian, label='Temporal Distribution')
    plt.xlabel('Time (s)')
    plt.ylabel('Density')
    plt.title(f'Temporal Distribution for {file_name}')
    plt.legend()
    plt.show()

    # Plot pitch distribution
    plt.figure(figsize=(10, 4))
    plt.plot(pitch_grid, pitch_gaussian, label='Pitch Distribution')
    plt.xlabel('Pitch')
    plt.ylabel('Density')
    plt.title(f'Pitch Distribution for {file_name}')
    plt.legend()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Path to a single MIDI file
    midi_file_path = "data/AC_DC_-_Back_In_Black.mid"

    # Extract distributions
    time_grid, time_gaussian, pitch_grid, pitch_gaussian = extract_distributions(midi_file_path)

    # Plot distributions
    plot_distributions(time_grid, time_gaussian, pitch_grid, pitch_gaussian, "AC_DC_-_Back_In_Black")
