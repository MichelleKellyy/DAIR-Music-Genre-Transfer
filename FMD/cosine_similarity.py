from scipy.spatial.distance import cosine
import numpy as np
from genre_processing import compute_distributions, plot_distributions

def compute_cosine_similarity(unknown_distribution, genre_average):
    """
    Compute cosine similarity between two distributions.
    """
    return 1 - cosine(unknown_distribution, genre_average)

def compare_to_genre(unknown_file, genre_name, avg_time_path, avg_pitch_path, time_steps=1000, sigma=1):
    """
    Compare an unknown file's distributions to a genre's average distributions.
    """
    # Load average distributions for the genre
    avg_time_gaussian = np.load(avg_time_path)
    avg_pitch_gaussian = np.load(avg_pitch_path)

    # Compute distributions for the unknown file
    time_gaussian, pitch_gaussian = compute_distributions(unknown_file, time_steps, sigma)

    if time_gaussian is None or pitch_gaussian is None:
        print(f"Error processing the unknown file: {unknown_file}")
        return None, None, None

    # Compute cosine similarities
    time_similarity = compute_cosine_similarity(time_gaussian, avg_time_gaussian)
    pitch_similarity = compute_cosine_similarity(pitch_gaussian, avg_pitch_gaussian)

    # Compute overall similarity (weighted average)
    overall_similarity = 0.75 * time_similarity + 0.25 * pitch_similarity

    return time_similarity, pitch_similarity, overall_similarity

# Example usage
if __name__ == "__main__":
    genres = ["electronic", "pop", "rock", "classical"]
    for genre in genres:
        # Paths to precomputed average distributions for the "electronic" genre
        avg_time_path = "genre_gaussian_data//" + genre + "_avg_time.npy"
        avg_pitch_path = "genre_gaussian_data//" + genre + "_avg_pitch.npy"

        # Path to the unknown MIDI file
        unknown_file = "data/Queen - Bohemian Rhapsody.mid"  # Replace with the actual file path

        # Compare the unknown file to the "electronic" genre
        time_sim, pitch_sim, overall_sim = compare_to_genre(
            unknown_file, genre, avg_time_path, avg_pitch_path
        )

        if overall_sim is not None:
            print(f"Similarity to genre '{genre}':")
            print(f"  Temporal Similarity: {time_sim:.2f}")
            print(f"  Pitch Similarity: {pitch_sim:.2f}")
            print(f"  Overall Similarity: {overall_sim:.2f}")
