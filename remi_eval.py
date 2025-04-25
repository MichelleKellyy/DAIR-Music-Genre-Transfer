import torch
from miditok import REMI
from model import get_model
from opts import args

# Load the model and tokenizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = REMI(params="C:/Users/Miche/Desktop/SMGT/tokenizer_12.json")

# Load the trained model
checkpoint = torch.load("c:/Users/Miche/Downloads/REMI_TEST_CHECKPOINTS/sample_experiment/epoch_1_loss7.767859131129077.pt")
model = get_model(args).to(device)
model.load_state_dict(checkpoint['model'])
model.eval()

# Function to tokenize MIDI file (using miditok's REMI tokenizer)
def tokenize_midi(midi_file_path):
    # Use the encode method to process the MIDI file
    tokenized_data = tokenizer.encode(midi_file_path)
    return tokenized_data

# Function to convert tokenized output back to MIDI
def detokenize_to_midi(tokenized_output):
    # Decode the tokenized output back to MIDI using REMI
    midi_bytes = tokenizer.decode(tokenized_output)
    return midi_bytes

# Load and tokenize the input MIDI
midi_file_path = "C:/Users/Miche/Desktop/input_songs/i1.mid"
tokenized_data = tokenize_midi(midi_file_path)

# Check the current length of the tokenized data
current_length = len(tokenized_data)
print(f"Current sequence length: {current_length}")

# Define the expected sequence length (95 based on model's configuration)
expected_sequence_length = 95

# Padding or truncating the sequence to match the expected length
if current_length < expected_sequence_length:
    # If the sequence is too short, pad it
    padding_size = expected_sequence_length - current_length
    print(f"Padding size: {padding_size}")
    tokenized_data.extend([0] * padding_size)  # Padding with zeros
elif current_length > expected_sequence_length:
    # If the sequence is too long, truncate it
    tokenized_data = tokenized_data[:expected_sequence_length]

# Check the new length after padding/truncation
print(f"New sequence length: {len(tokenized_data)}")

# Now, convert the padded/truncated tokenized data to a tensor
tokenized_data = torch.tensor(tokenized_data).unsqueeze(0).to(device)  # Adds the batch dimension

# Check the shape of the tensor before passing it to the model
print(f"Shape of tokenized data tensor: {tokenized_data.shape}")

# Ensure the shape matches the expected input to the model
assert tokenized_data.size(1) == expected_sequence_length, "The sequence length doesn't match the expected value"

# Run the model for genre transfer
with torch.no_grad():
    y_pred, mean_genre, log_var_genre, mean_instance, log_var_instance = model(tokenized_data)
    
    # Genre transfer (shuffling genres)
    shuffled_mean_genre = shuffle(mean_genre)
    z_shuffled = torch.cat((shuffled_mean_genre, mean_instance), dim=-1)

    seq_len = tokenized_data.size(1)
    output_tokenized_data = model.decoder(z_shuffled, seq_len)

# Detokenize the output and save the MIDI file
output_midi_data = detokenize_to_midi(output_tokenized_data)

# Save the generated MIDI
output_midi_path = "C:/Users/Miche/Desktop/input_songs/o1.mid"
with open(output_midi_path, 'wb') as f:
    f.write(output_midi_data)

print(f"Genre-transferred MIDI saved to: {output_midi_path}")