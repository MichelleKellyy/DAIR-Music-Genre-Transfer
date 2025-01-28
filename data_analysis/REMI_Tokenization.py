from miditok import REMI, TokenizerConfig
from pathlib import Path

# Tokenizer parameters
# TODO: Decide whether we want velocities as False
TOKENIZER_PARAMS = {
    "pitch_range": (21, 109),
    "beat_res": {(0, 4): 8, (4, 12): 4},  # Samples per beat
    "num_velocities": 32,  # Number of velocity bins
    "special_tokens": ["PAD", "BOS", "EOS", "MASK"],
    "use_chords": True,
    "use_rests": False,
    "use_tempos": True,
    "use_time_signatures": False,
    "use_programs": True,
    "num_tempos": 32,  # Number of tempo bins
    "tempo_range": (40, 250),  # (Min, Max)
}
config = TokenizerConfig(**TOKENIZER_PARAMS)

# Creates the tokenizer
tokenizer = REMI(config)

# TODO: Split MIDI files into one track per instrument
# TODO: Figure out vocabulary size for the training
# TODO: Train the tokenizer (BPE)

# Tokenize a MIDI file
tokens = tokenizer(Path("./dataset/000aec55332d26c818c0d6cf6af40010.mid"))  # Testing one midi file for now

# Convert to MIDI and save it
generated_midi = tokenizer(tokens)
generated_midi.dump_midi(Path("/tokenized_dataset/test_midi_tokenization.mid"))

# TODO: Tokenize the whole dataset and store it in JSON file as well as the tokenized MIDI files