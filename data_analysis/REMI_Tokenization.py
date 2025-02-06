from miditok import REMI, TokenizerConfig
from pathlib import Path

# Tokenizer parameters
TOKENIZER_PARAMS = {
    "pitch_range": (21, 109), # Range of note pitches to use
    "beat_res": {(0, 4): 8, (4, 12): 4},  # Samples per beat
    "special_tokens": ["PAD", "BOS", "EOS", "MASK"],
    "use_velocities": False,
    "use_chords": True,
    "use_rests": True,
    "use_tempos": True,
    "use_time_signatures": False,
    "use_programs": True,
    "use_pitch_drum_tokens": True,
    "num_tempos": 32,  # Number of tempo bins
    "tempo_range": (40, 250),  # (Min, Max)
    "chord_unknown": None, # Set unknown chords to None
    "programs": list(range(-1, 128)),  # Sequence of MIDI programs to use
    "one_token_stream_for_programs": False,
    "program_changes": False,  # Place Program token for each note
}
config = TokenizerConfig(**TOKENIZER_PARAMS)

# Creates the tokenizer
tokenizer = REMI(config)
midi_paths = list(Path("./dataset").glob('*.mid'))

# Train the tokenizer (BPE)
tokenizer.train(vocab_size=30000, model="BPE", files_paths=midi_paths)
tokenizer.save("tokenizer.json")

# Tokenize a MIDI file
tokens = tokenizer(midi_paths)  # Testing one midi file for now
