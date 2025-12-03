"""
Preprocess the piano roll data.
Current structure:
Piano_Rolls#
    <code>.mid  Midi song code. The genre for this song is in dataset.json.
        #.npz   contains the piano roll in csc matrix format
        #       zip file containing a file info.json which has the instrument name

What this file does:
1. Extract the instrument name from info.json for each instrument
2. Rename the .npz file to the instrument name (<index>-<instrument>.npz)
3. Delete files that do not correspond to valid instruments
"""

import os
import json
import zipfile


# Loop through each giant folder
valid_instruments = {
    'piano': 0,
    'piano1': 0,
    'a.piano': 0,
    'a.piano 1': 0,
    'a.piano 2': 0,
    'piano 1': 0,
    'piano 2': 0,
    'acoustic grand piano': 0,
    'organ': 0,
    'organ 3': 0,
    'piano (hi)': 0,
    'piano (lo)': 0,
    'synth': 0,
    'marimba': 0,
    'vibraphone': 0,
    'keyboard': 0,
    'e.piano 1': 0,
    'e.piano 2': 0,
    'grand piano': 0,

    'bass': 1,
    'acou bass': 1,
    'acoustic bass': 1,
    'fretless bass': 1,

    'strings': 2,
    'guitar': 2,
    'guitar 1': 2,
    'guitar 2': 2,
    'guitar 3': 2,
    'lead guitar': 2,
    'steel gtr': 2,
    'jazz gtr': 2,
    'harp': 2,
    'cello': 2,
    'violin': 2,
    'muted gtr': 2,
    'clean guitar': 2,
    'clean gtr': 2,
    'nylon gtr': 2,
    'acoustic guitar': 2,
    'banjo': 2,
    'viola': 2,

    'flute': 3,
    'clarinet': 3,
    'harmonica': 3,
    'oboe': 3,
    'piccolo': 3,
    'bassoon': 3,

    'trumpet': 4,
    'trombone': 4,
    'brass': 4,
    'brass 1': 4,
    'sax': 4,
    'alto sax': 4,
    'tenor sax': 4,
    'tuba': 4,
    'french horn': 4,
    'fr.horn': 4,
}

for i in range(117):
    root_path = f"/home/stuart/Downloads/Piano Rolls/Piano_Rolls{i}"
    for midi_folder in os.listdir(root_path):
        for file in os.listdir(os.path.join(root_path, midi_folder)):
            if file[:-4].isnumeric() and file[-4:] == '.npz':
                file_path = os.path.join(root_path, midi_folder)
                file_name = file[:-4]
                # Extract the instrument name from info.json for each instrument
                
                # Unzip the file

                # Missing file, delete and continue
                if "b86f8509033ab68abddc3f157a72f618.mid/680" in os.path.join(file_path, file_name):
                    os.remove(os.path.join(file_path, file_name))
                    os.remove(os.path.join(file_path, file))
                    continue

                with zipfile.ZipFile(os.path.join(file_path, file_name), 'r') as zip_ref:
                    zip_ref.extractall(file_path)

                # Get the instrument name from info.json
                with open(os.path.join(file_path, 'info.json')) as info_json:
                    info = json.load(info_json)
                    instrument_name = info['0']['name'].strip().lower()
                    if instrument_name in valid_instruments:
                        # Rename the .npz file to the instrument name
                        os.rename(os.path.join(file_path, file), os.path.join(file_path, f"{file_name}-{instrument_name}.npz"))
                    else:
                        os.remove(os.path.join(file_path, file))
                        os.remove(os.path.join(file_path, file_name))

                # Delete info.json
                os.remove(os.path.join(file_path, 'info.json'))
                # os.remove(os.path.join(file_path, file_name))

        # If the folder has no valid instruments, delete it
        if len(os.listdir(os.path.join(root_path, midi_folder))) == 0:
            os.rmdir(os.path.join(root_path, midi_folder))
