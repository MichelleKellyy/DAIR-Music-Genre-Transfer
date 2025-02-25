"""
Preprocess the piano roll data.
Current structure:
Piano_Rolls#
    <code>.mid  Midi song code. The genre for this song is in dataset.json.
        #.npz   contains the piano roll in csc matrix format
        #       zip file containing a file info.json which has the instrument name

What this file does:
1. Extract the instrument name from info.json for each instrument
2. Rename the .npz file to the instrument name
3. Delete the info.json and zip file
"""

import os
import json
import zipfile

# Loop through each giant folder
valid_instruments = ['drums', 'bass', 'piano', 'strings', 'guitar', 'alto', 'soprano', 'tenor', 'vocals', 'voice', 'trumpet', 'brass', 'organ', 'flute', 'trombone']

for i in range(1):
    for midi_folder in os.listdir(f"Piano_Rolls{i}"):
        for file in os.listdir(os.path.join(f"Piano_Rolls{i}", midi_folder)):
            if file[:-4].isnumeric() and file[-4:] == '.npz':
                file_path = os.path.join(f"Piano_Rolls{i}", midi_folder)
                file_name = file[:-4]
                # 1. Extract the instrument name from info.json for each instrument
                
                # Unzip the file
                with zipfile.ZipFile(os.path.join(file_path, file_name), 'r') as zip_ref:
                    zip_ref.extractall(file_path)

                # Get the instrument name from info.json
                with open(os.path.join(file_path, 'info.json')) as info_json:
                    info = json.load(info_json)
                    instrument_name = info['0']['name'].strip().lower()
                    if instrument_name in valid_instruments:
                        # Assert that there at most one of each instrument type
                        try:
                            assert instrument_name not in os.listdir(file_path)
                        except AssertionError:
                            print(midi_folder, file)

                        # Rename the .npz file to the instrument name
                        os.rename(os.path.join(file_path, file), os.path.join(file_path, f"{instrument_name}.npz"))

                        

                # Delete info.json and the zipped file
                os.remove(os.path.join(file_path, 'info.json'))
                # os.remove(os.path.join(file_path, file_name))

                
