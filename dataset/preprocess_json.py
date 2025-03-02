"""
Convert the dataset.json file into dictionary format for more eficient retrieval of MIDI information.
"""

import json

data_dict = {}

# Load file
with open("dataset/dataset.json", 'r') as file:
    data = json.load(file)

    # Convert entries to dictionary format
    for entry in data:
        location = entry["location"].split('/')[-1]
        data_dict[location] = entry["genre"]


# Save updated json file
json_object = json.dumps(data_dict, indent=4)
with open("dataset.json", 'w') as file:
    file.write(json_object)