import json


path = 'data/nsynth-train/examples.json'
file = open(path, 'rb')
metadata = json.load(file)
file.close()

# print(f'{metadata["guitar_acoustic_001-082-050"]["instrument_family"]}')

family_info = {
    0: {'count': 0, 'proportion': 0},
    1: {'count': 0, 'proportion': 0},
    2: {'count': 0, 'proportion': 0},
    3: {'count': 0, 'proportion': 0},
    4: {'count': 0, 'proportion': 0},
    5: {'count': 0, 'proportion': 0},
    6: {'count': 0, 'proportion': 0},
    7: {'count': 0, 'proportion': 0},
    8: {'count': 0, 'proportion': 0},
    9: {'count': 0, 'proportion': 0},
    10: {'count': 0, 'proportion': 0}
}

total_count = 0
for key in metadata.keys():
    total_count += 1
    family_info[metadata[key]["instrument_family"]]["count"] += 1
print(f'total samples: {total_count}')

for key, value in family_info.items():
    value["proportion"] += value["count"] / total_count
    print(f'family {key}: count = {value["count"]}, proportion = {value["proportion"]}')
