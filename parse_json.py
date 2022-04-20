import json


dataset_size = 10000
path = 'data/nsynth-train/examples.json'
file = open(path, 'rb')
metadata = json.load(file)
file.close()

# print(f'{metadata["guitar_acoustic_001-082-050"]["instrument_family"]}')

# Define dictionary to store info about each instrument family
family_info = {
    0: {'raw_count': 0, 'proportion': 0, 'scaled_count': 0, 'sample_counter': 0},
    1: {'raw_count': 0, 'proportion': 0, 'scaled_count': 0, 'sample_counter': 0},
    2: {'raw_count': 0, 'proportion': 0, 'scaled_count': 0, 'sample_counter': 0},
    3: {'raw_count': 0, 'proportion': 0, 'scaled_count': 0, 'sample_counter': 0},
    4: {'raw_count': 0, 'proportion': 0, 'scaled_count': 0, 'sample_counter': 0},
    5: {'raw_count': 0, 'proportion': 0, 'scaled_count': 0, 'sample_counter': 0},
    6: {'raw_count': 0, 'proportion': 0, 'scaled_count': 0, 'sample_counter': 0},
    7: {'raw_count': 0, 'proportion': 0, 'scaled_count': 0, 'sample_counter': 0},
    8: {'raw_count': 0, 'proportion': 0, 'scaled_count': 0, 'sample_counter': 0},
    9: {'raw_count': 0, 'proportion': 0, 'scaled_count': 0, 'sample_counter': 0},
    10: {'raw_count': 0, 'proportion': 0, 'scaled_count': 0, 'sample_counter': 0}
}

total_count = 0
for key in metadata.keys():
    total_count += 1
    family_info[metadata[key]['instrument_family']]['raw_count'] += 1
# print(f'total samples: {total_count}')

# Remove family 9 from training data (since it is not present in validation or test data)
total_count -= family_info[9]['raw_count']
# print(f'total samples without family 9: {total_count}')
family_info[9]['raw_count'] = 0
family_info[9]['proportion'] = 0

for inner_key in family_info.values():
    inner_key['proportion'] += inner_key['raw_count'] / total_count

for inner_key in family_info.values():
    inner_key['scaled_count'] += (inner_key['proportion'] * dataset_size) // 1

# for outer_key, inner_key in family_info.items():
#     print(f'family {outer_key}:')
#     for key, value in inner_key.items():
#         print(f'{key} = {value}')

sample_list = []
for key in metadata.keys():
    # check instrument family
    sample = family_info[metadata[key]['instrument_family']]
    # if counter for instrument family is below the max value
    if sample['sample_counter'] < sample['scaled_count']:
        # store sample name
        sample_list.append(f'{key}.wav')
        # increment family counter
        family_info[metadata[key]['instrument_family']]['sample_counter'] += 1

# for outer_key, inner_key in family_info.items():
#     print(f'family {outer_key}: {inner_key["sample_counter"]} samples')

# print('sample list:\n')
# for name in sample_list:
#     print(name)