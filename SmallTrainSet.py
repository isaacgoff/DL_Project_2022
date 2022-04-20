import json


class SmallTrainSet:

    def __init__(self, dataset_size, path):
        self.dataset_size = dataset_size
        self.path = path

        # Load JSON file data
        file = open(path, 'rb')
        self.metadata = json.load(file)
        file.close()

        # Define dictionary to store info about each instrument family
        self.family_info = {
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
        
        self.total_count = 0
        self.sample_list = self.sample_training_set()

    def sample_training_set(self):
        for key in self.metadata.keys():
            self.total_count += 1
            self.family_info[self.metadata[key]['instrument_family']]['raw_count'] += 1
        # print(f'total samples: {total_count}')

        # Remove family 9 from training data (since it is not present in validation or test data)
        self.total_count -= self.family_info[9]['raw_count']
        # print(f'total samples without family 9: {total_count}')
        self.family_info[9]['raw_count'] = 0
        self.family_info[9]['proportion'] = 0

        # Calculate relative frequency of each instrument family
        for inner_key in self.family_info.values():
            inner_key['proportion'] += inner_key['raw_count'] / self.total_count

        # Calculate the number of samples to gather for each family based on desired dataset size
        for inner_key in self.family_info.values():
            inner_key['scaled_count'] += (inner_key['proportion'] * self.dataset_size) // 1

        # for outer_key, inner_key in family_info.items():
        #     print(f'family {outer_key}:')
        #     for key, value in inner_key.items():
        #         print(f'{key} = {value}')

        sample_list = []
        for key in self.metadata.keys():
            # set instrument family
            sample = self.family_info[self.metadata[key]['instrument_family']]
            # if counter for instrument family is below the max value, store the sample name
            if sample['sample_counter'] < sample['scaled_count']:
                # store sample name
                sample_list.append(f'{key}.wav')
                # increment family counter
                self.family_info[self.metadata[key]['instrument_family']]['sample_counter'] += 1

        # for outer_key, inner_key in family_info.items():
        #     print(f'family {outer_key}: {inner_key["sample_counter"]} samples')

        # print('sample list:\n')
        # for name in sample_list:
        #     print(name)

        return sample_list
