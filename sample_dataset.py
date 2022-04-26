import torch
from datetime import datetime
import shutil
from SmallTrainSet import SmallTrainSet


def main():
    path_json = 'data/nsynth-train/examples.json'
    path_audio_inputs = 'data/nsynth-train/audio/'

    # Select GPU for runtime if available
    if not torch.cuda.is_available():
        device = torch.device("cpu")
        print('No GPU selected')
    else:
        device = torch.device("cuda")
        print(torch.cuda.get_device_name(device))

    start = datetime.now()
    # Create training dataset
    train_set_files = SmallTrainSet(50000, path_json)
    sample_list = train_set_files.sample_list
    for file in sample_list:
        shutil.copy(f'{path_audio_inputs}{file}', f'data/nsynth-train/medium_audio/{file}')

    end = datetime.now()
    print(f'\nelapsed time: {end - start}')


if __name__ == '__main__':
    main()
