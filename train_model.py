import argparse
import os
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import DataLoader
from create_dataset import create_dataset


def main():
    parser = argparse.ArgumentParser(description='Train the individual Transformer model')
    parser.add_argument('-f')  # Required for argument parser to work in Colab
    parser.add_argument('--train_folder', type=str, default='small_audio/')
    parser.add_argument('--val_folder', type=str, default='audio/')
    args = parser.parse_args()

    drive_path = '/content/drive/MyDrive/DL_data/'
    json_path_tng = f'{drive_path}nsynth-train/examples.json'
    json_path_val = f'{drive_path}nsynth-valid/examples.json'
    audio_input_path_tng = f'{drive_path}nsynth-train/{args.train_folder}'
    audio_input_path_val = f'{drive_path}nsynth-valid/{args.val_folder}'

    # Select GPU for runtime if available
    if not torch.cuda.is_available():
        device = torch.device("cpu")
        print('No GPU selected')
    else:
        device = torch.device("cuda")
        print(torch.cuda.get_device_name(device))

    start = datetime.now()
    # Create datasets
    tng_dataset = create_dataset(audio_input_path_tng, json_path_tng)
    # val_dataset = create_dataset(audio_input_path_val, json_path_val)

    # Create Data Loaders
    tng_dataloader = DataLoader(tng_dataset, batch_size=1, shuffle=False)
    # val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Test dataloader function
    img = next(iter(tng_dataloader))
    plt.imshow(img)
    plt.show()
    # print(f'label:{label}\nimg:\n{img}')
    print(f'img:\n{img}')

    end = datetime.now()
    print(f'\nelapsed time: {end - start}')


if __name__ == '__main__':
    main()
