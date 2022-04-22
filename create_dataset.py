import os
import torch
import librosa
import numpy as np
import json
from torch.utils.data import Dataset
import torch.nn.functional as F


def create_dataset(audio_input_path, json_path):
    # Load JSON file data
    file = open(json_path, 'rb')
    metadata = json.load(file)
    file.close()
    # Create list of audio files
    sample_list = os.listdir(audio_input_path)

    data = []
    labels = []
    # Loop through files and store spectrogram and instrument family for each sample
    for file in sample_list:
        labels.append(metadata[file[:-4]]['instrument_family'])
        # load the waveform y and sampling rate sr
        y, sr = librosa.load(f'{audio_input_path}{file}', sr=None)
        # convert to 2 dimensional spectogram format
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000, hop_length=502)
        # Convert raw power to dB
        S_dB = librosa.power_to_db(spectrogram, ref=np.max)
        data.append(S_dB)

    data_np = torch.tensor(np.stack(data))
    # labels = F.one_hot(torch.tensor(np.stack(labels)), num_classes=11)
    labels = torch.tensor(F.one_hot(torch.tensor(np.stack(labels)), num_classes=11), dtype=torch.float32)
    return AudioSpectogramDataset(data_np, labels)


class AudioSpectogramDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
