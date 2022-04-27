import os
import torch
import librosa
import numpy as np
import json
from torch.utils.data import Dataset
import torch.nn.functional as F


def create_dataset_fast(audio_input_path, json_path, n_mels, n_fft, h_l):
    # Load JSON file data
    file = open(json_path, 'rb')
    metadata = json.load(file)
    file.close()
    # Create list of audio files
    sample_list = os.listdir(audio_input_path)

    data = []
    labels = []
    # Loop through files and store spectrogram and instrument family for each sample
    i = 0
    while i < len(sample_list)-10:
        # Append 10 labels
        labels.append(metadata[sample_list[i][:-4]]['instrument_family'])
        labels.append(metadata[sample_list[i+1][:-4]]['instrument_family'])
        labels.append(metadata[sample_list[i+2][:-4]]['instrument_family'])
        labels.append(metadata[sample_list[i+3][:-4]]['instrument_family'])
        labels.append(metadata[sample_list[i+4][:-4]]['instrument_family'])
        labels.append(metadata[sample_list[i+5][:-4]]['instrument_family'])
        labels.append(metadata[sample_list[i+6][:-4]]['instrument_family'])
        labels.append(metadata[sample_list[i+7][:-4]]['instrument_family'])
        labels.append(metadata[sample_list[i+8][:-4]]['instrument_family'])
        labels.append(metadata[sample_list[i+9][:-4]]['instrument_family'])

        # load the waveform y and sampling rate sr
        y_0, sr = librosa.load(f'{audio_input_path}{sample_list[i]}', sr=None)
        y_1, sr = librosa.load(f'{audio_input_path}{sample_list[i+1]}', sr=None)
        y_2, sr = librosa.load(f'{audio_input_path}{sample_list[i+2]}', sr=None)
        y_3, sr = librosa.load(f'{audio_input_path}{sample_list[i+3]}', sr=None)
        y_4, sr = librosa.load(f'{audio_input_path}{sample_list[i+4]}', sr=None)
        y_5, sr = librosa.load(f'{audio_input_path}{sample_list[i+5]}', sr=None)
        y_6, sr = librosa.load(f'{audio_input_path}{sample_list[i+6]}', sr=None)
        y_7, sr = librosa.load(f'{audio_input_path}{sample_list[i+7]}', sr=None)
        y_8, sr = librosa.load(f'{audio_input_path}{sample_list[i+8]}', sr=None)
        y_9, sr = librosa.load(f'{audio_input_path}{sample_list[i+9]}', sr=None)

        # convert to 2 dimensional spectogram format
        spectrogram_0 = librosa.feature.melspectrogram(y=y_0, sr=sr, n_mels=n_mels, n_fft=n_fft,
                                                     fmax=8000, hop_length=h_l)
        spectrogram_1 = librosa.feature.melspectrogram(y=y_1, sr=sr, n_mels=n_mels, n_fft=n_fft,
                                                     fmax=8000, hop_length=h_l)
        spectrogram_2 = librosa.feature.melspectrogram(y=y_2, sr=sr, n_mels=n_mels, n_fft=n_fft,
                                                     fmax=8000, hop_length=h_l)
        spectrogram_3 = librosa.feature.melspectrogram(y=y_3, sr=sr, n_mels=n_mels, n_fft=n_fft,
                                                     fmax=8000, hop_length=h_l)
        spectrogram_4 = librosa.feature.melspectrogram(y=y_4, sr=sr, n_mels=n_mels, n_fft=n_fft,
                                                     fmax=8000, hop_length=h_l)
        spectrogram_5 = librosa.feature.melspectrogram(y=y_5, sr=sr, n_mels=n_mels, n_fft=n_fft,
                                                     fmax=8000, hop_length=h_l)
        spectrogram_6 = librosa.feature.melspectrogram(y=y_6, sr=sr, n_mels=n_mels, n_fft=n_fft,
                                                     fmax=8000, hop_length=h_l)
        spectrogram_7 = librosa.feature.melspectrogram(y=y_7, sr=sr, n_mels=n_mels, n_fft=n_fft,
                                                     fmax=8000, hop_length=h_l)
        spectrogram_8 = librosa.feature.melspectrogram(y=y_8, sr=sr, n_mels=n_mels, n_fft=n_fft,
                                                     fmax=8000, hop_length=h_l)
        spectrogram_9 = librosa.feature.melspectrogram(y=y_9, sr=sr, n_mels=n_mels, n_fft=n_fft,
                                                     fmax=8000, hop_length=h_l)

        # Convert raw power to dB
        S_dB_0 = librosa.power_to_db(spectrogram_0, ref=np.max)
        data.append(S_dB_0)
        S_dB_1 = librosa.power_to_db(spectrogram_1, ref=np.max)
        data.append(S_dB_1)
        S_dB_2 = librosa.power_to_db(spectrogram_2, ref=np.max)
        data.append(S_dB_2)
        S_dB_3 = librosa.power_to_db(spectrogram_3, ref=np.max)
        data.append(S_dB_3)
        S_dB_4 = librosa.power_to_db(spectrogram_4, ref=np.max)
        data.append(S_dB_4)
        S_dB_5 = librosa.power_to_db(spectrogram_5, ref=np.max)
        data.append(S_dB_5)
        S_dB_6 = librosa.power_to_db(spectrogram_6, ref=np.max)
        data.append(S_dB_6)
        S_dB_7 = librosa.power_to_db(spectrogram_7, ref=np.max)
        data.append(S_dB_7)
        S_dB_8 = librosa.power_to_db(spectrogram_8, ref=np.max)
        data.append(S_dB_8)
        S_dB_9 = librosa.power_to_db(spectrogram_9, ref=np.max)
        data.append(S_dB_9)

        i += 10

    for file in sample_list[i:]:
        labels.append(metadata[file[:-4]]['instrument_family'])
        y_0, sr = librosa.load(f'{audio_input_path}{file}', sr=None)
        spectrogram_0 = librosa.feature.melspectrogram(y=y_0, sr=sr, n_mels=n_mels, n_fft=n_fft,
                                                       fmax=8000, hop_length=h_l)
        S_dB_0 = librosa.power_to_db(spectrogram_0, ref=np.max)
        data.append(S_dB_0)

    data_np = torch.tensor(np.stack(data))
    # labels = F.one_hot(torch.tensor(np.stack(labels)), num_classes=11)
    labels = F.one_hot(torch.tensor(np.stack(labels)), num_classes=11).type(torch.float32)
    return AudioSpectrogramDataset(data_np, labels)


class AudioSpectrogramDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.labels[index]