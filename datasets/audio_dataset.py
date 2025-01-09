import torch
import numpy as np
import pandas as pd
import torchaudio
from pathlib import Path, PurePath
from torch.utils.data import Dataset

from utils.torchaudio_utils import TorchAudioUtils


class AudioDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        sr: int = 44100,
        step_size: float = 0.5,
        duration: int = 1,
        audio_channels: int = 2,
        audio_n_mels: float = 64,
        n_fft: int = 1024,
        hop_length: float = 1024 // 4,
        f_mask: int = 10,  # 15% of n_mels = 64
        t_mask: int = 25,  # 15% of 173
        **kwargs,
    ):
        self.df = df
        self.sr = sr
        self.step_size = step_size
        self.audio_channels = audio_channels
        self.duration = duration
        self.audio_n_mels = audio_n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.f_mask = f_mask
        self.t_mask = t_mask
        self.kwargs = kwargs

        # Get transform function
        self.transform_mel_func = torchaudio.transforms.MelSpectrogram(
            sr,
            n_mels=self.audio_n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        self.power_to_db_func = torchaudio.transforms.AmplitudeToDB(stype="power")

        # Set augumentation
        self.fmask = torchaudio.transforms.FrequencyMasking(freq_mask_param=self.f_mask)
        self.tmask = torchaudio.transforms.TimeMasking(time_mask_param=self.t_mask)

    def __len__(self) -> int:
        return self.df.shape[0]

    def __getitem__(self, index: int) -> None:
        row = self.df.iloc[index, :]
        # Get the audio file path
        audio_file_path = PurePath(f"./data/{row['cache_audio_path']}")
        # Get step index
        step_index = row["step"]
        # Get the label id
        label_id = row["label_id"]
        # Process the audio data
        audio_data = self.process_audio(audio_file_path, step_index)

        return audio_data, label_id

    def normalize(self, x):
        eps = 1e-6

        return (x - x.mean()) / (x.std() + eps)

    def process_audio(self, audio_path, step_index):
        sig, sr = TorchAudioUtils.read_slice_audio(
            audio_path, step_index, self.duration, self.step_size
        )
        sig, sr = TorchAudioUtils.resample(sig, sr, self.sr)
        sig, sr = TorchAudioUtils.rechannel(sig, sr, self.audio_channels)
        sig, sr = TorchAudioUtils.resize_length(sig, sr, self.duration)
        mel_sig = self.transform_mel_func(sig)
        mel_sig = self.power_to_db_func(mel_sig)

        # Standardize the mel spectrogram
        mel_sig = self.normalize(mel_sig)

        # Augmentation
        if self.kwargs.get("is_augment", False):
            mel_sig = self.fmask(mel_sig)
            mel_sig = self.tmask(mel_sig)

        return mel_sig


if __name__ == "__main__":
    df = pd.read_csv("./data/time_slices_50.csv")
    dataset = AudioDataset(df=df)
    mel_sig, label = dataset[200]
    print(mel_sig.shape, label)
    print(dataset.df.shape)
