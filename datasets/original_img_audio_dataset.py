from pathlib import Path

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from utils.torchaudio_utils import TorchAudioUtils


train_compose_list = [
    transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
    transforms.ToTensor(),  # Convert PIL image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]
test_compose_list = [
    transforms.ToTensor(),  # Convert PIL image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]


class OriginalImageAudioDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        model_type: str = "resnet",  # "resnet", "vit"
        image_size: int = 224,
        sr: int = 44100,
        step_size: float = 0.5,
        duration: float = 1,
        audio_channels: int = 2,
        audio_n_mels: int = 64,
        n_fft: int = 1024,
        hop_length: float = 1024 // 4,
        f_mask: int = 10,
        t_mask: int = 25,
        **kwargs,
    ):
        self.df = df
        self.sr = sr
        self.step_size = step_size
        self.duration = duration
        self.audio_channels = audio_channels
        self.audio_n_mels = audio_n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.f_mask = f_mask
        self.t_mask = t_mask
        self.kwargs = kwargs
        self.is_augment = kwargs.get("is_augment", None)
        self.image_size = (image_size, image_size)

        if self.is_augment:
            if model_type == "resnet":
                self.transform = transforms.Compose(train_compose_list)
            elif model_type == "vit":
                train_compose_list.insert(0, transforms.Resize(self.image_size))
                self.transform = transforms.Compose(train_compose_list)
        else:
            if model_type == "resnet":
                self.transform = transforms.Compose(test_compose_list)
            elif model_type == "vit":
                test_compose_list.insert(0, transforms.Resize(self.image_size))
                self.transform = transforms.Compose(test_compose_list)

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

        # Image data
        original_image_path = Path(f'./data/{row["img_path"]}')
        original_image = Image.open(original_image_path)
        original_image_tensor = self.transform(original_image)
        label_id = row["label_id"]

        # Audio data
        audio_file_path = Path(f'./data/{row["cache_audio_path"]}')
        step_index = row["step"]
        audio_data = self.process_audio(audio_file_path, step_index)
        # audio_data = audio_data.mean(dim=0)

        return audio_data, original_image_tensor, label_id

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

        # Augmentation
        if self.is_augment:
            mel_sig = self.fmask(mel_sig)
            mel_sig = self.tmask(mel_sig)

        # Standardize the mel spectrogram
        mel_sig_s = self.normalize(mel_sig)

        return mel_sig_s


if __name__ == "__main__":
    meta_file_path = Path("./data/time_slices_50.csv")
    df = pd.read_csv(meta_file_path)
    dataset = OriginalImageAudioDataset(df, model_type="vit", image_size=224)
    audio_data, img_tensor, label = dataset[100]
    print(audio_data.shape, img_tensor.shape, label)
    print(dataset.df.shape)
    print("Done")
