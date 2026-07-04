
import torch
import torchaudio
from torch.utils.data import Dataset
import torch.nn.functional as F


class BirdDatasetV2(Dataset):
    def __init__(
        self,
        file_paths,
        labels,
        sample_rate=32000,
        duration_seconds=5,
        n_fft=1024,
        hop_length=512,
        n_mels=64,
        silence_threshold=0.02
    ):
        self.file_paths = file_paths
        self.labels = labels
        self.sample_rate = sample_rate
        self.target_samples = sample_rate * duration_seconds
        self.silence_threshold = silence_threshold

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )

    def __len__(self):
        return len(self.file_paths)

    def to_mono(self, waveform):
        return waveform.mean(dim=0, keepdim=True)

    def resample_if_needed(self, waveform, sr):
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=self.sample_rate
            )
            waveform = resampler(waveform)

        return waveform

    def trim_silence(self, waveform):
        energy = waveform.abs().squeeze()

        active = energy > self.silence_threshold

        if active.sum() == 0:
            return waveform

        start = active.nonzero()[0].item()
        end = active.nonzero()[-1].item()

        return waveform[:, start:end + 1]

    def fix_waveform_length(self, waveform):
        length = waveform.shape[1]

        if length > self.target_samples:
            waveform = waveform[:, :self.target_samples]

        elif length < self.target_samples:
            pad_amount = self.target_samples - length
            waveform = F.pad(waveform, (0, pad_amount))

        return waveform

    def normalize_waveform(self, waveform):
        max_val = waveform.abs().max()

        if max_val > 0:
            waveform = waveform / max_val

        return waveform

    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.file_paths[idx])

        waveform = self.to_mono(waveform)
        waveform = self.resample_if_needed(waveform, sr)
        waveform = self.normalize_waveform(waveform)
        waveform = self.trim_silence(waveform)
        waveform = self.fix_waveform_length(waveform)

        mel_spec = self.mel_transform(waveform)

        label = torch.tensor(
            self.labels[idx],
            dtype=torch.long
        )

        return mel_spec, label
