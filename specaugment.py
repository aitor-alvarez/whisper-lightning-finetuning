from torchaudio import transforms
from torch import nn, Tensor

class SpecAugment(nn.Module):
    def __init__(self, input_fq, resample_fq=16000, n_fft=1024, n_mel=256, stretch_factor=0.8):
        super().__init__()
        self.resample = transforms.Resample(orig_freq=input_fq, new_freq=resample_fq)
        self.spct = transforms.Spectrogram(n_fft=n_fft, power=2)
        self.spec_augment = nn.Sequential(
            transforms.TimeStretch(stretch_factor, fixed_rate=True),
            transforms.FrequencyMasking(freq_mask_param=80),
            transforms.TimeMasking(time_mask_param=80),
        )

        self.mel_scale = transforms.MelScale(n_mels=n_mel, sample_rate=resample_fq, n_stft=n_fft // 2 + 1)

    def forward(self, wav: Tensor) -> Tensor:
        resample = self.resample(wav)
        spectro = self.spct(resample)
        spec_augment = self.spec_augment(spectro)
        mel_scaled = self.mel_scale(spec_augment)
        return mel_scaled