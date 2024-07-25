import os

import torch
import torchaudio
from torchaudio.transforms import GriffinLim, InverseSpectrogram, InverseMelScale, Resample, Speed

from einops import rearrange
from accelerate import Accelerator

from torch.optim import Adam

from e2_tts_pytorch.e2_tts import (
    E2TTS,
    DurationPredictor,
    MelSpec
)

duration_predictor = DurationPredictor(
    transformer = dict(
        dim = 80,
        depth = 2,
    )
)

model = E2TTS(
    # duration_predictor = duration_predictor,
    transformer = dict(
        dim = 512,
        depth = 12,
        skip_connect_type = 'none'
    )
)

n_fft = 1024
sample_rate = 20500
checkpoint_path = "./e2tts.pt"
ref_length = 300
duration=1000

def exists(v):
    return v is not None

def vocoder(melspec):
    inverse_melscale_transform = InverseMelScale(n_stft=n_fft // 2 + 1, n_mels=100, sample_rate=sample_rate, norm="slaney", f_min=0, f_max=8000)
    spectrogram = inverse_melscale_transform(melspec)
    transform = GriffinLim(n_fft=n_fft, hop_length=256, power=2)
    waveform = transform(spectrogram)
    return waveform

def load_checkpoint(checkpoint_path, model, accelerator, optimizer):
    if not exists(checkpoint_path) or not os.path.exists(checkpoint_path):
        return 0

    checkpoint = torch.load(checkpoint_path)
    accelerator.unwrap_model(model).load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['step']

accelerator = Accelerator(
            log_with="all",
        )

optimizer = Adam(model.parameters(), lr=1e-4)

start_step = load_checkpoint(checkpoint_path=checkpoint_path, model=model, accelerator=accelerator, optimizer=optimizer)

ref_waveform, ref_sample_rate = torchaudio.load("ref.wav", normalize=True)
resampler = Resample(orig_freq=ref_sample_rate, new_freq=sample_rate)
ref_waveform = resampler(ref_waveform)
speed_factor = sample_rate / ref_sample_rate
respeed = Speed(ref_sample_rate, speed_factor)
ref_waveform = respeed(ref_waveform)
ref_waveform_resampled = ref_waveform[0]

mel_model = MelSpec()
mel = mel_model(ref_waveform_resampled)
mel = torch.cat([mel, mel], dim=0)
mel = rearrange(mel, 'b d n -> b n d')
print(mel.shape)

text = ["It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.", "Waves crashed against the cliffs, their thunderous applause echoing for miles."]
sample = model.sample(mel[:,:ref_length], text = text, vocoder=vocoder, duration=duration, mask_ref=True)
sample = sample.to('cpu')

waveform = sample
print(waveform.shape)

mono_channel_1 = waveform[0,ref_length:].unsqueeze(0)
mono_channel_2 = waveform[1,ref_length:].unsqueeze(0)

torchaudio.save("output_channel_1.wav", mono_channel_1, sample_rate)
torchaudio.save("output_channel_2.wav", mono_channel_2, sample_rate)