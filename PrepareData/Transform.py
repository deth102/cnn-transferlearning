import numpy as np
import torch

N_FFT = 256
HOP   = 64

def transformation(seg, Fourier_transform):

    if Fourier_transform == "FFT":
        seg = np.asarray(seg, dtype=np.float32)
        fft_val = np.abs(np.fft.fft(seg))[: len(seg)//2] / len(seg)
        return fft_val.astype(np.float32)

    elif Fourier_transform == "STFT":
        seg_t = torch.tensor(seg, dtype=torch.float32)
        window = torch.hann_window(N_FFT, device=seg_t.device)

        stft_val = torch.stft(
            seg_t,
            n_fft=N_FFT,
            hop_length=HOP,
            window=window,
            return_complex=True,
            center=True
        )
        return torch.abs(stft_val)
