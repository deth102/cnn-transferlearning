import numpy as np
import torch

N_FFT = 256
HOP   = 64

def transformation(seg, Fourier_transform):

    # ===== FFT =====
    if Fourier_transform == "FFT":
        seg = np.asarray(seg, dtype=np.float32)

        # tránh NaN khi tín hiệu constant
        if np.max(seg) - np.min(seg) < 1e-9:
            return np.zeros(len(seg)//2, dtype=np.float32)

        fft_val = np.abs(np.fft.fft(seg))[: len(seg)//2]
        fft_val /= len(seg)

        return np.nan_to_num(fft_val, nan=0.0).astype(np.float32)


    # ===== STFT =====
    elif Fourier_transform == "STFT":
        seg_t = torch.tensor(seg, dtype=torch.float32)

        # tránh NaN khi tín hiệu constant
        if torch.max(seg_t) - torch.min(seg_t) < 1e-6:
            return torch.zeros((N_FFT//2+1, 10))

        window = torch.hann_window(N_FFT, device=seg_t.device)

        stft_val = torch.stft(
            seg_t,
            n_fft=N_FFT,
            hop_length=HOP,
            window=window,
            return_complex=True,
            center=True
        )

        mag = torch.abs(stft_val)
        return torch.nan_to_num(mag, nan=0.0)
