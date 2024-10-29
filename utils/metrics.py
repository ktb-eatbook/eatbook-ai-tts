import torch

def calculate_mel_spectrogram_error(output, target):
    # Mel spectrogram 오차 계산
    return torch.mean((output - target) ** 2).item()