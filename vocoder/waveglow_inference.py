import torch
import numpy as np
import soundfile as sf

# WaveGlow 모델 로드
def load_waveglow_model(model_path):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    waveglow = torch.load(model_path, map_location=device)['model']
    waveglow = waveglow.remove_weightnorm(waveglow)  # weight normalization 제거
    waveglow.eval()
    return waveglow.to(device)

# Mel-spectrogram을 오디오로 변환
def mel_to_audio(mel, waveglow, sigma=0.6):
    device = waveglow.device
    mel = torch.tensor(mel, dtype=torch.float32).to(device)

    # Mel-spectrogram 차원 확인
    if len(mel.shape) == 2:
        mel = mel.unsqueeze(0)

    with torch.no_grad():
        audio = waveglow.infer(mel, sigma=sigma)
    return audio.squeeze().cpu().numpy()

# 생성된 오디오 저장
def save_audio(audio, sample_rate, output_path):
    audio = audio / np.max(np.abs(audio))  # 오디오 정규화 (-1 ~ 1 범위)
    sf.write(output_path, audio, sample_rate)