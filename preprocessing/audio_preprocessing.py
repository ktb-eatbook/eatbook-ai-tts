import librosa
import numpy as np
import soundfile as sf

# 오디오 로드
def load_audio(path, sr=22050):
    audio, _ = librosa.load(path, sr=sr)
    return audio

# 정규화
def normalize_audio(audio):
    return audio / np.max(np.abs(audio))

# 멜 스펙트로그램 생성
def audio_to_mel_spectrogram(audio, sr=22050, n_mels=80):
    # 키워드 인자로 오디오, 샘플링 레이트, 멜 필터 수를 전달
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram

# 오디오 및 멜스펙트로그램 저장
def save_audio(audio, sr, path):
    sf.write(path, audio, sr)

def save_mel_spectrogram(mel_spectrogram, path):
    np.save(path, mel_spectrogram)