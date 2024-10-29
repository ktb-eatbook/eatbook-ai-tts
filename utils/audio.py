import numpy as np
import librosa
import soundfile as sf

def load_audio(filepath, sr=22050):
    """
    오디오 파일을 로드하여 지정된 샘플링 레이트로 반환합니다.
    """
    audio, sample_rate = librosa.load(filepath, sr=sr)
    return audio

def normalize_audio(audio):
    """
    오디오 신호를 -1과 1 사이로 정규화합니다.
    """
    return audio / np.abs(audio).max()

def audio_to_mel_spectrogram(audio, sr=22050, n_fft=1024, hop_length=256, n_mels=80):
    """
    오디오 신호를 멜 스펙트로그램으로 변환합니다.
    """
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=1.0
    )
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram_db

def save_audio(audio, sr, filepath):
    """
    오디오 신호를 파일로 저장합니다.
    """
    sf.write(filepath, audio, sr)

def save_mel_spectrogram(mel_spectrogram, filepath):
    """
    멜 스펙트로그램을 Numpy 파일로 저장합니다.
    """
    np.save(filepath, mel_spectrogram)