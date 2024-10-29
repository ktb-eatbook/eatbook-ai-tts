import os
import sys
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 각 전처리 모듈에서 함수 호출
from preprocessing.text_preprocessing import text_to_sequence
from preprocessing.audio_preprocessing import load_audio, normalize_audio, audio_to_mel_spectrogram, save_audio, save_mel_spectrogram
from preprocessing.dataset_preparation import prepare_dataset

# 기본 경로 설정
text_metadata_path = 'data/kss/metadata.txt'
audio_save_path = 'data/processed/audio/'
mel_spectrogram_save_path = 'data/processed/mel_spectrograms/'
output_dataset_path = 'data/processed/matched_metadata.csv'

# 메타데이터 파일 경로
text_csv_path = 'data/processed/text_metadata.csv'
audio_csv_path = 'data/processed/audio_metadata.csv'

# 설정된 경로가 존재하는지 확인 후 생성
os.makedirs(audio_save_path, exist_ok=True)
os.makedirs(mel_spectrogram_save_path, exist_ok=True)

# 메타데이터 로드 및 전처리 함수
def make_metadata(input_file):
    metadata = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            # 파이프로 구분된 데이터를 분리
            fields = line.strip().split('|')
            if len(fields) >= 3:
                # 첫 번째 필드는 파일 이름
                wav_filename = fields[0].strip()
                processed_text = fields[2].strip()  # 확장된 텍스트
                metadata.append([wav_filename, processed_text])

    # 파일 이름과 텍스트 데이터를 DataFrame으로 변환
    df = pd.DataFrame(metadata, columns=['wav_filename', 'processed_text'])
    return df

# 텍스트 전처리 함수 호출 및 시퀀스 생성
def process_text(input_file):
    # 메타데이터에서 텍스트 전처리 및 시퀀스 변환
    text_metadata = make_metadata(input_file)
    text_metadata['text_sequence'] = text_metadata['processed_text'].apply(lambda x: text_to_sequence(x))
    
    # 전처리된 텍스트 저장
    text_metadata[['wav_filename', 'text_sequence']].to_csv(text_csv_path, index=False)
    return text_metadata

# 오디오 전처리 함수 호출
def process_audio(text_metadata):
    audio_metadata = []

    for wav_filename in text_metadata['wav_filename']:
        # 확장자가 누락된 경우를 대비해 파일 이름에 .wav 추가
        if not wav_filename.endswith(".wav"):
            wav_filename += ".wav"
        
        audio_path = os.path.join('data/kss', wav_filename)  # 올바른 파일 경로로 설정
        mel_spectrogram_file = os.path.join(mel_spectrogram_save_path, f"{wav_filename.split('.')[0]}.npy")
        audio_save_file = os.path.join(audio_save_path, wav_filename)

        # 디렉토리 생성 (오디오 저장 경로가 없을 경우 생성)
        os.makedirs(os.path.dirname(audio_save_file), exist_ok=True)
        os.makedirs(os.path.dirname(mel_spectrogram_file), exist_ok=True)  # 멜 스펙트로그램 경로에 대한 디렉토리 생성
        
        # 오디오 로드 및 전처리
        audio = load_audio(audio_path)
        if audio is None:
            print(f"Error loading {wav_filename}")
            continue
        
        audio = normalize_audio(audio)
        mel_spectrogram = audio_to_mel_spectrogram(audio)

        # 전처리된 오디오 및 멜 스펙트로그램 저장
        save_audio(audio, 22050, audio_save_file)
        save_mel_spectrogram(mel_spectrogram, mel_spectrogram_file)

        # 오디오 메타데이터에 추가
        audio_metadata.append([wav_filename, mel_spectrogram_file])

    # 오디오 메타데이터 저장
    audio_metadata_df = pd.DataFrame(audio_metadata, columns=['wav_filename', 'mel_spectrogram_path'])
    audio_metadata_df.to_csv(audio_csv_path, index=False)
    return audio_metadata_df

# 텍스트와 오디오를 매칭한 최종 데이터셋 생성
def process_dataset():
    prepare_dataset(text_csv_path, audio_csv_path, output_dataset_path)

if __name__ == '__main__':
    # 텍스트 처리
    text_metadata = process_text(text_metadata_path)

    # 오디오 처리
    audio_metadata = process_audio(text_metadata)

    # 데이터셋 생성
    process_dataset()

    print("Preprocessing complete!")