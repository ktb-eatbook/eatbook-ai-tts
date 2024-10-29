import pandas as pd

# 텍스트-오디오 쌍 매칭
def match_text_audio(text_metadata, audio_metadata):
    matched_df = pd.merge(text_metadata, audio_metadata, on='wav_filename')
    return matched_df

# 데이터셋 준비
def prepare_dataset(text_path, audio_path, output_file):
    text_metadata = pd.read_csv(text_path)
    audio_metadata = pd.read_csv(audio_path)
    matched_df = match_text_audio(text_metadata, audio_metadata)
    matched_df.to_csv(output_file, index=False, encoding='utf-8')