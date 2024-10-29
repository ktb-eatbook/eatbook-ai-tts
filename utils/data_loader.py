import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class TextAudioDataset(Dataset):
    def __init__(self, metadata_file, text_column='text_sequence', audio_column='mel_spectrogram_path'):
        # 메타데이터 로드
        self.metadata = pd.read_csv(metadata_file)
        self.text_column = text_column
        self.audio_column = audio_column

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # 텍스트 시퀀스 로드
        text_sequence = eval(self.metadata.iloc[idx][self.text_column])  # 문자열로 저장된 리스트를 리스트로 변환
        text_tensor = torch.tensor(text_sequence, dtype=torch.long)

        # 멜 스펙트로그램 로드
        mel_spectrogram_path = self.metadata.iloc[idx][self.audio_column]
        mel_spectrogram = np.load(mel_spectrogram_path)
        mel_tensor = torch.tensor(mel_spectrogram, dtype=torch.float32)

        return {'text': text_tensor, 'audio': mel_tensor}

def get_data_loader(metadata_file, batch_size, shuffle=True, num_workers=2):
    dataset = TextAudioDataset(metadata_file)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    return data_loader

def collate_fn(batch):
    # 패딩 처리: 텍스트와 멜 스펙트로그램을 동적으로 패딩하여 배치 생성
    text_lengths = [len(item['text']) for item in batch]
    max_text_len = max(text_lengths)
    audio_lengths = [item['audio'].shape[1] for item in batch]  # 멜 스펙트로그램의 길이
    max_audio_len = max(audio_lengths)

    text_padded = torch.zeros(len(batch), max_text_len, dtype=torch.long)
    audio_padded = torch.zeros(len(batch), batch[0]['audio'].shape[0], max_audio_len)

    for i, item in enumerate(batch):
        text = item['text']
        text_padded[i, :len(text)] = text

        audio = item['audio']
        audio_padded[i, :, :audio.shape[1]] = audio

    return {'text': text_padded, 'audio': audio_padded, 'text_lengths': torch.tensor(text_lengths), 'audio_lengths': torch.tensor(audio_lengths)}