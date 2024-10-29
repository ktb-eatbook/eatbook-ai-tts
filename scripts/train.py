import torch
import sys
import os
from tqdm import tqdm  # tqdm 임포트 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)

import json
from torch.utils.data import DataLoader
from model.tacotron2 import Tacotron2
from utils.data_loader import TextAudioDataset, get_data_loader
from utils.loss import Tacotron2Loss

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            param.requires_grad = True

    # tqdm으로 배치 진행 상황 시각화
    for batch in tqdm(train_loader, desc="Training Batches", unit="batch"):
        optimizer.zero_grad()

        # 모든 입력을 float32로 변환
        inputs = {key: value.to(device).float() for key, value in batch.items()}
        outputs, mel_outputs_postnet, gate_outputs, alignments = model(inputs)

        # 손실 계산
        loss = criterion(outputs, inputs['audio'])
        outputs.requires_grad_(True)
        loss.requires_grad_(True)

        # 역전파 수행
        loss.backward()
        optimizer.step()

        # 배치 손실 누적
        total_loss += loss.item()

    return total_loss / len(train_loader)


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, '..', 'configs', 'tacotron2_config.json')

    with open(config_path, 'r') as f:
        config = json.load(f)
    device = torch.device("cpu")  # 쿠다 제발..."mps" 사용 시 torch.device("mps")

    model = Tacotron2().to(device)
    train_loader = get_data_loader(
        metadata_file=config['paths']['data_path'] + '/matched_metadata.csv',
        batch_size=config['training']['batch_size'],
        shuffle=True
    )

    criterion = Tacotron2Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    print("Training info\n","Epoch :", config["training"]["epochs"], "Batch size :", config["training"]["batch_size"], "Learning rate :", config["training"]["learning_rate"])
    # tqdm으로 에포크 진행 상황 시각화
    for epoch in tqdm(range(config["training"]["epochs"]), desc="Epochs", unit="epoch"):
        loss = train(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch + 1}/{config['training']['epochs']}, Loss: {loss:.4f}")

        # 체크포인트 저장
        if (epoch + 1) % config["training"]["save_interval"] == 0:
            torch.save(model.state_dict(), f"checkpoints/checkpoint_{epoch + 1}.pth")


if __name__ == "__main__":
    main()