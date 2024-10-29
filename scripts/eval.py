import os
import torch
import json
from model.tacotron2 import Tacotron2
from utils.data_loader import TextAudioDataset
from torch.utils.data import DataLoader
from utils.metrics import calculate_mel_spectrogram_error
from utils.loss import Tacotron2Loss
#from utils.audio import load_audio


def evaluate(model, eval_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in eval_loader:
            inputs, targets = batch['text'], batch['audio']
            inputs, targets = inputs.to(device), targets.to(device)

            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # 추가 평가 메트릭
            mel_spectrogram_error = calculate_mel_spectrogram_error(outputs, targets)

    avg_loss = total_loss / len(eval_loader)
    avg_mel_error = mel_spectrogram_error / len(eval_loader)
    return avg_loss, avg_mel_error


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, '..', 'configs', 'tacotron2_config.json')

    with open(config_path, 'r') as f:
        config = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Tacotron2().to(device)
    model.load_state_dict(torch.load(config["eval_checkpoint"]))

    eval_dataset = TextAudioDataset(config["eval_data_path"])
    eval_loader = DataLoader(eval_dataset, batch_size=config["batch_size"], shuffle=False)

    criterion = Tacotron2Loss()
    loss, mel_error = evaluate(model, eval_loader, criterion, device)

    print(f"Evaluation Loss: {loss}, Mel Spectrogram Error: {mel_error}")


if __name__ == "__main__":
    main()