import torch
import torch.nn as nn


class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.BCE_loss = nn.BCELoss()

    def forward(self, outputs, targets):
        mel_out, stop_token_prediction = outputs[0], outputs[1]
        mel_target, stop_token_target = targets[0], targets[1]

        stop_token_prediction = torch.sigmoid(stop_token_prediction)
        stop_token_target = torch.sigmoid(stop_token_target)

        mel_loss = self.mse_loss(mel_out, mel_target)

        stop_token_loss = self.BCE_loss(stop_token_prediction, stop_token_target)

        total_loss = mel_loss + stop_token_loss
        return total_loss

