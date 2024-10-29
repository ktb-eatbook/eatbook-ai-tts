import torch
import torch.nn as nn
import json
from model.layers import Encoder, Decoder, Postnet

# JSON 파일에서 설정 불러오기
with open("configs/tacotron2_config.json") as f:
    config = json.load(f)


class Tacotron2(nn.Module):
    def __init__(self):
        super(Tacotron2, self).__init__()

        # JSON 파일에서 모델 관련 파라미터 가져오기
        encoder_embedding_dim = config["model"]["encoder_embedding_dim"]
        decoder_rnn_dim = config["model"]["decoder_rnn_dim"]
        attention_rnn_dim = config["model"]["attention_rnn_dim"]
        attention_dim = config["model"]["attention_dim"]
        prenet_dim = config["model"]["prenet_dim"]
        postnet_embedding_dim = config["model"]["postnet_embedding_dim"]
        n_mel_channels = config["model"]["n_mel_channels"]

        # Tacotron2 모델 구성 요소 정의
        self.channel_expander = nn.Linear(32, encoder_embedding_dim)
        self.encoder = Encoder(encoder_embedding_dim,
                               config["model"]["encoder_conv_layers"],
                               config["model"]["encoder_kernel_size"])
        self.decoder = Decoder(
            decoder_rnn_dim=decoder_rnn_dim,
            attention_rnn_dim=attention_rnn_dim,
            attention_dim=attention_dim,
            prenet_dim=prenet_dim,
            n_mel_channels=n_mel_channels
        )
        self.postnet = Postnet(n_mel_channels, postnet_embedding_dim)

    def forward(self, inputs):
        # Encoder
        text_inputs = inputs['text'].to(torch.float32)
        text_lengths = inputs['text_lengths']
        mel_specs = inputs['audio']
        max_len = inputs['audio_lengths']

        # 텐서의 마지막 차원을 맞추기 위해 unsqueeze 사용
        text_inputs = text_inputs.unsqueeze(-1)  # (batch_size, sequence_length, 1)로 변환
        text_inputs = text_inputs.expand(-1, -1, 32)  # 필요한 크기(예: 32)로 확장

        # 채널 확장
        text_inputs = self.channel_expander(text_inputs)
        text_inputs = text_inputs.transpose(1, 2)  # (batch_size, encoder_embedding_dim, sequence_length)

        embedded_inputs = self.encoder(text_inputs)

        # Decoder
        mel_outputs, gate_outputs, alignments = self.decoder(
            embedded_inputs, mel_specs, text_lengths)

        # Postnet
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return mel_outputs, mel_outputs_postnet, gate_outputs, alignments

    def inference(self, text_inputs):
        # 인퍼런스를 위한 간단한 함수 예시

        # 채널 확장
        text_inputs = self.channel_expander(text_inputs)
        text_inputs = text_inputs.transpose(1, 2)

        embedded_inputs = self.encoder(text_inputs)
        mel_outputs, gate_outputs, alignments = self.decoder.inference(embedded_inputs)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return mel_outputs_postnet