import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearNorm(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias)
        nn.init.xavier_uniform_(
            self.linear_layer.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)

class ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        # 입력 텐서를 float32로 변환하여 Conv1d에 전달
        x = x.to(torch.float32)
        return self.conv(x)

class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size) for in_size, out_size in zip(in_sizes, sizes)]
        )

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=self.training)
        return x

class Encoder(nn.Module):
    def __init__(self, input_channels=32, encoder_embedding_dim=512, conv_layers=3, kernel_size=5):
        super(Encoder, self).__init__()
        convolutions = []
        for i in range(conv_layers):
            conv_layer = nn.Sequential(
                ConvNorm(input_channels, encoder_embedding_dim,
                         kernel_size=kernel_size, stride=1, padding=2),
                nn.BatchNorm1d(encoder_embedding_dim)
            )
            convolutions.append(conv_layer)
            input_channels = encoder_embedding_dim
        self.convolutions = nn.ModuleList(convolutions)
        self.lstm = nn.LSTM(encoder_embedding_dim, encoder_embedding_dim // 2, 1,
                            batch_first=True, bidirectional=True)

    def forward(self, x):
        x = x.float()
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)
        x = x.transpose(1, 2)
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        return outputs


class Decoder(nn.Module):
    def __init__(self, decoder_rnn_dim, attention_rnn_dim, attention_dim, prenet_dim, n_mel_channels,
                 max_decoder_steps=1000, gate_threshold=0.5, p_attention_dropout=0.1, p_decoder_dropout=0.1):
        super(Decoder, self).__init__()
        self.decoder_rnn_dim = decoder_rnn_dim
        self.attention_rnn_dim = attention_rnn_dim
        self.attention_dim = attention_dim
        self.prenet_dim = prenet_dim
        self.n_mel_channels = n_mel_channels
        self.max_decoder_steps = max_decoder_steps
        self.gate_threshold = gate_threshold
        self.p_attention_dropout = p_attention_dropout
        self.p_decoder_dropout = p_decoder_dropout

    def forward(self, encoder_outputs, mel_specs, text_lengths):

        encoder_outputs = encoder_outputs.float()
        mel_specs = mel_specs.float()

        mel_outputs = torch.randn(
            (encoder_outputs.size(0), self.n_mel_channels, mel_specs.size(2)))
        gate_outputs = torch.zeros((encoder_outputs.size(0), mel_specs.size(2)))
        alignments = torch.zeros(
            (encoder_outputs.size(0), int(text_lengths.max().item()), mel_specs.size(2)))

        return mel_outputs, gate_outputs, alignments

class Postnet(nn.Module):
    def __init__(self, n_mel_channels, postnet_embedding_dim, kernel_size=5, n_convolutions=5):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(n_mel_channels, postnet_embedding_dim,
                         kernel_size=kernel_size, stride=1, padding=2, w_init_gain='tanh'),
                nn.BatchNorm1d(postnet_embedding_dim))
        )

        for _ in range(1, n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(postnet_embedding_dim, postnet_embedding_dim,
                             kernel_size=kernel_size, stride=1, padding=2, w_init_gain='tanh'),
                    nn.BatchNorm1d(postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(postnet_embedding_dim, n_mel_channels,
                         kernel_size=kernel_size, stride=1, padding=2, w_init_gain='linear'),
                nn.BatchNorm1d(n_mel_channels))
        )

    def forward(self, x):
        x = x.float()
        for i, conv in enumerate(self.convolutions):
            if i < len(self.convolutions) - 1:
                x = F.dropout(torch.tanh(conv(x)), 0.5, self.training)
            else:
                x = F.dropout(conv(x), 0.5, self.training)
        return x

