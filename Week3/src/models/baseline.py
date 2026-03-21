import torch.nn as nn
from transformers import ResNetModel
import torch

chars = ['<SOS>', '<EOS>', '<PAD>', ' ', '!', '"', '#', '&', "'", '(', ')', ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '=', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

NUM_CHAR = len(chars)
idx2char = {k: v for k, v in enumerate(chars)}
char2idx = {v: k for k, v in enumerate(chars)}
TEXT_MAX_LEN = 201


class Encoder(nn.Module):
    def __init__(self, resnet_model: str = 'microsoft/resnet-18', device='cuda'):
        super().__init__()
        self.device = device
        self.resnet = ResNetModel.from_pretrained(resnet_model).to(self.device)

    def forward(self, img):
        feat = self.resnet(img)
        feat = feat.pooler_output.squeeze(-1).squeeze(-1).unsqueeze(0)
        return feat


class Decoder(nn.Module):
    def __init__(self, hidden_size=512, num_chars=NUM_CHAR, max_len=TEXT_MAX_LEN, device='cuda', rnn_type='GRU'):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.num_chars = num_chars
        self.rnn_type = rnn_type.upper()

        self.embed = nn.Embedding(num_chars, hidden_size)
        
        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(hidden_size, hidden_size, num_layers=1)
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers=1)
        else:
            raise ValueError(f"Unknown rnn_type: {rnn_type}")
        
        self.proj = nn.Linear(hidden_size, num_chars)

    def forward(self, visual_features, batch_size, target=None, teacher_forcing_ratio=0.0):
        """
        Args:
            visual_features: encoded image features from encoder
            batch_size: number of samples in batch
            target: ground truth tokens for teacher forcing (optional)
            teacher_forcing_ratio: probability of using ground truth instead of model output
        """
        start = torch.tensor(char2idx['<SOS>']).to(self.device)
        start_embed = self.embed(start)
        start_embeds = start_embed.repeat(batch_size, 1).unsqueeze(0)

        current_inp = start_embeds
        hidden = visual_features
        
        if self.rnn_type == 'LSTM':
            hidden = (hidden, torch.zeros_like(hidden))
        
        outputs = []

        for t in range(self.max_len - 1):
            if self.rnn_type == 'GRU':
                out, hidden = self.rnn(current_inp, hidden)
            else:  # LSTM
                out, hidden = self.rnn(current_inp, hidden)
            outputs.append(out)

            # Teacher forcing: use ground truth token with probability teacher_forcing_ratio
            if target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                current_inp = self.embed(target[:, t]).unsqueeze(0)
            else:
                current_inp = out

        # Concatenate the start token and all generated steps
        inp = torch.cat([start_embeds] + outputs, dim=0)

        res = inp.permute(1, 0, 2)
        res = self.proj(res)
        res = res.permute(0, 2, 1)
        return res


class Baseline(nn.Module):
    def __init__(self, device='cuda', resnet_model: str = 'microsoft/resnet-18', rnn_type: str = 'GRU'):
        super().__init__()
        self.device = device
        self.resnet = ResNetModel.from_pretrained(resnet_model).to(device)
        self.rnn_type = rnn_type.upper()
        self.embed = nn.Embedding(NUM_CHAR, 512)
        
        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(512, 512, num_layers=1)
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(512, 512, num_layers=1)
        else:
            raise ValueError(f"Unknown rnn_type: {rnn_type}")
        
        self.proj = nn.Linear(512, NUM_CHAR)

    def forward(self, img, target=None, teacher_forcing_ratio=0.0):
        batch_size = img.shape[0]
        feat = self.resnet(img)
        feat = feat.pooler_output.squeeze(-1).squeeze(-1).unsqueeze(0)
        start = torch.tensor(char2idx['<SOS>']).to(self.device)
        start_embed = self.embed(start)
        start_embeds = start_embed.repeat(batch_size, 1).unsqueeze(0)

        current_inp = start_embeds
        hidden = feat
        
        if self.rnn_type == 'LSTM':
            hidden = (hidden, torch.zeros_like(hidden))
        
        outputs = []

        for t in range(TEXT_MAX_LEN-1):
            if self.rnn_type == 'GRU':
                out, hidden = self.rnn(current_inp, hidden)
            else:  # LSTM
                out, hidden = self.rnn(current_inp, hidden)
            outputs.append(out)

            # Teacher forcing: use ground truth token with probability teacher_forcing_ratio
            if target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                current_inp = self.embed(target[:, t]).unsqueeze(0)
            else:
                current_inp = out

        # Concatenate the start token and all generated steps because thats what the teacher wants 
        inp = torch.cat([start_embeds] + outputs, dim=0)

        res = inp.permute(1, 0, 2)
        res = self.proj(res)
        res = res.permute(0, 2, 1)
        return res
