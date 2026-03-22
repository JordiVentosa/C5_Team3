import torch.nn as nn
from transformers import ResNetModel
import torch
from text_tokenizers import BaseTokenizer


class Baseline(nn.Module):
    def __init__(
        self,
        tokenizer: BaseTokenizer,
        device='cuda',
        resnet_model: str = 'microsoft/resnet-18',
        rnn_type: str = 'GRU',
        freeze_encoder: bool = False
    ):
        super().__init__()
        self.device = device
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.max_len = tokenizer.max_len
        self.special_tokens = tokenizer.get_special_token_indices()

        self.resnet = ResNetModel.from_pretrained(resnet_model).to(device)

        # Get ResNet output dimension from config
        self.hidden_dim = self.resnet.config.hidden_sizes[-1]

        # Freeze encoder if requested
        if freeze_encoder:
            for param in self.resnet.parameters():
                param.requires_grad = False
            print("✓ Encoder (ResNet) frozen - parameters will not be updated during training")

        self.rnn_type = rnn_type.upper()
        self.embed = nn.Embedding(self.vocab_size, self.hidden_dim)

        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(self.hidden_dim, self.hidden_dim, num_layers=1)
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=1)
        else:
            raise ValueError(f"Unknown rnn_type: {rnn_type}")

        self.proj = nn.Linear(self.hidden_dim, self.vocab_size)

    def forward(self, img, target=None, teacher_forcing_ratio=0.0):
        batch_size = img.shape[0]
        feat = self.resnet(img)
        feat = feat.pooler_output.squeeze(-1).squeeze(-1).unsqueeze(0)

        start = torch.tensor(self.special_tokens['sos']).to(self.device)
        start_embed = self.embed(start)
        start_embeds = start_embed.repeat(batch_size, 1).unsqueeze(0)

        current_inp = start_embeds
        hidden = feat

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

        # Concatenate the start token and all generated steps because thats what the teacher wants
        inp = torch.cat([start_embeds] + outputs, dim=0)

        res = inp.permute(1, 0, 2)
        res = self.proj(res)
        res = res.permute(0, 2, 1)
        return res

