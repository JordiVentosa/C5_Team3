import torch.nn as nn
from transformers import ResNetModel
import torch
from text_tokenizers import BaseTokenizer


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim, attention_dim):
        super().__init__()
        self.hidden_proj = nn.Linear(hidden_dim, attention_dim)
        self.feature_proj = nn.Linear(hidden_dim, attention_dim)
        self.score_proj = nn.Linear(attention_dim, 1)

    def forward(self, features, hidden):
        # features: (batch, num_pixels, hidden_dim)
        # hidden: (1, batch, hidden_dim) for GRU or tuple for LSTM

        # Extract hidden state from LSTM tuple if needed
        if isinstance(hidden, tuple):
            hidden = hidden[0]

        # hidden: (1, batch, hidden_dim) -> (batch, hidden_dim)
        hidden = hidden.squeeze(0)

        # Project features and hidden state to attention dimension
        proj_features = self.feature_proj(features)  # (batch, num_pixels, attention_dim)
        proj_hidden = self.hidden_proj(hidden).unsqueeze(1)  # (batch, 1, attention_dim)

        # Add them together and apply tanh
        combined = torch.tanh(proj_features + proj_hidden)  # (batch, num_pixels, attention_dim)

        # Project to scalar scores
        scores = self.score_proj(combined).squeeze(-1)  # (batch, num_pixels)

        # Apply softmax to get attention weights
        attention_weights = torch.softmax(scores, dim=1)  # (batch, num_pixels)

        # Compute context vector as weighted sum
        context = torch.bmm(attention_weights.unsqueeze(1), features).squeeze(1)  # (batch, hidden_dim)

        return context, attention_weights


class Baseline(nn.Module):
    def __init__(
        self,
        tokenizer: BaseTokenizer,
        device='cuda',
        resnet_model: str = 'microsoft/resnet-18',
        rnn_type: str = 'GRU',
        freeze_encoder: bool = False,
        attention: bool = False
    ):
        super().__init__()
        self.device = device
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.max_len = tokenizer.max_len
        self.special_tokens = tokenizer.get_special_token_indices()
        self.use_attention = attention

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

        # Initialize attention module if enabled
        if self.use_attention:
            self.attention = BahdanauAttention(self.hidden_dim, attention_dim=256)
            # RNN input is embedding + context vector
            rnn_input_dim = self.hidden_dim * 2
            print("✓ Bahdanau Attention enabled")
        else:
            rnn_input_dim = self.hidden_dim

        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(rnn_input_dim, self.hidden_dim, num_layers=1)
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(rnn_input_dim, self.hidden_dim, num_layers=1)
        else:
            raise ValueError(f"Unknown rnn_type: {rnn_type}")

        self.proj = nn.Linear(self.hidden_dim, self.vocab_size)

    def forward(self, img, target=None, teacher_forcing_ratio=0.0):
        batch_size = img.shape[0]
        feat = self.resnet(img)

        if self.use_attention:
            # Use spatial features for attention
            spatial_features = feat.last_hidden_state  # (batch, C, H, W)
            # Reshape to (batch, num_pixels, hidden_dim)
            b, c, h, w = spatial_features.shape
            spatial_features = spatial_features.view(b, c, h * w).permute(0, 2, 1)  # (batch, H*W, C)

            # Initialize hidden state with mean of spatial features
            init_hidden = spatial_features.mean(dim=1).unsqueeze(0)  # (1, batch, hidden_dim)
        else:
            # Use pooled features (original behavior)
            feat = feat.pooler_output.squeeze(-1).squeeze(-1).unsqueeze(0)
            init_hidden = feat

        start = torch.tensor(self.special_tokens['sos']).to(self.device)
        start_embed = self.embed(start)
        start_embeds = start_embed.repeat(batch_size, 1).unsqueeze(0)

        current_inp = start_embeds
        hidden = init_hidden

        if self.rnn_type == 'LSTM':
            hidden = (hidden, torch.zeros_like(hidden))

        outputs = []

        for t in range(self.max_len - 1):
            if self.use_attention:
                # Compute attention and get context vector
                context, _ = self.attention(spatial_features, hidden)  # (batch, hidden_dim)
                # Concatenate embedding with context
                rnn_input = torch.cat([current_inp.squeeze(0), context], dim=1).unsqueeze(0)
            else:
                rnn_input = current_inp

            if self.rnn_type == 'GRU':
                out, hidden = self.rnn(rnn_input, hidden)
            else:  # LSTM
                out, hidden = self.rnn(rnn_input, hidden)
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

