import torch.nn as nn
from transformers import ResNetModel
import torch

chars = ['<SOS>', '<EOS>', '<PAD>', ' ', '!', '"', '#', '&', "'", '(', ')', ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '=', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

NUM_CHAR = len(chars)
idx2char = {k: v for k, v in enumerate(chars)}
char2idx = {v: k for k, v in enumerate(chars)}
TEXT_MAX_LEN = 201


class Encoder(nn.Module):
    def __init__(self, resnet_model: str = 'microsoft/resnet-18', device='cpu'):
        super().__init__()
        self.device = device
        self.resnet = ResNetModel.from_pretrained(resnet_model).to(self.device)
        
    def forward(self, img):
        feat = self.resnet(img)
        feat = feat.pooler_output.squeeze(-1).squeeze(-1).unsqueeze(0)
        return feat


class Decoder(nn.Module):
    def __init__(self, hidden_size=512, num_chars=NUM_CHAR, max_len=TEXT_MAX_LEN, device='cpu'):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.num_chars = num_chars
        
        self.embed = nn.Embedding(num_chars, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=1)
        self.proj = nn.Linear(hidden_size, num_chars)
        
    def forward(self, visual_features, batch_size):
        start = torch.tensor(char2idx['<SOS>']).to(self.device)
        start_embed = self.embed(start)
        start_embeds = start_embed.repeat(batch_size, 1).unsqueeze(0)
        
        inp = start_embeds
        hidden = visual_features
        
        for t in range(self.max_len - 1):
            out, hidden = self.gru(inp, hidden)
            inp = torch.cat((inp, out[-1:]), dim=0)
        
        res = inp.permute(1, 0, 2)
        res = self.proj(res)
        res = res.permute(0, 2, 1)
        return res


class Model(nn.Module):
    def __init__(self, device='cpu', resnet_model: str = 'microsoft/resnet-18'):
        super().__init__()
        self.device = device
        self.encoder = Encoder(resnet_model=resnet_model, device=device)
        self.decoder = Decoder(device=device)
        
    def forward(self, img):
        batch_size = img.shape[0]
        visual_features = self.encoder(img)
        logits = self.decoder(visual_features, batch_size)
        return logits