import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


class AttentiveStatsPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x: (B, T, D)
        w = self.attention(x)          # (B, T, 1)
        w = torch.softmax(w, dim=1)

        mean = torch.sum(w * x, dim=1)
        std = torch.sqrt(
            torch.sum(w * (x - mean.unsqueeze(1)) ** 2, dim=1) + 1e-9
        )

        return torch.cat([mean, std], dim=1)  # (B, 2D)


class AudioFakeDetector(nn.Module):
    def __init__(self):
        super().__init__()

        self.wav2vec = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base"
        )

        self.pooling = AttentiveStatsPooling(768)

        self.classifier = nn.Sequential(
            nn.LayerNorm(1536),
            nn.Linear(1536, 512),
            nn.GELU(),
            nn.Dropout(0.5),

            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(128, 2)  # real / fake
        )

    def forward(self, input_values):
        outputs = self.wav2vec(input_values)

        hidden_states = outputs.last_hidden_state  # (B, T, 768)

        pooled = self.pooling(hidden_states)

        return self.classifier(pooled)
