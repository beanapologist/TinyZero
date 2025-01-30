import torch
import torch.nn as nn
import math
from dataclasses import dataclass

@dataclass
class QDTConfig:
    d_model: int = 768
    n_layer: int = 12
    n_head: int = 12
    d_head: int = 64
    d_ff: int = 3072
    vocab_size: int = 32000
    max_seq_len: int = 512
    dropout: float = 0.1
    # QDT Constants
    lambda_start: float = 0.867
    lambda_target: float = 0.500
    gamma: float = 0.4497
    beta: float = 0.310
    eta: float = 0.520

class QDTAttention(nn.Module):
    def __init__(self, config: QDTConfig):
        super().__init__()
        self.config = config
        self.scale = 1.0 / math.sqrt(config.d_head)
        self.q_proj = nn.Linear(config.d_model, config.n_head * config.d_head)
        self.k_proj = nn.Linear(config.d_model, config.n_head * config.d_head)
        self.v_proj = nn.Linear(config.d_model, config.n_head * config.d_head)
        self.o_proj = nn.Linear(config.n_head * config.d_head, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.config.n_head, -1).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.config.n_head, -1).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.config.n_head, -1).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(out)

class QDTFeedForward(nn.Module):
    def __init__(self, config: QDTConfig):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff)
        self.w2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.w2(self.dropout(torch.relu(self.w1(x))))

class QDTBlock(nn.Module):
    def __init__(self, config: QDTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.attn = QDTAttention(config)
        self.ff = QDTFeedForward(config)
        self.config = config

    def forward(self, x, mask=None):
        # Apply QDT coupling
        lambda_factor = self.config.lambda_start * math.exp(-self.config.gamma * 0.1)
        h = x + lambda_factor * self.attn(self.ln1(x), mask)
        out = h + self.config.beta * self.ff(self.ln2(h))
        return out

class QDTModel(nn.Module):
    def __init__(self, config: QDTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.max_seq_len, config.d_model))
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([QDTBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize parameters with QDT-based scaling
        self.apply(self._qdt_init_weights)

    def _qdt_init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            scale = math.sqrt(self.config.lambda_start / module.weight.shape[0])
            module.weight.data.normal_(mean=0.0, std=scale)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(self, idx, mask=None):
        batch_size, seq_len = idx.shape
        
        # Token + Position embeddings
        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb[:, :seq_len, :]
        x = self.dropout(tok_emb + pos_emb)

        # Apply QDT transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x)
        logits = self.head(x)

        return logits

def create_qdt_model():
    config = QDTConfig()
    model = QDTModel(config)
    return model, config

# Example usage
if __name__ == "__main__":
    model, config = create_qdt_model()
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test forward pass
    x = torch.randint(0, config.vocab_size, (2, 32))
    logits = model(x)
    print(f"Output shape: {logits.shape}")
