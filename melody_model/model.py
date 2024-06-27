import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from melody_model.DataLoader import DatasetMelody
from melody_model.ConfigModel import ModelConfig

torch.nn.Transformer()
# HYPERPARAMETERS
conf = ModelConfig


class Head(nn.Module):
    """ this is a single head as part of the self attention methodology"""
    def __init__(self, head_size, n_embd, dropout, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False) # what do I contain
        self.query = nn.Linear(n_embd, head_size, bias=False) # what am I looking for
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # register_buffer is PyTorch way to say "tril" is not a parameter to optimize
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        # we normalize by "C ** -0.5" to avoid high values in the softmax leads to "one-hot-vector"
        # when key and query are more aligned they produce a high affinity, and we get more information
        wei = q @ k.transpose(-2, -1) * C ** -0.5  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        # the masking helps gather information by increasing history look back up from 1 to T tokens
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, n_embd, dropout, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, dropout,block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, dropout, block_size):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head #this gives us the same number of params but split to many heads
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, dropout, block_size)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class MelodyExtractionModel(pl.LightningModule):
    "Final transformer based model for extraction the final frequencies"
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Linear(conf.n_freq_bins, conf.n_embd)
        self.position_embedding_table = nn.Embedding(conf.block_size, conf.n_embd)
        self.blocks = nn.TransformerEncoder(nn.TransformerEncoderLayer(conf.n_embd, conf.n_head, dim_feedforward=4 * conf.n_embd, dropout=conf.dropout, activation='relu', norm_first=True, batch_first=True), conf.n_blocks)
        self.ln_f = nn.LayerNorm(conf.n_embd) # final layer norm
        self.lm_head = nn.Linear(conf.n_embd, conf.n_freq_bins)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # @torch.autocast(device_type="cuda")
    def forward(self, idx):
        B, T, C = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=conf.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C) we add the positions of Time*Channels to each sample in Batch
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = y.view(B * T).type(torch.LongTensor).to(conf.device)
        # if (torch.any(targets>=266) or torch.any(targets<0)):
        #     raise Exception(f"ERROR in DATATYPE , target is {torch.max(targets)} or {torch.min(targets)}")

        train_loss = F.cross_entropy(logits, targets)
        self.log("train_loss", train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = y.view(B * T).type(torch.LongTensor).to(conf.device)
        val_loss = F.cross_entropy(logits, targets)
        self.log("val_loss", val_loss, prog_bar=True, logger=True)
        return val_loss

    def train_dataloader(self):
        return DataLoader(DatasetMelody(run_type="train", device=conf.device), batch_size=conf.batch_size, shuffle=True, pin_memory=True, num_workers=conf.workers,persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(DatasetMelody(run_type="val", device=conf.device), batch_size=conf.batch_size, shuffle=False, pin_memory=True, num_workers=conf.workers,persistent_workers=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=conf.learning_rate)

    def version(self):
        return conf.version_run

    def generate_melody(self, context):  # context is (T,C)
        align_size = (context.shape[0] + conf.block_size - 1) // conf.block_size * conf.block_size
        input = F.pad(context, (0, 0,0, align_size - context.shape[0]))
        results = torch.empty((align_size), dtype=torch.long)

        for idx in range(0, input.shape[0], conf.block_size):
            input_block = input[idx:idx + conf.block_size, :].unsqueeze(0)
            logits = self(input_block).squeeze(0) # (T,C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (T, C)
            # sample from the distribution
            _, idx_next = torch.max(probs, dim=-1)  # (T)
            results[idx:idx + conf.block_size] = idx_next

        results = results[:context.shape[0]]
        return results
