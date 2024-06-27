from dataclasses import dataclass
import torch


def get_device_no_mps():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class ModelConfig:
    version_run: str         = "v13"   # name of the version that the model was trained with
    save_dir                 = "lightning_logs"
    name                     = "Melody"
    workers: int             = 20   # to run multiprocess, the number depends on the cpu number, Technion has 20
    batch_size: int          = 64
    block_size: int          = 512   # what is the maximum context length for predictions == History
    learning_rate: int       = 3e-4
    n_embd: int              = 1024   # the size of the vector the network will represent each single token
    n_head: int              = 8   # the number of heads: this is the self-attention mechanism
    n_blocks: int            = 16   # a block is the sequence of layer: [multi-heads->layer-norm->multi-heads->layer-norm->feed-forward->layer-norm]
    dropout: int             = 0.2
    n_freq_bins: int         = 664
    device: str              = torch.device('cuda' if torch.cuda.is_available() else 'cpu')






