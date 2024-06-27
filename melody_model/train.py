import torch
from model import MelodyExtractionModel
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch.multiprocessing as mp
from ConfigModel import ModelConfig, get_device_no_mps
import os


def save_config():
    conf = ModelConfig()
    file_path = f"./{conf.save_dir}/{conf.name}/{conf.version_run}/checkpoints"
    if not os.path.exists(f'{file_path}'):
        os.makedirs(f'{file_path}')
    file_path = f"{file_path}/Config_Model.txt"
    with open(file_path, 'w') as file:
        for field_name, field_value in conf.__dict__.items():
            file.write(f'{field_name} = {field_value}\n')

def print_layers_info(model):
    for name, param in model.named_parameters():
        print(f"\n\n|       {name}        |")
        mean = param.data.mean().item()
        std = param.data.std().item()
        shape = tuple(param.data.shape)
        print(f"| mean={mean:.4f}   |   std={std:.4f} |   shape={shape} |\n\n")


if __name__ == "__main__":
    # Setting the device and precision for the model
    torch.set_float32_matmul_precision('medium')


    # |---------- Init model ----------|
    # model = MelodyExtractionModel.load_from_checkpoint(f"lightning_logs/Melody/v8_log/checkpoints/epoch=0-step=303537.ckpt")
    model = MelodyExtractionModel()
    model.to(device=get_device_no_mps())

    # |---------- Train ----------|
    conf = ModelConfig()
    logger = TensorBoardLogger(save_dir=conf.save_dir, name=conf.name, version=model.version())
    save_config()
    trainer = pl.Trainer(accelerator='auto',
                         logger=logger,
                         max_epochs=20,
                         val_check_interval=0.1,
                         accumulate_grad_batches=1)
    trainer.fit(model)