import torch
from model import MelodyExtractionModel
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch.multiprocessing as mp
from DataLoader import ModelConfig
import os
import csv
from data_preperation.quantisize_freqs import infer_one_song
from data_preperation.quantisize_freqs import idx_list_to_freqs
from preprocess.utility_functions import HOP_LENGTH
from preprocess.utility_functions import make_from_csv




if __name__ == "__main__":
    # Setting the device and precision for the model
    torch.set_float32_matmul_precision('medium')

    # |---------- Inference ----------|
    conf = ModelConfig()
    model = MelodyExtractionModel.load_from_checkpoint(f"../../Val_Loss_0324/VAL_LOSS03244_epoch=0-step=76087.ckpt")
    wav_file_path = '../../sirduke_short.wav'
    csv_file_path = '../../sirduke_short.csv'
    vec_input , timestamps = infer_one_song(wav_file_path)
    numpy_vec_input = torch.from_numpy(vec_input).float().to(conf.device)
    idx_infer_results = model.generate_melody(numpy_vec_input).tolist()
    res_freqs = idx_list_to_freqs(idx_infer_results,semitone_fraction=10)
    rows = zip(timestamps, res_freqs)

    with open(csv_file_path, 'w',newline='') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

    make_from_csv(path=csv_file_path,path_to_save=csv_file_path + '/..',name='sirduke_short_RESULT')