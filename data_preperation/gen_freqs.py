import numpy as np
import torch
from melody_model.ConfigModel import ModelConfig
from melody_model.model import MelodyExtractionModel
import os
import pandas as pd
from preprocess.melodia import make_freqs
from preprocess.utility_functions import read_freqs, read_ground_truth

def extract_name(filename):
    parts = filename.split('_')
    if len(parts) > 2:
        return parts[0] + '_' + parts[1]
    else:
        parts = filename.split('.')
        return parts[0]


def write_names_of_files(folder_path, output_file_path):
    file_list = os.listdir(folder_path)
    file_list.sort()
    with open(output_file_path, 'w') as file:
        for filename in file_list:
            extracted_name = extract_name(filename)
            file.write(extracted_name + '\n')


def freqs_on_all_wavs():
    wav_folder = "/Users/tomermassas/Desktop/medlyDB/wavs"
    target_folder = "/Users/tomermassas/Desktop/medlyDB/freqs"
    song_names_path = f"/Users/tomermassas/Desktop/medlyDB/song_names.txt"

    song_names = []
    with open(song_names_path, 'r') as file:
        for line in file:
            line = line.strip()
            song_names.append(line)

    for name in song_names:
        make_freqs(f"{wav_folder}/{name}.wav", f"{target_folder}/{name}")

if __name__ == "__main__":
    conf = ModelConfig()
    pass
