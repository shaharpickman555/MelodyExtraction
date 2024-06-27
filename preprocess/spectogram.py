import librosa
import numpy as np
import matplotlib.pyplot as plt
from utility_functions import *

path_to_song = "/Users/tomermassas/Desktop/project melody extraction/datasets/medleyDB/AClassicEducation_NightOwl/AClassicEducation_NightOwl_RAW/AClassicEducation_NightOwl_RAW_01_01.wav"

signal, sr = librosa.load(path_to_song)


# display_signal(signal,sr)
play_signal(signal,sr)

n_fft = 2048  # this is the number of samples in a window per fft
hop_length = 512  # The amount of samples we are shifting after each fft

mel_signal = librosa.feature.melspectrogram(y=signal, sr=sr, hop_length=hop_length, n_fft=n_fft)
spectrogram = np.abs(mel_signal)
mel_spec_db = librosa.power_to_db(spectrogram, ref=np.max)


show_mel_spec(mel_spec_db, sr, hop_length)

print()
