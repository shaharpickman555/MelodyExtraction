import numpy as np
import pandas as pd
from scipy.io import wavfile
from preprocess.extract_freqs_amps import process_freqs_amps_extraction
from preprocess.utility_functions import (make_audio, save_lines, read_ground_truth, read_freqs, plot_histogram,
                                          match_justing_freqs_to_ground_truth, analize_justin_with_ground_truth,
                                          make_from_many, make_from_csv, song_names_txt_to_list)
from quantisize_freqs import freqs_bin_edges, make_input_labels_timestretch, make_input_labels_pitch, make_input_labels_normal, make_input_labels_timestretch_pitch
from audiomentations import Compose, AddGaussianNoise,TimeStretch, PitchShift, Gain
import librosa
import soundfile as sf
import librosa.display
import matplotlib.pyplot as plt
from typing import List

####################################################################################
# PITCH increase/decrease
####################################################################################
def add_pitch_to_input(arr, shift):
    if shift < 0:
        # Shift to the left
        shift = -shift
        shifted = np.concatenate([arr[:, shift:], np.zeros((arr.shape[0], shift))], axis=1)
    elif shift > 0:
        # Shift to the right
        shifted = np.concatenate([np.zeros((arr.shape[0], shift)), arr[:, :-shift]], axis=1)
    return shifted

def add_pitch_to_labels(arr, value_to_add):
    non_zero_mask = arr != 0
    arr[non_zero_mask] += value_to_add
    return arr
####################################################################################


def write_names_to_file(names, file_path):
    with open(file_path, 'w') as file:
        for name in names:
            file.write(name + '\n')

def _plot_signal_and_augmented_signal(signal, augmented_signal, sr):
    fig, ax = plt.subplots(nrows=2)
    librosa.display.waveshow(signal, sr=sr, ax=ax[0])
    ax[0].set(title="Original signal")
    librosa.display.waveshow(augmented_signal, sr=sr, ax=ax[1])
    ax[1].set(title="Augmented signal")
    plt.show()


# # Time stretch
# rate = 0.94 #,1.06, 0.88,1.12,  0.82,1.18
# augment_stretch = Compose([TimeStretch(min_rate=rate, max_rate=rate, p=1)])

# # Pitch
# pitch_factor = -1
# augment_pitch = Compose([PitchShift(min_semitones=pitch_factor, max_semitones=pitch_factor, p=1)])

# # Noise + Gain
# augment_noise_gain = Compose(
#     [Gain(min_gain_db=-15, max_gain_db=15, p=1),
#      AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.01, p=1)])

# # pitch, stretch, noise
# pitch_factor = -1#,1 -2,2
# rate = 0.94 #,1.06, 0.88,1.12
# augment_pitch_stretch_noise = Compose([
#     TimeStretch(min_rate=rate, max_rate=rate, p=1),
#     PitchShift(min_semitones=pitch_factor, max_semitones=pitch_factor, p=1),
#     AddGaussianNoise(min_amplitude=0.01, max_amplitude=0.015, p=0.5)
# ])

def aug_timestretch(rate):
    augment_stretch = Compose([TimeStretch(min_rate=rate, max_rate=rate, p=1, leave_length_unchanged=False)])
    return augment_stretch

def aug_pitch(pitch_factor):
    augment_pitch = Compose([PitchShift(min_semitones=pitch_factor, max_semitones=pitch_factor, p=1)])
    return augment_pitch

def aug_noise_gain():
    augment_noise_gain = Compose(
        [Gain(min_gain_db=-15, max_gain_db=15, p=1),
         AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.01, p=1)])
    return augment_noise_gain

def aug_all(pitch_factor, rate):
    augment_pitch_stretch_noise = Compose([
        TimeStretch(min_rate=rate, max_rate=rate, p=1, leave_length_unchanged=False),
        PitchShift(min_semitones=pitch_factor, max_semitones=pitch_factor, p=1),
        AddGaussianNoise(min_amplitude=0.01, max_amplitude=0.015, p=0.5)])
    return augment_pitch_stretch_noise

def make_wav_aug(aug_obj:Compose, song_names_path: str, path_wav:str, path_save:str):
    song_names = song_names_txt_to_list(song_names_path)
    for i, name in enumerate(song_names):
        signal, sr = librosa.load(f"{path_wav}/{name}.wav", sr=None)
        augmented_signal = aug_obj(signal, sr)
        sf.write(f"{path_save}/{name}.wav", augmented_signal, sr)


if __name__ == '__main__':

    path_labels = "/Users/tomermassas/Desktop/project melody extraction/Datasets/medlyDB/labels"


    make_input_labels_normal(path_labels=path_labels,
                             path_freqs="/Volumes/New Volume/Tomer/original/freqs",
                             path_names="/Users/tomermassas/Desktop/project melody extraction/Datasets/medlyDB/medly_song_names.txt",
                             path_save="/Volumes/New Volume/Tomer/original"
                             )
    print(" | ----- Done Original ----- |")


    make_input_labels_normal(path_labels=path_labels,
                             path_freqs="/Volumes/New Volume/Tomer/aug_noise_gain/freqs",
                             path_names="/Users/tomermassas/Desktop/project melody extraction/Datasets/medlyDB/medly_song_names.txt",
                             path_save="/Volumes/New Volume/Tomer/aug_noise_gain"
                             )
    print(" | ----- Done Noise + Gain ----- |")

    make_input_labels_timestretch(path_labels=path_labels,
                                  path_freqs="/Volumes/New Volume/Tomer/aug_timestretch_1/freqs",
                                  path_names="/Volumes/New Volume/Tomer/sorted_songs/timestretch/group_1.txt",
                                  path_save="/Volumes/New Volume/Tomer/aug_timestretch_1",
                                  rate=0.94)
    make_input_labels_timestretch(path_labels=path_labels,
                                  path_freqs="/Volumes/New Volume/Tomer/aug_timestretch_1/freqs",
                                  path_names="/Volumes/New Volume/Tomer/sorted_songs/timestretch/group_2.txt",
                                  path_save="/Volumes/New Volume/Tomer/aug_timestretch_1",
                                  rate=1.06)
    print(" | ----- Done TimeStretch 1 ----- |")

    make_input_labels_timestretch(path_labels=path_labels,
                                  path_freqs="/Volumes/New Volume/Tomer/aug_timestretch_2/freqs",
                                  path_names="/Volumes/New Volume/Tomer/sorted_songs/timestretch/group_1.txt",
                                  path_save="/Volumes/New Volume/Tomer/aug_timestretch_2",
                                  rate=1.12)
    make_input_labels_timestretch(path_labels=path_labels,
                                  path_freqs="/Volumes/New Volume/Tomer/aug_timestretch_2/freqs",
                                  path_names="/Volumes/New Volume/Tomer/sorted_songs/timestretch/group_2.txt",
                                  path_save="/Volumes/New Volume/Tomer/aug_timestretch_2",
                                  rate=0.88)
    print(" | ----- Done TimeStretch 2 ----- |")

    make_input_labels_timestretch(path_labels=path_labels,
                                  path_freqs="/Volumes/New Volume/Tomer/aug_timestretch_3/freqs",
                                  path_names="/Volumes/New Volume/Tomer/sorted_songs/timestretch/group_1.txt",
                                  path_save="/Volumes/New Volume/Tomer/aug_timestretch_3",
                                  rate=0.82)
    make_input_labels_timestretch(path_labels=path_labels,
                                  path_freqs="/Volumes/New Volume/Tomer/aug_timestretch_3/freqs",
                                  path_names="/Volumes/New Volume/Tomer/sorted_songs/timestretch/group_2.txt",
                                  path_save="/Volumes/New Volume/Tomer/aug_timestretch_3",
                                  rate=1.18)
    print(" | ----- Done TimeStretch 3 ----- |")

    make_input_labels_pitch(path_labels=path_labels,
                            path_freqs="/Volumes/New Volume/Tomer/aug_pitch_1/freqs",
                            path_names="/Volumes/New Volume/Tomer/sorted_songs/pitch/up_final_1.txt",
                            path_save="/Volumes/New Volume/Tomer/aug_pitch_1",
                            pitch=1)
    make_input_labels_pitch(path_labels=path_labels,
                            path_freqs="/Volumes/New Volume/Tomer/aug_pitch_1/freqs",
                            path_names="/Volumes/New Volume/Tomer/sorted_songs/pitch/down_final_1.txt",
                            path_save="/Volumes/New Volume/Tomer/aug_pitch_1",
                            pitch=-1)
    print(" | ----- Done Pitch 1 ----- |")


    make_input_labels_pitch(path_labels=path_labels,
                            path_freqs="/Volumes/New Volume/Tomer/aug_pitch_2/freqs",
                            path_names="/Volumes/New Volume/Tomer/sorted_songs/pitch/up_final_2.txt",
                            path_save="/Volumes/New Volume/Tomer/aug_pitch_2",
                            pitch=2)
    make_input_labels_pitch(path_labels=path_labels,
                            path_freqs="/Volumes/New Volume/Tomer/aug_pitch_2/freqs",
                            path_names="/Volumes/New Volume/Tomer/sorted_songs/pitch/down_final_2.txt",
                            path_save="/Volumes/New Volume/Tomer/aug_pitch_2",
                            pitch=-2)
    print(" | ----- Done Pitch 2 ----- |")


    make_input_labels_pitch(path_labels=path_labels,
                            path_freqs="/Volumes/New Volume/Tomer/aug_pitch_3/freqs",
                            path_names="/Volumes/New Volume/Tomer/sorted_songs/pitch/up_final_3.txt",
                            path_save="/Volumes/New Volume/Tomer/aug_pitch_3",
                            pitch=3)
    make_input_labels_pitch(path_labels=path_labels,
                            path_freqs="/Volumes/New Volume/Tomer/aug_pitch_3/freqs",
                            path_names="/Volumes/New Volume/Tomer/sorted_songs/pitch/down_final_3.txt",
                            path_save="/Volumes/New Volume/Tomer/aug_pitch_3",
                            pitch=-3)
    print(" | ----- Done Pitch 3 ----- |")

    make_input_labels_timestretch_pitch(path_labels=path_labels,
                                        path_freqs="/Volumes/New Volume/Tomer/aug_TS_P_N_G_1/freqs",
                                        path_names="/Volumes/New Volume/Tomer/sorted_songs/timestretch/group_1.txt",
                                        path_save="/Volumes/New Volume/Tomer/aug_TS_P_N_G_1",
                                        rate=1.06,
                                        pitch=1)
    make_input_labels_timestretch_pitch(path_labels=path_labels,
                                        path_freqs="/Volumes/New Volume/Tomer/aug_TS_P_N_G_1/freqs",
                                        path_names="/Volumes/New Volume/Tomer/sorted_songs/timestretch/group_2.txt",
                                        path_save="/Volumes/New Volume/Tomer/aug_TS_P_N_G_1",
                                        rate=0.94,
                                        pitch=-1)
    print(" | ----- Done TimeStretch + Pitch + Noise + Gain 1 ----- |")


    make_input_labels_timestretch_pitch(path_labels=path_labels,
                                        path_freqs="/Volumes/New Volume/Tomer/aug_TS_P_N_G_2/freqs",
                                        path_names="/Volumes/New Volume/Tomer/sorted_songs/timestretch/group_1.txt",
                                        path_save="/Volumes/New Volume/Tomer/aug_TS_P_N_G_2",
                                        rate=0.88,
                                        pitch=-2)
    make_input_labels_timestretch_pitch(path_labels=path_labels,
                                        path_freqs="/Volumes/New Volume/Tomer/aug_TS_P_N_G_2/freqs",
                                        path_names="/Volumes/New Volume/Tomer/sorted_songs/timestretch/group_2.txt",
                                        path_save="/Volumes/New Volume/Tomer/aug_TS_P_N_G_2",
                                        rate=1.12,
                                        pitch=2)
    print(" | ----- Done TimeStretch + Pitch + Noise + Gain 2 ----- |")


