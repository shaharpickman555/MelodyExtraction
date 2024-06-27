import os.path

import numpy as np
import matplotlib.pyplot as plt
from preprocess.utility_functions import (make_audio, save_lines, read_ground_truth, read_freqs, plot_histogram,
                                          match_justing_freqs_to_ground_truth, analize_justin_with_ground_truth,
                                          make_from_many, make_from_csv, song_names_txt_to_list)
from preprocess.Lifshits import process

def freqs_bin_edges(start: int = 55, end: int = 1760, semitone_fraction: int =10):
    edges = []
    edges.append(0)
    edges.append(start)
    cur = start
    while cur < end:
        cur = cur * (2 ** (1 / (12 * semitone_fraction)))
        edges.append(cur)
    return np.array(edges)


def find_bin_index(freq, bin_edges):
    index = np.digitize(freq, bin_edges)
    return index - 1

def vectorize_label(freq, edges):
        index = find_bin_index(freq, edges)
        vec = np.zeros(edges.shape[0]-1)
        vec[index] = 1
        return vec

def vectorize_input(freqs, edges, correction_val):
    vec = np.zeros(edges.shape[0] - 1)
    for f, amp in zip(freqs["line_freqs_amps"]["freqs"], freqs["line_freqs_amps"]["amps"]):
        index = find_bin_index(f*correction_val, edges)
        vec[index] = max(vec[index], round(amp, 3))
    return vec


def infer_one_song(song_file_path):
    freq_file_path = song_file_path + '.freqs'
    if not os.path.exists(freq_file_path):
        process(song_file_path, path_to_save='.', outputname=song_file_path, output_wave=False)
    input_freqs = read_freqs(freq_file_path)
    freq_edges = freqs_bin_edges(start=53.43425676, end=2453, semitone_fraction=10)
    vectors_input = []

    for freq in input_freqs:
        vectors_input.append(vectorize_input(freq, freq_edges, 1.0))

    input_vec_final = np.array(vectors_input)
    return input_vec_final , [elem['time'] for elem in input_freqs]

def idx_list_to_freqs(res_idx_list,semitone_fraction=10):
    res_idx_list[0] = 0
    res_idx_list[1:] = [ (53.434 *(2 ** (1/(12*semitone_fraction)))** (x-1)) if x > 0 else x for x in res_idx_list[1:]]
    return res_idx_list


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
def make_input_labels_timestretch_pitch(path_labels, path_freqs, path_names, path_save, rate, pitch, semitone_fraction=4):
    song_names = song_names_txt_to_list(path_names)
    low_range = 53.43425676
    high_range = 2453
    freq_edges = freqs_bin_edges(start=low_range, end=high_range, semitone_fraction=semitone_fraction)

    for name in song_names:
        gt_df = read_ground_truth(f"{path_labels}/{name}.csv")
        gt_df["time"] = gt_df["time"] / rate
        input_freqs = read_freqs(f"{path_freqs}/{name}.freqs")

        vectors_labels = []
        vectors_input = []

        s = 0
        for freq in input_freqs:
            time_input = freq["time"]
            min_dist = np.inf
            for i in range(s, len(gt_df), 1):
                cur_gt_row = gt_df.iloc[i]
                corc_val = 1 #correction_values[i]
                cur_dist = abs((round(time_input,6) - round(cur_gt_row["time"],6)))
                if cur_dist <= min_dist:
                    min_dist = cur_dist
                    s = i+1
                else:
                    vectors_labels.append(find_bin_index(cur_gt_row["freq"], freq_edges))
                    vectors_input.append(vectorize_input(freq, freq_edges, corc_val))
                    break

        input_vec_final = np.array(vectors_input)
        new_input = add_pitch_to_input(input_vec_final, shift=pitch*semitone_fraction)
        labels_vec_final = np.array(vectors_labels)
        new_labels = add_pitch_to_labels(labels_vec_final, value_to_add=pitch*semitone_fraction)

        np.save(f"{path_save}/input/{name}", new_input)
        np.save(f"{path_save}/label/{name}", new_labels)


def make_input_labels_timestretch(path_labels, path_freqs, path_names, path_save, rate, semitone_fraction=4):
    song_names = song_names_txt_to_list(path_names)
    low_range = 53.43425676
    high_range = 2453
    freq_edges = freqs_bin_edges(start=low_range, end=high_range, semitone_fraction=semitone_fraction)

    for name in song_names:
        gt_df = read_ground_truth(f"{path_labels}/{name}.csv")
        gt_df["time"] = gt_df["time"] / rate
        input_freqs = read_freqs(f"{path_freqs}/{name}.freqs")

        vectors_labels = []
        vectors_input = []

        s = 0
        for freq in input_freqs:
            time_input = freq["time"]
            min_dist = np.inf
            for i in range(s, len(gt_df), 1):
                cur_gt_row = gt_df.iloc[i]
                corc_val = 1  # correction_values[i]
                cur_dist = abs((round(time_input, 6) - round(cur_gt_row["time"], 6)))
                if cur_dist <= min_dist:
                    min_dist = cur_dist
                    s = i + 1
                else:
                    vectors_labels.append(find_bin_index(cur_gt_row["freq"], freq_edges))
                    vectors_input.append(vectorize_input(freq, freq_edges, corc_val))
                    break

        input_vec_final = np.array(vectors_input)
        labels_vec_final = np.array(vectors_labels)

        np.save(f"{path_save}/input/{name}", input_vec_final)
        np.save(f"{path_save}/label/{name}", labels_vec_final)


def make_input_labels_pitch(path_labels, path_freqs, path_names, path_save, pitch, semitone_fraction=4):
    song_names = song_names_txt_to_list(path_names)
    low_range = 53.43425676
    high_range = 2453
    freq_edges = freqs_bin_edges(start=low_range, end=high_range, semitone_fraction=semitone_fraction)

    for name in song_names:
        gt_df = read_ground_truth(f"{path_labels}/{name}.csv")
        input_freqs = read_freqs(f"{path_freqs}/{name}.freqs")

        vectors_labels = []
        vectors_input = []

        s = 0
        for freq in input_freqs:
            time_input = freq["time"]
            min_dist = np.inf
            for i in range(s, len(gt_df), 1):
                cur_gt_row = gt_df.iloc[i]
                corc_val = 1  # correction_values[i]
                cur_dist = abs((round(time_input, 6) - round(cur_gt_row["time"], 6)))
                if cur_dist <= min_dist:
                    min_dist = cur_dist
                    s = i + 1
                else:
                    vectors_labels.append(find_bin_index(cur_gt_row["freq"], freq_edges))
                    vectors_input.append(vectorize_input(freq, freq_edges, corc_val))
                    break

        input_vec_final = np.array(vectors_input)
        new_input = add_pitch_to_input(input_vec_final, shift=pitch*semitone_fraction)
        labels_vec_final = np.array(vectors_labels)
        new_labels = add_pitch_to_labels(labels_vec_final, value_to_add=pitch*semitone_fraction)

        np.save(f"{path_save}/input/{name}", new_input)
        np.save(f"{path_save}/label/{name}", new_labels)

def make_input_labels_normal(path_labels, path_freqs, path_names, path_save, semitone_fraction=4):
    song_names = song_names_txt_to_list(path_names)
    low_range = 53.43425676
    high_range = 2453
    freq_edges = freqs_bin_edges(start=low_range, end=high_range, semitone_fraction=semitone_fraction)

    for name in song_names:
        gt_df = read_ground_truth(f"{path_labels}/{name}.csv")
        input_freqs = read_freqs(f"{path_freqs}/{name}.freqs")

        vectors_labels = []
        vectors_input = []

        s = 0
        for freq in input_freqs:
            time_input = freq["time"]
            min_dist = np.inf
            for i in range(s, len(gt_df), 1):
                cur_gt_row = gt_df.iloc[i]
                corc_val = 1  # correction_values[i]
                cur_dist = abs((round(time_input, 6) - round(cur_gt_row["time"], 6)))
                if cur_dist <= min_dist:
                    min_dist = cur_dist
                    s = i + 1
                else:
                    vectors_labels.append(find_bin_index(cur_gt_row["freq"], freq_edges))
                    vectors_input.append(vectorize_input(freq, freq_edges, corc_val))
                    break

        input_vec_final = np.array(vectors_input)
        labels_vec_final = np.array(vectors_labels)

        np.save(f"{path_save}/input/{name}", input_vec_final)
        np.save(f"{path_save}/label/{name}", labels_vec_final)