import math, time, itertools, os, zipfile, bisect, random, subprocess
import torch
import numpy as np
from scipy.io import wavfile
import librosa
import yulewalker
from scipy import signal
from tqdm.contrib.concurrent import process_map
import tqdm
from preprocess.utility_functions import make_audio, save_lines
import sys


# PARAMETERS:
EQUAL_LOUDNESS_CURVE_80PHON = [(0, 120), (20, 113), (30, 103), (40, 97), (50, 93), (60, 91), (70, 89), (80, 87), (90, 86),
                               (100, 85), (200, 78), (300, 76), (400, 76), (500, 76), (600, 76), (700, 77), (800, 78),
                               (900, 79.5), (1000, 80), (1500, 79), (2000, 77), (2500, 74), (3000, 71.5), (3700, 70),
                               (4000, 70.5), (5000, 74), (6000, 79), (7000, 84), (8000, 86), (9000, 86), (10000, 85),
                               (12000, 95), (15000, 110), (20000, 125)]

NYQUIST_DB = 140 #44100 and 48000 are basically the same value

#parameters according to Salomon
N_FFT = 8192
WIN_LENGTH = 2048
HOP_LENGTH = 512 #about 10ms
THRESHOLD_FOR_MAX = 1e-4 #-80db
SUBHARMONICS_NUM = 20
ALPHA = 0.8
SALIENCE_THRESHOLD = 1e-2 #-40db
BETA = 1.0
SALIENCE_SECOND_FILTER_FACTOR = 0.9
# Salomon, 55 -> 0, 1760 -> 600 (5 octaves), log scale, every 10 bins is a semitone
LOGBIN_COUNT = 600
LOGBIN_MIN_FREQ = 55
LOGBIN_MAX_FREQ = 1760

#precomputed values for faster implementation
HARMONICS_COEFFS = np.arange(1, SUBHARMONICS_NUM + 1)
SUBHARMONICSS_COEFFS = np.ones((SUBHARMONICS_NUM,)) / HARMONICS_COEFFS
LOGBIN_RANGE = np.arange(LOGBIN_COUNT).reshape((LOGBIN_COUNT, 1, 1))
ALPHA_FACTOR_COEFFS = ALPHA ** (HARMONICS_COEFFS - 1)


def np_freqs_to_logbins(freqs, floor_):
    logbins = (1200 / 10) * np.log2(freqs / 55) + 0.5

    if not floor_:
        return logbins
    return np.floor(logbins).astype(np.int32)

def torch_freqs_to_logbins(freqs, floor_):
    logbins = (1200 / 10) * torch.log2(freqs / 55) + 0.5

    if not floor_:
        return logbins
    return torch.floor(logbins).long()

def logbins_to_freqs(logbins):
    return (2 ** ((logbins - 0.5) / (1200 / 10))) * 55

use_torch=False
if use_torch:
    device = torch.device('cuda')
    HARMONICS_COEFFS = torch.from_numpy(HARMONICS_COEFFS).to(device)
    SUBHARMONICSS_COEFFS = torch.from_numpy(SUBHARMONICSS_COEFFS).to(device)
    LOGBIN_RANGE = torch.from_numpy(LOGBIN_RANGE).to(device)
    ALPHA_FACTOR_COEFFS = torch.from_numpy(ALPHA_FACTOR_COEFFS).to(device)

    np_max = lambda x: torch.max(x).item()
    np_stack = lambda x, dim: torch.stack(x, dim=dim)
    np_diag = torch.diag
    np_cos = torch.cos
    freqs_to_logbins = torch_freqs_to_logbins
    np_sum = lambda x, dim: torch.sum(x, dim=dim)
else:
    np_max = np.max
    np_stack = lambda x, dim: np.stack(x, axis=dim)
    np_diag = np.diag
    np_cos = np.cos
    freqs_to_logbins = np_freqs_to_logbins
    np_sum = lambda x, dim: np.sum(x, axis=dim)


# CODE:
def equal_loudness(sr):
    inverted = [(f / (sr / 2), db2mag(70 - db)) for f, db in EQUAL_LOUDNESS_CURVE_80PHON + [(sr / 2, NYQUIST_DB)]]

    inverted_f, inverted_m = zip(*inverted)

    a_yw, b_yw = yulewalker.yulewalk(10, np.array(inverted_f), np.array(inverted_m))
    b_bw, a_bw = signal.butter(2, 150 / (sr / 2), 'high')
    return a_yw, b_yw, a_bw, b_bw

def db2mag(db):
    """converts Decibel to magnitude"""
    return 10 ** (db / 20)

def process_map_(f, args, do_multi=False, **kwargs):
    if do_multi:
        return process_map(f, args, **kwargs)
    else:
        return [f(arg) for arg in tqdm.tqdm(args)]

@torch.no_grad()
def process_single_frame(args):
    freqs, sr = args

    # filter local maxima, with threshold from global maxima
    peaks, _ = signal.find_peaks(freqs)

    if not peaks.size:
        return []

    if use_torch:
        peaks = torch.from_numpy(peaks).to(device)
        freqs = torch.from_numpy(freqs).to(device)

    thresh = np_max(freqs[peaks]) * THRESHOLD_FOR_MAX
    peaks = peaks[freqs[peaks] >= thresh]

    # quadric interpolation of frequency
    with_adj = np_stack((freqs[peaks - 1], freqs[peaks], freqs[peaks + 1]), dim=-1)

    corrected_d = ((with_adj[:, 0] - with_adj[:, 2]) / (
                with_adj[:, 0] + with_adj[:, 2] - 2 * with_adj[:, 1] + 1e-9)) / 2

    corrected_freqs = (peaks + corrected_d) * sr / N_FFT
    corrected_amps = (with_adj[:, 1] - (corrected_d * (with_adj[:, 0] - with_adj[:, 2]) / 4))

    # filter to frequency range
    mask = (LOGBIN_MIN_FREQ <= corrected_freqs) & (corrected_freqs <= LOGBIN_MAX_FREQ)
    corrected_freqs = corrected_freqs[mask]
    corrected_amps = corrected_amps[mask]

    num_freqs = corrected_freqs.shape[0]
    if not num_freqs:
        return []

    max_amp = np_max(corrected_amps)

    # prepare subharmonic coefficients
    coeffs_matrix = vecrepeat(SUBHARMONICSS_COEFFS, True, num_freqs)

    # (num_freqs, subharmonics_num)
    subharmonics = np_diag(corrected_freqs) @ coeffs_matrix
    subharmonic_logs = freqs_to_logbins(subharmonics, False)

    subharmonics_logs_tiled = vecrepeat(subharmonic_logs, True, LOGBIN_COUNT)

    # calculate (B(fi/hi) - b) / 10, according to Salomon to all frequencies and subharmonics (double sum) (logbin_count, num_freqs, subharmonics_num)
    weights = (subharmonics_logs_tiled - LOGBIN_RANGE) / 10
    alpha_factor = vecrepeat(ALPHA_FACTOR_COEFFS, True, LOGBIN_COUNT * num_freqs).reshape(
        (LOGBIN_COUNT, num_freqs, *ALPHA_FACTOR_COEFFS.shape))

    # filter out of range
    weight_mask = (-1 <= weights) & (weights <= 1)

    # weight each subharmonic contribution according to a cosine squared
    # cosine^2 lobes with alpha^ (h - 1)
    # weights[weight_mask] = cos2_pi_2(weights[weight_mask])
    weights[weight_mask] = np_cos(weights[weight_mask] * torch.pi / 2) ** 2
    weights[~weight_mask] = 0

    # calculate G according to Salomon (for all values in sums)
    G = weights * alpha_factor

    # multiply by filtered (amps ** beta) (both E and ai^b in Salomon)
    amps_tiled = vecrepeat_both(corrected_amps, LOGBIN_COUNT, SUBHARMONICS_NUM)

    amps_tiled[amps_tiled < (max_amp * SALIENCE_THRESHOLD)] = 0
    amps_tiled **= BETA

    # multiply all factors and double sum
    salience = np_sum(amps_tiled * G, dim=(2, 1))

    if use_torch:
        salience = salience.cpu().numpy()

    # filter only peaks
    salience_peaks, _ = signal.find_peaks(salience)

    # filter more
    salience_mean = np.mean(salience[salience_peaks])
    salience_std = np.std(salience[salience_peaks])
    salience_second_threshold = salience_mean - salience_std * SALIENCE_SECOND_FILTER_FACTOR
    salience_peaks = salience_peaks[salience[salience_peaks] >= salience_second_threshold]

    salience_peaks = salience_peaks[salience[salience_peaks] >= 1.0]  # absolute and reasonable #TODO check this is good
    ###

    # convert back to Hz pairs (f,a)
    freq_frame = [(logbins_to_freqs(i), salience[i]) for i in salience_peaks]
    # sort by amplitude
    freq_frame.sort(key=lambda x: x[1], reverse=True)

    return freq_frame

def vecrepeat(x, to_start, amount):
    axis = 0 if to_start else -1
    tilearg = [1] * (len(x.shape) + 1)
    tilearg[axis] = amount

    if use_torch:
        return x.unsqueeze(dim=axis).repeat(tilearg)
    else:
        return np.tile(np.expand_dims(x, axis=axis), tilearg)

def vecrepeat_both(x, amount_start, amount_end):
    tilearg = [1] * (len(x.shape) + 2)
    tilearg[0] = amount_start
    tilearg[-1] = amount_end

    if use_torch:
        return x.unsqueeze(0).unsqueeze(-1).repeat(tilearg)
    else:
        return np.tile(np.expand_dims(np.expand_dims(x, axis=0), axis=-1), tilearg)


# def process_freqs_amps_extraaction(path_wav, outputname, output_wave=False):
def process_freqs_amps_extraction(path_wav):
    """implemented according to Justin Salomon's paper:
        runs the process of extracting the most probable freqquncies and amplitudes for every time frames"""
    sr, x = wavfile.read(path_wav)
    if sr not in [44100, 48000]:# this is mainly because of the equal_loudness filter
        raise ValueError('unsupported sr')

    # convert stereo to mono
    if len(x.shape) != 1:
        x = np.mean(x, axis=1)

    # attenuate signal to human perception
    a_yw, b_yw, a_bw, b_bw = equal_loudness(sr)
    x = signal.filtfilt(b_bw, a_bw, signal.filtfilt(b_yw, a_yw, x))

    # normalize
    x /= np.max(np.abs(x))

    Sxx = np.abs(librosa.stft(x, n_fft=N_FFT, win_length=WIN_LENGTH, hop_length=HOP_LENGTH))

    result = process_map_(process_single_frame, [(Sxx[:, t], sr) for t in range(Sxx.shape[1])], max_workers=2, chunksize=10, do_multi=True)

    # convert list of numpy arrays to matrix
    result = np.stack(list(itertools.zip_longest(*result, fillvalue=np.zeros((2,)))), axis=1)

    return result, sr
    # if output_wave:
    #     print()
    #     # make_audio(result, sr, f"{path_to_save}/wavs/{outputname}", split_lines=True)
    # else:
    #     save_lines(result, sr, outputname)

