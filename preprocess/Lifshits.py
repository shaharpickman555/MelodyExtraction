import math, time, itertools, os, zipfile, bisect, random, subprocess
import torch
import numpy as np
from scipy.io import wavfile
import librosa
import yulewalker
from scipy import signal
from tqdm.contrib.concurrent import process_map
import tqdm
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
from    preprocess.utility_functions import read_freqs
def db2mag(db):
    return 10 ** (db / 20)

#(freq, db) (Fletcher–Munson curve at 80db)
equal_loudness_curve_80db = [(0, 120), (20, 113), (30, 103), (40, 97), (50, 93), (60, 91), (70, 89), (80, 87), (90, 86), (100, 85), (200, 78), (300, 76), (400, 76), (500, 76), (600, 76), (700, 77), (800, 78), (900, 79.5), (1000, 80), (1500, 79), (2000, 77), (2500, 74), (3000, 71.5), (3700, 70), (4000, 70.5), (5000, 74), (6000, 79), (7000, 84), (8000, 86), (9000, 86), (10000, 85), (12000, 95), (15000, 110), (20000, 125)]
nyquist_db = 140 #44100 and 48000 are basically the same value

#create equal loudness time domain filter coefficients
def equal_loudness(sr):
    #                            invert? why 70
    inverted = [(f / (sr / 2), db2mag(70 - db)) for f, db in equal_loudness_curve_80db + [(sr / 2, nyquist_db)]]
    
    inverted_f, inverted_m = zip(*inverted)
    
    a_yw, b_yw = yulewalker.yulewalk(10, np.array(inverted_f), np.array(inverted_m))
    b_bw, a_bw = signal.butter(2, 150 / (sr / 2), 'high')
    return a_yw, b_yw, a_bw, b_bw

def mean(l):
    return sum(l) / len(l)
   
def most(l):
    return max(set(l), key=l.count)
    
def f_to_220_440(f):
    if f <= 0:
        return f
        
    while f < 220:
        f *= 2
    
    while f > 440:
        f /= 2
    
    return f
    
#csv of the form time,freq[,confidence, silence_confidence] to be turned into wav file
def make_from_csv(path,path_to_save, debounce=1, silence_debounce=1, same_octave=False):
    data = open(path, 'r').readlines()
    
    if data[0][0] not in '0123456789.-':
        data = data[1:]

    freqs = [tuple(float(i) for i in d.replace('\t', ',').replace(' ', ',').split(',')) for d in data]
    
    confidence_threshold = 0.1
    silence_confidence_threshold = (0.5, 0.5)
    
    if len(freqs[0]) == 2:
        freqs = [(t, f, 1 if f > 0 else 0, 0 if f > 0 else 1) for t,f in freqs]
    elif len(freqs[0]) == 3:
        freqs = [(t, f, c if f > 0 else 0, 0 if f > 0 else 1) for t,f,c in freqs]
    
    freqs[-2] = (freqs[-2][0], freqs[-2][1], 0, 1)
    freqs[-1] = (freqs[-1][0], freqs[-1][1], 0, 1)
    
    sr = 44100

    lasttime = freqs[-1][0]

    sample_num = math.ceil(lasttime * sr + 1)

    out = np.zeros((sample_num,))
    
    last_good = 0
    last_silence = True

    phase = 0
    for i, ((t, f, c, s), (t_next, _, _, _)) in enumerate(zip(freqs, freqs[1:])):
        start = math.floor(t * sr)
        many = math.ceil((t_next - t) * sr)
            
        if c < confidence_threshold:
            #phase = 0
            real_f = last_good
        else:
            real_f = most([ff for _, ff, _, _ in freqs[max(0, i - debounce) : i + 1]])
            if same_octave:
                real_f = f_to_220_440(real_f)
            
            last_good = real_f
        
        if not (silence_confidence_threshold[0] < s < silence_confidence_threshold[1]):
            real_silence = most([ss >= silence_confidence_threshold[1] for _, _, _, ss in freqs[max(0, i - silence_debounce) : i + 1]])
        
            last_silence = real_silence
            
        if last_silence:
            real_f = 0
        
        phase_add = real_f * 2 * np.pi * many / sr
        
        samples = np.linspace(phase, phase + phase_add, many, endpoint=False)
        
        phase = (phase + phase_add) % (2 * np.pi)
        
        out[start : start + many] = np.sin(samples)
        
    wavfile.write(f'{path_to_save}/result.wav', sr, out)
    
# def make_from_many(path):
    # data = open(path, 'r').readlines()
    # data = np.array([[(i, float(v)) for i,v in enumerate(line.split(','))] for line in data])
    
    # #print(data)
    # for t in range(data.shape[0]):
        # data[t, :, 0] = (2 ** ((data[t, :, 0].astype(np.float64) - 0.5) / (1200 / 10))) * 55
    
    # #print(data)
    
    # sr = 44100
    # make_audio(data, sr, 'result')
    
#optimization for cos(x*pi/2)**2
def cos2_pi_2(x):
    pi2 = np.pi ** 2
    x2 = x ** 2
    return 1 - (pi2 * x2 / 5) + (pi2 * pi2 * (x2 ** 2) / 100)

#parameters according to Salomon
N_fft = 8192 
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

#precomputed values for faster implementation
HARMONICS_COEFFS = np.arange(1, SUBHARMONICS_NUM + 1)
SUBHARMONICSS_COEFFS = np.ones((SUBHARMONICS_NUM,)) / HARMONICS_COEFFS
LOGBIN_RANGE = np.arange(LOGBIN_COUNT).reshape((LOGBIN_COUNT, 1, 1))
ALPHA_FACTOR_COEFFS = ALPHA ** (HARMONICS_COEFFS - 1)

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
 
@torch.no_grad() 
def process_single_frame(args):
    freqs, sr = args

    #filter local maxima, with threshold from global maxima
    peaks, _ = signal.find_peaks(freqs)

    if not peaks.size:
        return []

    if use_torch:
        peaks = torch.from_numpy(peaks).to(device)
        freqs = torch.from_numpy(freqs).to(device)

    thresh = np_max(freqs[peaks]) * THRESHOLD_FOR_MAX
    peaks = peaks[freqs[peaks] >= thresh]

    #quadric interpolation of frequency
    with_adj = np_stack((freqs[peaks - 1], freqs[peaks], freqs[peaks + 1]), dim=-1)

    corrected_d = ((with_adj[:,0] - with_adj[:,2]) / (with_adj[:,0] + with_adj[:,2] - 2 * with_adj[:,1] + 1e-9)) / 2

    corrected_freqs = (peaks + corrected_d) * sr / N_fft
    corrected_amps = (with_adj[:,1] - (corrected_d * (with_adj[:,0] - with_adj[:,2]) / 4))

    #filter to frequency range
    mask = (LOGBIN_MIN_FREQ <= corrected_freqs) & (corrected_freqs <= LOGBIN_MAX_FREQ)
    corrected_freqs = corrected_freqs[mask]
    corrected_amps = corrected_amps[mask]

    num_freqs = corrected_freqs.shape[0]
    if not num_freqs:
        return []

    max_amp = np_max(corrected_amps)

    #prepare subharmonic coefficients
    coeffs_matrix = vecrepeat(SUBHARMONICSS_COEFFS, True, num_freqs)

    # (num_freqs, subharmonics_num)
    subharmonics = np_diag(corrected_freqs) @ coeffs_matrix
    subharmonic_logs = freqs_to_logbins(subharmonics, False)

    subharmonics_logs_tiled = vecrepeat(subharmonic_logs, True, LOGBIN_COUNT)

    #calculate (B(fi/hi) - b) / 10, according to Salomon to all frequencies and subharmonics (double sum) (logbin_count, num_freqs, subharmonics_num)
    weights = (subharmonics_logs_tiled - LOGBIN_RANGE) / 10
    alpha_factor = vecrepeat(ALPHA_FACTOR_COEFFS, True, LOGBIN_COUNT * num_freqs).reshape((LOGBIN_COUNT, num_freqs, *ALPHA_FACTOR_COEFFS.shape))

    #filter out of range
    weight_mask = (-1 <= weights) & (weights <= 1)

    #weight each subharmonic contribution according to a cosine squared
    #cosine^2 lobes with alpha^ (h - 1)
    #weights[weight_mask] = cos2_pi_2(weights[weight_mask])
    weights[weight_mask] = np_cos(weights[weight_mask] * torch.pi / 2) ** 2
    weights[~weight_mask] = 0
    #################

    #calculate G according to Salomon (for all values in sums)
    G = weights * alpha_factor

    #multiply by filtered (amps ** beta) (both E and ai^b in Salomon)
    amps_tiled = vecrepeat_both(corrected_amps, LOGBIN_COUNT, SUBHARMONICS_NUM)

    amps_tiled[amps_tiled < (max_amp * SALIENCE_THRESHOLD)] = 0
    amps_tiled **= BETA

    #multiply all factors and double sum
    #(num_freqs,)
    salience = np_sum(amps_tiled * G, dim=(2, 1))

    if use_torch:
        salience = salience.cpu().numpy()

    #filter only peaks
    salience_peaks, _ = signal.find_peaks(salience)

    #filter more
    salience_mean = np.mean(salience[salience_peaks])
    salience_std = np.std(salience[salience_peaks])
    salience_second_threshold = salience_mean - salience_std * SALIENCE_SECOND_FILTER_FACTOR
    salience_peaks = salience_peaks[salience[salience_peaks] >= salience_second_threshold]

    salience_peaks = salience_peaks[salience[salience_peaks] >= 1.0] #absolute and reasonable


    #convert back to Hz pairs (f,a)
    freq_frame = [(logbins_to_freqs(i), salience[i]) for i in salience_peaks]
    #sort by amplitude
    freq_frame.sort(key=lambda x: x[1], reverse=True)

    return freq_frame

def process_map_(f, args, do_multi=False, **kwargs):
    if do_multi:
        return process_map(f, args, **kwargs)
    else:
        return [f(arg) for arg in tqdm.tqdm(args)]

def plot_spectrogram(Y, sr, hop_length, y_axis="linear"):
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(Y,
                             sr=sr,
                             hop_length=hop_length,
                             x_axis="time",
                             y_axis=y_axis)
    plt.colorbar(format="%+2.f")
    plt.show()

def process(path,path_to_save, outputname, output_wave=False):
    sr, x = wavfile.read(path)
    # if sr not in [44100, 48000]:
    #     # this is mainly because of the equal_loudness filter
    #     raise ValueError('unsupported sr')

    if len(x.shape) != 1:
        #average all channels
        x = np.mean(x, axis=1)

    # frequencies, times, spectrogram = signal.spectrogram(x, sr)
    # plt.pcolormesh(times, frequencies, np.log(spectrogram))
    # plt.imshow(spectrogram)
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()

    #attenuate signal to human perception
    a_yw, b_yw, a_bw, b_bw = equal_loudness(sr)
    x = signal.filtfilt(b_bw, a_bw, signal.filtfilt(b_yw, a_yw, x))

    #normalize
    x /= np.max(np.abs(x))

    Sxx = np.abs(librosa.stft(x, n_fft=N_fft, win_length=WIN_LENGTH, hop_length=HOP_LENGTH))
    # print(Sxx.shape)
    # print(type(Sxx[0][0]))
    #
    # Y_scale = np.abs(Sxx) ** 2
    # print(Y_scale.shape)
    # print(type(Y_scale[0][0]))
    # # plot_spectrogram(Y_scale, sr, HOP_LENGTH)
    #
    # Y_log_scale = librosa.power_to_db(Y_scale)
    # # plot_spectrogram(Y_log_scale, sr, HOP_LENGTH)
    # plot_spectrogram(Y_log_scale, sr, HOP_LENGTH, y_axis="log")

    result = process_map_(process_single_frame, [(Sxx[:, t], sr) for t in range(Sxx.shape[1])], max_workers=1, chunksize=10, do_multi=True)
    # result = process_single_frame([Sxx[:, 450], sr])

    #convert list of numpy arrays to matrix
    result = np.stack(list(itertools.zip_longest(*result, fillvalue=np.zeros((2,)))), axis=1)

    if output_wave:
        make_audio(result, sr, f"{path_to_save}/{outputname}", split_lines=True)
    else:
        save_lines(result, sr, f"{path_to_save}/{outputname}")

def process_mt(args):
    return process(*args)
        
def process_dirs(d):
    filenames = []
    for root, dirs, files in os.walk(d):
        for f in files:
            if os.path.splitext(f)[1] in ['.wav']:
                filenames.append(os.path.join(root, f))
    
    args = []
    for i, filename in enumerate(filenames):
        print(f'{i}/{len(filenames)} {filename}')
        outputname = os.path.splitext(filename)[0]
        if os.path.exists(f'{outputname}.freqs'):
            continue
        args.append((filename, outputname, False))

    process_map_(process_mt, args, max_workers=10, chunksize=1, do_multi=True)

def save_lines(result, sr, outputname):
    data = '\n'.join(', '.join(f'{f:.8f}: {a:.8f}' for f,a in timeframe if f != 0 and a != 0) for timeframe in result)
    with open(f'{outputname}.freqs', 'w') as fh:
        fh.write(f'{sr}\n')
        fh.write(data)

def process_augment(inputfile, outputname): 
    pitch_range = range(-3, 3 + 1, 1)
    tempo_range = range(-10, 10 + 1, 5)
    
    for p in pitch_range:
        for t in tempo_range:
            if t == 0 and p == 0:
                continue
            subprocess.run(['soundstretch', inputfile, f'{outputname}_p_{p}_t_{t}.wav', f'-tempo={t}', f'-pitch={p}'])
            freq_factor = math.pow(2, p / 12)
            tempo_factor = 100 / (100 + t)

            if os.path.exists(f'{outputname}.mel'):
                lines = open(f'{outputname}.mel', 'r').readlines()
            else:
                lines = open(f'{outputname.replace("_MIX", "_MELODY2")}.csv', 'r').readlines()
            lines = [[float(f) for f in line.replace('\t', ',').replace(' ', ',').split(',')] for line in lines]
            lines = [f'{t * tempo_factor},{f * freq_factor}' for t,f in lines]
            with open(f'{outputname}_p_{p}_t_{t}.csv', 'w') as fh:
                fh.write('\n'.join(lines))

def process_augment_mt(args):
    return process_augment(*args)

def process_augment_dirs(d):
    
    filenames = []
    for root, dirs, files in os.walk(d):
        for f in files:
            if os.path.splitext(f)[1] in ['.wav']:
                filenames.append((os.path.join(root, f), os.path.join(root, os.path.splitext(f)[0])))

    process_map_(process_augment_mt, filenames, max_workers=20, chunksize=1, do_multi=True) 

#array of (t, num_freqs, (f, a))
def make_audio(result, sr, outputname, split_lines=False):
    
    freq_threshold = 1
    many = HOP_LENGTH
    
    waves = []
    for l in tqdm.tqdm(range(result.shape[1])):
        line = result[:,l,:]
        wave = np.zeros((result.shape[0] * many,), dtype=np.float64)
        phase = 0
        for t, (f,a) in enumerate(line):
            if f < freq_threshold or a == 0:
                continue

            phase_add = f * 2 * np.pi * many / sr
            wave[HOP_LENGTH * t: (HOP_LENGTH * t) + many] = np.sin(np.linspace(phase, phase + phase_add, many, endpoint=False)) * a
            
            phase += phase_add
        
        if split_lines:
            normwave = wave * (32768.0 / np.max(np.abs(wave)))
            wavfile.write(f'{outputname}.wav', sr, normwave.astype(np.int16))
            
        waves.append(wave)
            
    wave = np.sum(waves, axis=0)
    normwave = wave * (32768.0 / np.max(np.abs(wave)))
    wavfile.write(f'{outputname}.wav', sr, normwave.astype(np.int16))

def closest_float(sorted_arr, v, value_convert=lambda x:x, pad_value=None):
    threshold = 0.05
    
    i = bisect.bisect_left(sorted_arr, v)
    if i == 0:
        chosen = sorted_arr[0]
    elif i == len(sorted_arr):
        chosen = sorted_arr[i - 1]
    else:
        chosen = min((sorted_arr[i - 1], sorted_arr[i]), key=lambda x: abs(x - v))
        
    if abs(chosen - v) >= threshold:
        #if v is before or after sorted_arr, we just output pad_value (if the melody file stops before the wave actually ends)
        if i != len(sorted_arr) and i != 0:
            raise ValueError(f'Warning: too far: {v} - {chosen}')
        return pad_value
    
    return value_convert(chosen)
    
def add_melody_to_freqs(d):
    melody_files = {}
    freqs_files = []
    for root, dirs, files in os.walk(d):
        for f in files:
            ext = os.path.splitext(f)[1]
            if ext in ['.mel', '.csv']:
                melody_files[os.path.splitext(f)[0].replace('_MELODY2', '').replace('_MIX', '')] = os.path.join(root, f)
            if ext == '.freqs':
                freqs_files.append(os.path.join(root, f))

    #compression=zipfile.ZIP_DEFLATED
    for i, name in enumerate(tqdm.tqdm(freqs_files)):
        freqs = open(name, 'r').read()
        basename = os.path.splitext(os.path.basename(name))[0].replace('_MIX', '')
        melody_file = melody_files.get(basename)
        
        if not freqs:
            continue
            
        if not melody_file:
            raise ValueError(f'no melody for {basename}')
            
        #print(os.path.basename(name), melody_file, len(freqs))
        
        melody = dict((float(f) for f in l.replace('\t', ',').replace(' ',',').split(',')) for l in open(melody_file, 'r').readlines())
        melody_times = list(melody.keys()) #guaranteed sorted
        
        freq_lines = freqs.splitlines()
        sr = int(freq_lines[0])

        out_filename = f'{name}mel' #.freqsmel
        
        with open(out_filename, 'w') as out_fh:
            out_fh.write(freq_lines[0])
            
            for n, line in enumerate(freq_lines[1:]):
                t = HOP_LENGTH * n / sr
                
                mel = closest_float(melody_times, t, lambda k: melody[k], 0.0)
                    
                out_fh.write('\n')
                out_fh.write(str(mel))
                out_fh.write(', ')
                out_fh.write(line)

def prepare_dataset_worker(args):
    name = args

    freqs = open(name, 'r').read().splitlines()
    freqs = freqs[2:] #first line is sr

    total_Xb = []
    total_Yb = []

    first_moment_sum  = np.zeros((LOGBIN_COUNT,), dtype=np.float64)
    second_moment_sum = np.zeros((LOGBIN_COUNT,), dtype=np.float64)
    max_value = None
    min_value = None

    max_logbins_in_line = 50
    min_amp_filter = 1

    for time_index, line in enumerate(freqs):

        if ', ' not in line:
            raise ValueError('what')
        parts = line.split(', ')
        truth = float(parts[0])
        
        if truth <= 1.0:
            truthbin = 0
        else:
            bin = np_freqs_to_logbins(truth, True)
            if bin < 0 or bin >= LOGBIN_COUNT:
                truthbin = 0 #set out of bounds to silent
            else:
                truthbin = bin + 1 #0 is silent, 1-600 are bins (~0hz is silent in melody files)
        
        if parts[1]:
            line_freqs = [[float(f) for f in part.split(': ')] for part in parts[1:]]
            fs, amps = zip(*line_freqs)
        else:
            fs, amps = [], []
        
        logbins = np_freqs_to_logbins(np.array(fs, dtype=np.float32), True)
        
        amps = np.array(amps, dtype=np.float32)
        expanded = np.full((LOGBIN_COUNT,), math.log(min_amp_filter), dtype=np.float32)
        if amps.shape != (0,):
            
            amps[amps < min_amp_filter] = min_amp_filter
            amps = np.log(amps) #amplitude is logarithmic
 
            #sum calc
            expanded[logbins] = amps
        
            max_val = np.max(expanded)
            min_val = np.min(expanded)

            max_value = max(max_value, max_val) if max_value is not None else max_val 
            min_value = min(min_value, min_val) if min_value is not None else min_val 


        first_moment_sum += expanded
        second_moment_sum += expanded ** 2

        X = np.stack((logbins.astype(np.float32), amps.astype(np.float32)), axis=1).flatten()
        if X.shape[0] > max_logbins_in_line*2:
            raise ValueError('too many logbins: {data.shape[0]}')

        X = np.pad(X, (0, max_logbins_in_line*2 + 2 - X.shape[0]), constant_values=LOGBIN_COUNT)
        X[max_logbins_in_line*2 + 1] = time_index

        Y = np.zeros((1,), dtype=np.int32)
        Y[0] = truthbin
    
        Xb = X.tobytes()
        Yb = Y.tobytes()
    
        total_Xb.append(Xb)
        total_Yb.append(Yb)

    return b''.join(total_Xb), b''.join(total_Yb), first_moment_sum, second_moment_sum, len(freqs), max_value, min_value
    
#x structure is (num_samples, logbin_count + 1), y structure is (logbin_count + 1,)
#dataset[:, 0] is 
def prepare_dataset(d, out):
    freqsmel_files = []
    for root, dirs, files in os.walk(d):
        for f in files:
            ext = os.path.splitext(f)[1]
            if ext == '.freqsmel':
                freqsmel_files.append(os.path.join(root, f))
    
    freqsmel_files.sort()
    random.Random(1234).shuffle(freqsmel_files)
    print(freqsmel_files[0])

    #freqsmel_files = freqsmel_files[:3000]

    total_first_moment = None
    total_second_moment = None
    total_samples = 0
    total_max_value = None
    total_min_value = None

    with open(f'{out}_Y', 'wb') as out_y_fh:
        with open(f'{out}_X_{LOGBIN_COUNT}', 'wb') as out_fh:
            for i in range(0, len(freqsmel_files), 500):
                files = freqsmel_files[i : i + 500]
                result = process_map_(prepare_dataset_worker, files, max_workers=20, chunksize=1, do_multi=True)

                total_Xb, total_Yb, first_moments, second_moments, sample_num, max_val, min_val = zip(*result)

                if total_first_moment is not None:
                    total_first_moment += sum(first_moments)
                    total_second_moment += sum(second_moments)
                    total_max_value = max(total_max_value, max(max_val))
                    total_min_value = min(total_min_value, min(min_val))
                else:
                    total_first_moment = sum(first_moments)
                    total_second_moment = sum(second_moments)
                    total_max_value = max(max_val)
                    total_min_value = min(min_val)
                total_samples += sum(sample_num)

                out_fh.write(b''.join(total_Xb))
                out_y_fh.write(b''.join(total_Yb))

            total_first_moment /= total_samples
            total_second_moment = (total_second_moment / total_samples) - (total_first_moment ** 2)

            print(f'mean: {np.mean(total_first_moment)}, var: {np.mean(total_second_moment)}, max: {total_max_value}, min: {total_min_value}')

            out_fh.write(total_first_moment.astype(np.float32).tobytes())
            out_fh.write(total_second_moment.astype(np.float32).tobytes())
    

def csv_envelope(src, truth, dst):
    lineses = [None, None]
    for n, f in enumerate([src, truth]):
        data = open(f, 'r').readlines()
        
        if data[0][0] not in '0123456789.-':
            data = data[1:]

        lineses[n] = [tuple(float(i) for i in d.replace('\t', ',').replace(' ', ',').split(',')) for d in data]
    
    srclines, truthlines = lineses
    
    truthlines = dict(truthlines)
    
    aligned_lines = [None] * len(srclines)
    for i, (t,_, _, _) in enumerate(srclines):
        aligned_lines[i] = closest_float(list(truthlines.keys()), t, lambda x: (t, truthlines[x]), (t, 0.0))
        
    lines = [(*a[:1], a[1] if b[1] != 0 else b[1], a[2] if b[1] != 0 else 0, int(b[1] == 0)) for a,b in zip(srclines, aligned_lines)]
    
    open(dst, 'w').write('\n'.join(','.join(str(f) for f in l) for l in lines))
    
#training process:
#process_dirs(dir) would go over all wavs in dir recursively and create a new file .freqs
#add_melody_to_freqs(dir, out_zip) goes over all csv/mel files in dir recursively and matches ground truth to each line in every .freq file
#prepare_dataset(zip, out) expand each line in each freq file to 600 logbins, history of n ground truths (inputs) and the current ground truth as an integer between 0-600 including (600 logbins + quiet)

def main(): 
    #process('manoa.wav', 'manoa_out')
    #process('duke.wav', 'duke_out')
    #process('sound.wav', 'result')
    process_dirs('data')
    add_melody_to_freqs('data/')
    #prepare_dataset('data/', 'data/dataset2')
    #process_augment_dirs('data')
    
    process('manoa.wav', 'manoatest', output_wave=True)
    #process('allstar.wav', 'allstar512', output_wave=False)
    #process('data/beatles.wav', 'data/beatles512', output_wave=False)
    #process('manoa.wav', 'manoa512', output_wave=False)
    
    #csv_envelope('data/beatles.csv', 'data/MusicDelta_Beatles_MELODY2.csv', 'data/beatles_env.csv')
    #make_from_csv('data/MusicDelta_Beatles_ENV.csv')
    #make_from_csv('data/beatles.csv')
    make_from_csv('manoa.csv')
    #make_from_csv('allstar.csv')
    #process('data/phoenix.wav', 'data/phoenix512', output_wave=False)

# def read_freqs(file_path):
#     freq_amp_list = []
#     segments_counter = 1
#     with open(file_path, 'r') as file:
#         for line in file:
#             line = line.strip()
#             if line.isdigit(): #ignores empty lines and the sample_rate line (it is the first one)
#                 sr = int(line)
#                 time_delta = HOP_LENGTH / sr
#                 continue
#             elif not line:
#                 continue
#
#             parts = line.split(", ")
#             line_dict = {"freqs": [], "amps":[]}
#             for part in parts:
#                 freq, amp = part.strip().split(":")
#                 line_dict["freqs"].append(float(freq.strip()))
#                 line_dict["amps"].append(float(amp.strip()))
#             freq_amp_list.append({"time":segments_counter*time_delta, "line_freqs_amps":line_dict})
#             segments_counter +=1
#     return freq_amp_list
#
#
# def read_ground_truth(file_path):
#     df = pd.read_csv(file_path)
#     df.columns = ["time", "freq"]
#     return df

# def match_justing_freqs_to_ground_truth(justin, gt):
#     res_df = pd.DataFrame(columns=["gt_time", "justin_time","gt_freq","justin_freq"])
#     gap = 1000
#     row_index = 0
#     for jst in justin:
#         time = jst["time"]
#         new_row = {}
#         for i in range(row_index, len(gt)+1, 1):
#             cur_gt_row = gt.iloc[i]
#             time_diff = np.abs((time-cur_gt_row["time"]))
#             if time_diff < gap:
#                 gap = time_diff
#                 index_min = np.argmin(np.abs(np.array(jst["line_freqs_amps"]["freqs"])-cur_gt_row["freq"]))
#                 new_row = {"gt_time":cur_gt_row["time"],
#                            "justin_time":jst["time"],
#                            "gt_freq":cur_gt_row["freq"],
#                            "justin_freq":jst["line_freqs_amps"]["freqs"][index_min]}
#             else:
#                 gap=1000
#                 new_row = pd.DataFrame(new_row, index=[1])
#
#                 res_df = pd.concat([res_df, new_row], ignore_index=True)
#                 # res_df._append(new_row, ignore_index=True)
#                 row_index = i+1
#                 break



if __name__ == '__main__':
    path_wav = "C:\\Users\\Shahar\\Desktop\\PROJ\\AUG_TESTING_NEW\\BS_song_IR_try.wav"
    #path_csv = "/Users/tomermassas/Desktop/project melody extraction/datasets/medleyDB/AClassicEducation_NightOwl/AClassicEducation_NightOwl_MELODY2.csv"
    path_save = "../.."

    # make_from_csv(pa th,path_save)
    process(path_wav,path_save,"test_ir_song", output_wave=True)
    #freq_amp_list = read_freqs("C:\\Users\\Shahar\\Desktop\\PROJ\\GitHub_Repos\\test_fast_song.freqs")
    # ground_truth = read_ground_truth("/Users/tomermassas/Desktop/project melody extraction/datasets/medleyDB/AClassicEducation_NightOwl/AClassicEducation_NightOwl_MELODY2.csv")
    # match_justing_freqs_to_ground_truth(freq_amp_list, ground_truth)

