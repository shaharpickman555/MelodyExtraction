import matplotlib.pyplot as plt
import librosa
import numpy as np
import IPython.display as ipd
import sounddevice as sd
import tqdm
from scipy.io import wavfile
# from preprocess.extract_freqs_amps import HOP_LENGTH
import pandas as pd
import math
from preprocess import melodia
HOP_LENGTH = melodia.hop_length #about 10ms



def write_list_elements_to_files(myList:str):# Todo DELETE
    output_file = "/Users/tomermassas/Desktop/medlyDB/for_train/version_1/bads_4444.txt"
    with open(output_file, "w") as file:
        for elem in zip(myList):
            line = "".join(map(str, elem))
            file.write(line)
            file.write("\n")

def display_signal(signal, smaple_rate):
    plt.figure(figsize=(20, 5))
    librosa.display.waveshow(signal, sr=smaple_rate)
    plt.title("Waveplot", fontdict = dict(size=18))
    plt.xlabel("Time", fontdict = dict(size=15))
    plt.ylabel("Amplitude", fontdict = dict(size=15))
    plt.show()


def play_signal(signal, smaple_rate):
    sd.play(signal,smaple_rate)


def show_mel_spec(mel_spec_db, smaple_rate, hop_length, title="Mel Spectrogram"):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, sr=smaple_rate, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def make_audio(result, sr, path_to_save, name, split_lines=False):
    freq_threshold = 1
    many = HOP_LENGTH

    waves = []
    for l in tqdm.tqdm(range(result.shape[1])):
        line = result[:, l, :]
        wave = np.zeros((result.shape[0] * many,), dtype=np.float64)
        phase = 0
        for t, (f, a) in enumerate(line):
            if f < freq_threshold or a == 0:
                continue

            phase_add = f * 2 * np.pi * many / sr
            wave[HOP_LENGTH * t: (HOP_LENGTH * t) + many] = np.sin(
                np.linspace(phase, phase + phase_add, many, endpoint=False)) * a

            phase += phase_add

        if split_lines:
            normwave = wave * (32768.0 / np.max(np.abs(wave)))
            wavfile.write(f'{path_to_save}/{name}_{l}.wav', sr, normwave.astype(np.int16))

        waves.append(wave)

    wave = np.sum(waves, axis=0)
    normwave = wave * (32768.0 / np.max(np.abs(wave)))
    wavfile.write(f'{path_to_save}/{name}_compressed.wav', sr, normwave.astype(np.int16))


def save_lines(result, sr, path_to_save, name):
    data = '\n'.join(', '.join(f'{f:.8f}: {a:.8f}' for f,a in timeframe if f != 0 and a != 0) for timeframe in result)
    with open(f'{path_to_save}/{name}.freqs', 'w') as fh:
        fh.write(f'{sr}\n')
        fh.write(data)


def match_justing_freqs_to_ground_truth(justin, gt):
    res_df = pd.DataFrame(columns=["gt_time", "justin_time","gt_freq","justin_freq"])
    gap = 1000
    row_index = 0
    for jst in justin:
        time = jst["time"]
        new_row = {}
        for i in range(row_index, len(gt)+1, 1):
            cur_gt_row = gt.iloc[i]
            time_diff = np.abs((time-cur_gt_row["time"]))
            if time_diff < gap:
                gap = time_diff
                index_min = np.argmin(np.abs(np.array(jst["line_freqs_amps"]["freqs"])-cur_gt_row["freq"]))
                new_row = {"gt_time":cur_gt_row["time"],
                           "justin_time":jst["time"],
                           "gt_freq":cur_gt_row["freq"],
                           "justin_freq":jst["line_freqs_amps"]["freqs"][index_min]}
            else:
                gap=1000
                new_row = pd.DataFrame(new_row, index=[1])

                res_df = pd.concat([res_df, new_row], ignore_index=True)
                # res_df._append(new_row, ignore_index=True)
                row_index = i+1
                break
    return res_df

def read_ground_truth(file_path):
    df = pd.read_csv(file_path)
    df.columns = ["time", "freq"]
    return df

def read_freqs(file_path):
    freq_amp_list = []
    segments_counter = 1
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.isdigit(): #ignores empty lines and the sample_rate line (it is the first one)
                sr = int(line)
                time_delta = HOP_LENGTH / sr
                continue
            line_dict = {"freqs": [], "amps":[]}
            if (line != ""):
                parts = line.split(", ")
                for part in parts:
                    freq, amp = part.strip().split(":")
                    line_dict["freqs"].append(float(freq.strip()))
                    line_dict["amps"].append(float(amp.strip()))
            else:
                line_dict["freqs"].append(float(0))
                line_dict["amps"].append(float(0))
            freq_amp_list.append({"time":segments_counter*time_delta, "line_freqs_amps":line_dict})
            segments_counter +=1
    return freq_amp_list

def analize_justin_with_ground_truth(result_df, title,path=""):
    non_zero_l = []
    zero_l = []
    for i in range(len(result_df)):
        cur_row = result_df.iloc[i]
        if cur_row["gt_freq"] == 0:
            zero_l.append(cur_row["justin_freq"])
        else:
            non_zero_l.append(cur_row["gt_freq"]- cur_row["justin_freq"])

    # non_zero_l = np.array(non_zero_l)
    # zero_l = np.array(zero_l)
    # hist_nonzero, bin_edges_nonzero = np.histogram(non_zero_l, bins=100)
    # hist_zero, bin_edges_zero = np.histogram(zero_l, bins=100)
    fig = plt.figure()
    plt.hist(non_zero_l, bins=1000)
    plt.title(f"Distance from ground truth\n{title}")
    if path != "":
        fig.savefig(path, dpi=300)

def plot_histogram(array,title, path=""):
    array_l = array["freq"].values.tolist()
    array_l = [val for val in array_l if val !=0]
    fig = plt.figure()
    plt.hist(array_l, bins=1000)
    plt.title(f"Melody Frequency Distribution\n{title}")
    if path != "":
        fig.savefig(path, dpi=300)






def make_from_many(path_load, path_to_save, name):
    data = open(path_load, 'r').readlines()
    data = np.array([[(i, float(v)) for i,v in enumerate(line.split(','))] for line in data])

    for t in range(data.shape[0]):
        data[t, :, 0] = (2 ** ((data[t, :, 0].astype(np.float64) - 0.5) / (1200 / 10))) * 55

    sr = 44100
    make_audio(data, sr, path_to_save, name, split_lines=False)

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

def make_from_csv(path, path_to_save, name, debounce=1, silence_debounce=1, same_octave=False):
    data = open(path, 'r').readlines()

    if data[0][0] not in '0123456789.-':
        data = data[1:]

    freqs = [tuple(float(i) for i in d.replace('\t', ',').replace(' ', ',').split(',')) for d in data]

    confidence_threshold = 0.1
    silence_confidence_threshold = (0.5, 0.5)

    if len(freqs[0]) == 2:
        freqs = [(t, f, 1 if f > 0 else 0, 0 if f > 0 else 1) for t, f in freqs]
    elif len(freqs[0]) == 3:
        freqs = [(t, f, c if f > 0 else 0, 0 if f > 0 else 1) for t, f, c in freqs]

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
            # phase = 0
            real_f = last_good
        else:
            real_f = most([ff for _, ff, _, _ in freqs[max(0, i - debounce): i + 1]])
            if same_octave:
                real_f = f_to_220_440(real_f)

            last_good = real_f

        if not (silence_confidence_threshold[0] < s < silence_confidence_threshold[1]):
            real_silence = most(
                [ss >= silence_confidence_threshold[1] for _, _, _, ss in freqs[max(0, i - silence_debounce): i + 1]])

            last_silence = real_silence

        if last_silence:
            real_f = 0

        phase_add = real_f * 2 * np.pi * many / sr

        samples = np.linspace(phase, phase + phase_add, many, endpoint=False)

        phase = (phase + phase_add) % (2 * np.pi)

        out[start: start + many] = np.sin(samples)

    wavfile.write(f'{path_to_save}/{name}.wav', sr, out)

def song_names_txt_to_list(song_names_path:str):
    song_names = []
    with open(song_names_path, 'r') as file:
        for line in file:
            line = line.strip()
            song_names.append(line)
    return song_names