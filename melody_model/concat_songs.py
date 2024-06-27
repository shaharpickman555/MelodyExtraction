from DataLoader import DatasetMelody
import torch
from melody_model.ConfigModel import ModelConfig
import numpy as np


config = ModelConfig()

def chain_songs(dataset_name = "medly", version="version_5"):

    song_names_path = f"../dataset/{dataset_name}/song_names_{dataset_name}.txt"

    augs_names_list = ["original", "aug_pitch_1", "aug_pitch_2", "aug_pitch_3", "aug_timestretch_1", "aug_timestretch_2",
                  "aug_timestretch_3", "aug_noise_gain", "aug_TS_P_N_G_1", "aug_TS_P_N_G_2"]

    for aug_name in augs_names_list:
        path_labels = f"../dataset/{dataset_name}/{version}/{aug_name}/label/"
        path_inputs = f"../dataset/{dataset_name}/{version}/{aug_name}/input/"
        song_names = []
        with open(song_names_path, 'r') as file:
            for line in file:
                line = line.strip()
                song_names.append(line)

        pad_input = np.zeros(shape=(config.block_size, config.n_freq_bins))
        pad_labels = np.zeros(shape=(config.block_size))

        concatenated_input = np.zeros(shape=(config.block_size, config.n_freq_bins))
        concatenated_labels = np.zeros(shape=(config.block_size))

        for name in song_names:
            cur_label = np.load(f"{path_labels}/{name}.npy")
            if cur_label.max() > config.n_freq_bins - 1:
                tmp = cur_label.max()
                indexes = np.where(cur_label > config.n_freq_bins - 1)[0]
                cur_label[indexes] = config.n_freq_bins - 1
            concatenated_labels = np.concatenate([concatenated_labels, pad_labels, cur_label], axis=0)

            cur_input = np.load(f"{path_inputs}/{name}.npy")
            concatenated_input = np.concatenate([concatenated_input, pad_input, cur_input], axis=0)


        if aug_name == "original" and dataset_name=="medly": # enter here only when splitting to val and train
            concatenated_labels = torch.from_numpy(np.concatenate([concatenated_labels, pad_labels], axis=0))
            length = concatenated_labels.size(dim=0)
            len_train = int(0.8 * length)
            labels_train = concatenated_labels[:len_train]
            labels_val = concatenated_labels[len_train:]
            torch.save(labels_train,f"../dataset/{dataset_name}/{version}/train/labels/labels_{aug_name}.pt")
            torch.save(labels_val,f"../dataset/validation/{version}/validation_labels.pt")

            concatenated_input = torch.from_numpy(np.concatenate([concatenated_input, pad_input], axis=0))
            input_train = concatenated_input[:len_train]
            input_val = concatenated_input[len_train:]
            torch.save(input_train,f"../dataset/{dataset_name}/{version}/train/input/input_{aug_name}.pt")
            torch.save(input_val,f"../dataset/validation/{version}/validation_input.pt")

        else:
            concatenated_labels = torch.from_numpy(np.concatenate([concatenated_labels, pad_labels], axis=0))
            torch.save(concatenated_labels, f"../dataset/{dataset_name}/{version}/train/labels/labels_{aug_name}.pt")

            concatenated_input = torch.from_numpy(np.concatenate([concatenated_input, pad_input], axis=0))
            torch.save(concatenated_input, f"../dataset/{dataset_name}/{version}/train/input/input_{aug_name}.pt")


def concat_all_augs(path_load, is_log:bool):
    types = ["input", "log_label"] if is_log else ["input", "label"]
    for type_ in types:
        org          = torch.load(f"{path_load}/log_{type_}/original.pt")#.type(torch.float16)
        aug_pitch_p1 = torch.load(f"{path_load}/log_{type_}/aug_pitch_1.pt")#.type(torch.float16)
        aug_pitch_p2 = torch.load(f"{path_load}/log_{type_}/aug_pitch_2.pt")#.type(torch.float16)
        aug_pitch_p3 = torch.load(f"{path_load}/log_{type_}/aug_pitch_3.pt")#.type(torch.float16)
        aug_ts_p1    = torch.load(f"{path_load}/log_{type_}/aug_timestretch_1.pt")#.type(torch.float16)
        aug_ts_p2    = torch.load(f"{path_load}/log_{type_}/aug_timestretch_2.pt")#.type(torch.float16)
        aug_ts_p3    = torch.load(f"{path_load}/log_{type_}/aug_timestretch_3.pt")#.type(torch.float16)
        aug_ng       = torch.load(f"{path_load}/log_{type_}/aug_noise_gain.pt")#.type(torch.float16)
        aug_mix_p1   = torch.load(f"{path_load}/log_{type_}/aug_TS_P_N_G_1.pt")#.type(torch.float16)
        aug_mix_p2   = torch.load(f"{path_load}/log_{type_}/aug_TS_P_N_G_2.pt")#.type(torch.float16)

        all = torch.cat((org,aug_ng, aug_pitch_p1, aug_pitch_p2, aug_pitch_p3, aug_ts_p1, aug_ts_p2, aug_ts_p3, aug_mix_p1, aug_mix_p2), dim=0)
        # all[all<0] = 0

        save_path = f"{path_load}/{type_}_log_all.pt" if is_log else f"{path_load}/{type_}_all.pt"
        torch.save(all, save_path)

def chain_all_datasets(is_log:bool, version="version_5"):
    types = ["input", "log_labels"] if is_log else ["input", "labels"]
    for type_ in types:
        medly   = torch.load(f"../dataset/medly/{version}/train/{type_}_all.pt")
        mirex   = torch.load(f"../dataset/mirex/{version}/train/{type_}_all.pt")
        orchset = torch.load(f"../dataset/orchset/{version}/train/{type_}_all.pt")
        all = torch.cat(((medly,mirex,orchset)))
        torch.save(all, f"/home/tomer.massas/docker_melody/melody_extraction/dataset/Train/{version}/{type_}_model.pt")


def apply_log(path):
    aug = torch.load(f"{path}")
    aug_log = torch.log(aug)
    aug_log[aug_log<0] = 0
    return aug_log.type(torch.float16)


def convert_to_log_and_save(path_load):
    augs_names = ["original", "aug_pitch_1", "aug_pitch_2", "aug_pitch_3", "aug_timestretch_1", "aug_timestretch_2",
                  "aug_timestretch_3", "aug_noise_gain", "aug_TS_P_N_G_1", "aug_TS_P_N_G_2"]
    for name in augs_names:
        aug_logged = apply_log(f"{path_load}/input/input_{name}.pt")
        torch.save(aug_logged, f"{path_load}/log_input/input_{name}.pt")


def convert_val_to_log_and_save(version):
    aug = torch.load(f"../dataset/validation/{version}/validation_input.pt")
    aug_log = torch.log(aug)
    aug_log[aug_log < 0] = 0
    torch.save(aug_log.type(torch.float16), f"../dataset/validation/{version}/validation_log_input.pt")


if __name__ == "__main__":

    dataset_name = "orchset"
    version_run = "version_5"

    #1)
    chain_songs(dataset_name, version_run)

    #2) Optional !!!
    convert_to_log_and_save(f"../dataset/{dataset_name}/{version_run}/train")
    convert_val_to_log_and_save(version_run)

    #3)
    concat_all_augs(f"../dataset/{dataset_name}/{version_run}/train")

    print()


