import numpy as np
from torch.utils.data import Dataset
import torch
from melody_model.ConfigModel import ModelConfig
import torch.nn.functional as F


class DatasetMelody(Dataset):
    config = ModelConfig()
    # def chain_all_datasets(self, type_):
    #     medly = torch.load(f"../dataset/medly/version_4/train/{type_}_all.pt")
    #     mirex = torch.load(f"../dataset/mirex/version_4/train/{type_}_all.pt")
    #     orchset = torch.load(f"../dataset/orchset/version_4/train/{type_}_all.pt")
    #     return torch.cat(((medly, mirex, orchset)))

    def __init__(self, run_type: str, device):
        """ run_type specify the data we want to load, can be ["train", "val", "test"]"""
        self.device = device

        if run_type == "train":
            self.X_ = torch.load("../dataset/train_data_NEW.pt")
            self.Y_ = torch.load("../dataset/train_labels_NEW.pt")
            self.Y_[self.Y_ < 0] = 0
            self.Y_[self.Y_ > 663] = 663
            print("\n|--- loaded train ---|")

        if run_type == "val":
            self.X_ = torch.load(f"../dataset/val_data_NEW.pt")
            self.Y_ = torch.load(f"../dataset/val_labels_NEW.pt")
            self.Y_[self.Y_ < 0] = 0
            self.Y_[self.Y_ > 663] = 663
            print("|--- loaded val ---|")

    def __len__(self):
        """returns the size of the set (train/val/test)"""
        return self.Y_.shape[0] - self.config.block_size

    def __getitem__(self, idx):
        x = self.X_[idx:idx + self.config.block_size].type(torch.float32)
        y = self.Y_[idx:idx + self.config.block_size]

        return x, y








################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################



# def increase(val):
#     x = torch.ones(65)*(-1)
#     x[val] = val
#     return x
#
#
# class DatasetShakespeare(Dataset):  # custom Dataset class must implement three functions: __init__, __len__, __getitem__
#     config = ModelConfig()
#     def load_dataset(self):
#         with open('./input.txt', 'r', encoding='utf-8') as f:
#             text = f.read()
#
#         # here are all the unique characters that occur in this text
#         chars = sorted(list(set(text)))
#         vocab_size = len(chars)
#         # create a mapping from characters to integers
#         stoi = {ch: increase(i) for i, ch in enumerate(chars)}
#         itos = {increase(i): ch for i, ch in enumerate(chars)}
#         encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
#         decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string
#
#         # Encode dataset
#         # my_dataset = torch.tensor(encode(text), dtype=torch.long)
#         my_dataset = torch.stack(encode(text))
#
#         # define the size of train/val/test sets
#         n_total = len(my_dataset)
#         n_train = int(0.8 * len(my_dataset))  # first 90% will be train, rest val
#         n_val = int(0.2 * len(my_dataset))
#         n_test = n_total - n_val - n_train
#
#         # split the dataset to train/val/test sets
#         train_data = my_dataset[:n_train]
#         val_data = my_dataset[n_train:n_train + n_val]
#         test_data = my_dataset[n_train + n_val:]
#
#         return train_data, val_data, test_data
#
#     def __init__(self, run_type: str, device):
#         """ run_type specify the data we want to load, can be ["train", "val", "test"]"""
#         full_dataset = self.load_dataset()
#
#         self.train_data = full_dataset[0]
#         self.val_data = full_dataset[1]
#         self.test_data = full_dataset[2]
#
#         self.indices = full_dataset[["train", "val", "test"].index(run_type)]
#         self.device = device
#
#     def __len__(self):
#         """returns the size of the set (train/val/test)"""
#         return self.indices.shape[0] - self.config.block_size
#
#     def __getitem__(self, idx):
#         x = self.indices[idx:idx + self.config.block_size]
#         y = self.indices[idx + 1:idx + self.config.block_size + 1].max(axis=1)[0].type(torch.LongTensor)
#
#         # if x.size()[0] != self.config.block_size:
#         #     pad_x = self.config.block_size - x.size()[0]
#         #     x = F.pad(x, (0, pad_x), mode="constant", value=1)
#         #
#         # if y.size()[0] != self.config.block_size:
#         #     pad_y = self.config.block_size - y.size()[0]
#         #     y = F.pad(y, (0, pad_y), mode="constant", value=1)
#         return x, y