####################################################################################################################
""" create banks of songs that can be pitch shifted according to the min/max values"""
# def write_names_to_file(names, file_path):
#     with open(file_path, 'w') as file:
#         for name in names:
#             file.write(name + '\n')
# path_labels = "/Users/tomermassas/Desktop/project melody extraction/Datasets/medlyDB/labels"
#     song_names = song_names_txt_to_list("/Users/tomermassas/Desktop/project melody extraction/Datasets/medlyDB/medly_song_names.txt")
#
#     low_range = 53.43425676
#     high_range = 2453
#     semitone_fraction = 1
#     shift_factor = 1
#     min_freq = low_range * (2 ** (shift_factor / (12 * semitone_fraction)))
#     max_freq = high_range / (2 ** (shift_factor / (12 * semitone_fraction)))
#
#     up = []
#     down = []
#
#     for name in song_names:
#         GT = read_ground_truth(f"{path_labels}/{name}.csv")
#
#         freqs = sorted(set(GT["freq"]))
#         min_gt = freqs[1]
#         max_gt = freqs[-1]
#
#         if min_gt >= min_freq:
#             down.append(name)
#         if max_gt <= max_freq:
#             up.append(name)
#
#
#     path_save = "/Volumes/New Volume/Tomer/sorted_songs"
#     write_names_to_file(up, f"{path_save}/up_bank_{shift_factor}.txt")
#     write_names_to_file(down, f"{path_save}/down_bank_{shift_factor}.txt")
####################################################################################################################
""" split songs into banks of pitch feasible groups"""
# up_1 = song_names_txt_to_list("/Volumes/New Volume/Tomer/sorted_songs/up_bank_1.txt")
# up_2 = song_names_txt_to_list("/Volumes/New Volume/Tomer/sorted_songs/up_bank_2.txt")
# up_3 = song_names_txt_to_list("/Volumes/New Volume/Tomer/sorted_songs/up_bank_3.txt")
# down_1 = song_names_txt_to_list("/Volumes/New Volume/Tomer/sorted_songs/down_bank_1.txt")
# down_2 = song_names_txt_to_list("/Volumes/New Volume/Tomer/sorted_songs/down_bank_2.txt")
# down_3 = song_names_txt_to_list("/Volumes/New Volume/Tomer/sorted_songs/down_bank_3.txt")
#
# intr1 = set(up_1).intersection(set(down_2)).intersection(set(up_3))
# intr2 = set(down_1).intersection(set(up_2)).intersection(set(down_3))
# intrall = intr1.intersection(intr2)
#
# g1 = list(intrall)[:48]
# g2 = list(intrall)[48:]
#
#
####### plus and minus 1 groups
# intr_1 = set(up_1).intersection(set(down_1))
# for name in intr_1:
#         up_1.remove(name)
#         down_1.remove(name)
# up_1 = up_1 + g1
# down_1 = down_1 + g2
#
# diff1 = intr_1.difference(set(up_1+down_1))
# up_1 = up_1 + list(diff1)[:1]
# down_1 = down_1 + list(diff1)[1:]
# path_save = "/Volumes/New Volume/Tomer/sorted_songs"
# write_names_to_file(up_1, f"{path_save}/up_final_1.txt")
# write_names_to_file(down_1, f"{path_save}/down_final_1.txt")
#
#
# ####### plus and minus 2 groups
# intr_2 = set(up_2).intersection(set(down_2))
# for name in intr_2:
#         up_2.remove(name)
#         down_2.remove(name)
# up_2 = up_2 + g1
# down_2 = down_2 + g2
#
# diff2 = intr_2.difference(set(up_2+down_2))
# down_2 = down_2 + list(diff2)
#
# path_save = "/Volumes/New Volume/Tomer/sorted_songs"
# write_names_to_file(up_2, f"{path_save}/up_final_2.txt")
# write_names_to_file(down_2, f"{path_save}/down_final_2.txt")
#
#
# ####### plus and minus 3 groups
# intr_3 = set(up_3).intersection(set(down_3))
# for name in intr_3:
#         up_3.remove(name)
#         down_3.remove(name)
# up_3 = up_3 + g1
# down_3 = down_3 + g2
#
# diff3 = intr_3.difference(set(up_3+down_3))
#
# path_save = "/Volumes/New Volume/Tomer/sorted_songs"
# write_names_to_file(up_3, f"{path_save}/up_final_3.txt")
# write_names_to_file(down_3, f"{path_save}/down_final_3.txt")
####################################################################################################################
""" split into banks for timestretch"""
# song_names = song_names_txt_to_list(
#     "/Users/tomermassas/Desktop/project melody extraction/Datasets/medlyDB/medly_song_names.txt")
# g1 = song_names[::2]
# g2 = song_names[1::2]
# write_names_to_file(g1, f"/Volumes/New Volume/Tomer/sorted_songs/timestretch/group_1.txt")
# write_names_to_file(g2, f"/Volumes/New Volume/Tomer/sorted_songs/timestretch/group_2.txt")
####################################################################################################################
""" make wavs for all augmentations"""
###### timestretch + pitch + noise + gain
# path_wavs = "/Users/tomermassas/Desktop/project melody extraction/Datasets/medlyDB/wavs"
# path_names = "/Volumes/New Volume/Tomer/sorted_songs/timestretch"
# path_saves = "/Volumes/New Volume/Tomer"
#
# make_wav_aug(aug_all(-1, 0.94), f"{path_names}/group_2.txt", path_wavs, f"{path_saves}/aug_TS_P_N_G_1/wavs")
# make_wav_aug(aug_all(1, 1.06),  f"{path_names}/group_1.txt", path_wavs, f"{path_saves}/aug_TS_P_N_G_1/wavs")
#
# make_wav_aug(aug_all(2, 1.12), f"{path_names}/group_2.txt", path_wavs, f"{path_saves}/aug_TS_P_N_G_2/wavs")
# make_wav_aug(aug_all(-2, 0.88),  f"{path_names}/group_1.txt", path_wavs, f"{path_saves}/aug_TS_P_N_G_2/wavs")
#
#
# ###### noise + gain
# # path_wavs = "/Users/tomermassas/Desktop/project melody extraction/Datasets/medlyDB/wavs"
# # path_names = "/Users/tomermassas/Desktop/project melody extraction/Datasets/medlyDB/medly_song_names.txt"
# # path_saves = "/Volumes/New Volume/Tomer/aug_noise_gain/wavs"
# #
# # make_wav_aug(aug_noise_gain(), path_names, path_wavs, path_saves)
#
#
# ###### Timestretch
# # path_wavs = "/Users/tomermassas/Desktop/project melody extraction/Datasets/medlyDB/wavs"
# # path_names = "/Volumes/New Volume/Tomer/sorted_songs/timestretch"
# # path_saves = "/Volumes/New Volume/Tomer"
# #
# # make_wav_aug(aug_timestretch(0.94), f"{path_names}/group_1.txt", path_wavs, f"{path_saves}/aug_timestretch_1/wavs")
# # make_wav_aug(aug_timestretch(1.06), f"{path_names}/group_2.txt", path_wavs, f"{path_saves}/aug_timestretch_1/wavs")
# #
# # make_wav_aug(aug_timestretch(1.12), f"{path_names}/group_1.txt", path_wavs, f"{path_saves}/aug_timestretch_2/wavs")
# # make_wav_aug(aug_timestretch(0.88), f"{path_names}/group_2.txt", path_wavs, f"{path_saves}/aug_timestretch_2/wavs")
# #
# # make_wav_aug(aug_timestretch(0.82), f"{path_names}/group_1.txt", path_wavs, f"{path_saves}/aug_timestretch_3/wavs")
# # make_wav_aug(aug_timestretch(1.18), f"{path_names}/group_2.txt", path_wavs, f"{path_saves}/aug_timestretch_3/wavs")
#
#
# ##### Pitch
# # path_wavs = "/Users/tomermassas/Desktop/project melody extraction/Datasets/medlyDB/wavs"
# # path_names = "/Volumes/New Volume/Tomer/sorted_songs"
# # path_saves = "/Volumes/New Volume/Tomer"
# # make_wav_aug(aug_pitch(1),  f"{path_names}/up_final_1.txt", path_wavs,   f"{path_saves}/aug_pitch_1/wavs")
# # make_wav_aug(aug_pitch(-1), f"{path_names}/down_final_1.txt", path_wavs, f"{path_saves}/aug_pitch_1/wavs")
# #
# # make_wav_aug(aug_pitch(2),  f"{path_names}/up_final_2.txt", path_wavs,   f"{path_saves}/aug_pitch_2/wavs")
# # make_wav_aug(aug_pitch(-2), f"{path_names}/down_final_2.txt", path_wavs, f"{path_saves}/aug_pitch_2/wavs")
# #
# # make_wav_aug(aug_pitch(3),  f"{path_names}/up_final_3.txt", path_wavs,   f"{path_saves}/aug_pitch_3/wavs")
# # make_wav_aug(aug_pitch(-3), f"{path_names}/down_final_3.txt", path_wavs, f"{path_saves}/aug_pitch_3/wavs")
####################################################################################################################
"""augment pitch"""
    # up = song_names_txt_to_list("/Users/tomermassas/Desktop/augmentations/songs_sorted/final/up_1_final_54.txt")
    # down = song_names_txt_to_list("/Users/tomermassas/Desktop/augmentations/songs_sorted/final/down_1_final_52.txt")
    #
    # path_save_labels = "/Users/tomermassas/Desktop/augmentations/pitch/labels/p3"
    # path_save_input  = "/Users/tomermassas/Desktop/augmentations/pitch/input/p3"
    #
    # for i, l in zip([1, -1], [up, down]):
    #     val = i
    #     for name in l:
    #         labels = np.load(f"/Users/tomermassas/Desktop/labels_org/{name}.npy")
    #         new_labels = add_pitch_to_labels(labels, value_to_add=val)
    #         if new_labels.max() >= 68:
    #             print("ERROR")
    #         np.save(f"{path_save_labels}/{name}.npy", new_labels)
    #
    #         input = np.load(f"/Users/tomermassas/Desktop/input_org/{name}.npy")
    #         new_input = add_pitch_to_input(input, shift=val)
    #         np.save(f"{path_save_input}/{name}.npy", new_input)
####################################################################################################################
"""check input VS labels same size"""
# path_input = "/Volumes/New Volume/Tomer/medlyDB/Train/version_3/augmentations/Pitch/final/input/p3"
#     path_labels = "/Volumes/New Volume/Tomer/medlyDB/Train/version_3/augmentations/Pitch/final/labels/p3"
#     song_names = song_names_txt_to_list("/Users/tomermassas/Desktop/project melody extraction/Datasets/medlyDB/medly_song_names.txt")
#     for name in song_names:
#         input = np.load(f"{path_input}/{name}.npy")
#         labels = np.load(f"{path_labels}/{name}.npy")
#         diff = input.shape[0]- labels.shape[0]
#         if diff != 0:
#             print(f"Error --> {name}")
#     print()
####################################################################################################################
"""make input and labels final"""
# path_labels = "/Users/tomermassas/Desktop/project melody extraction/Datasets/medlyDB/labels"
#
# ################################################################################################################################################################
# make_input_labels_normal(path_labels=path_labels,
#                          path_freqs="/Volumes/New Volume/Tomer/original/freqs",
#                          path_names="/Users/tomermassas/Desktop/project melody extraction/Datasets/medlyDB/medly_song_names.txt",
#                          path_save="/Volumes/New Volume/Tomer/original"
#                          )
#
# make_input_labels_normal(path_labels=path_labels,
#                          path_freqs="/Volumes/New Volume/Tomer/aug_noise_gain/freqs",
#                          path_names="/Users/tomermassas/Desktop/project melody extraction/Datasets/medlyDB/medly_song_names.txt",
#                          path_save="/Volumes/New Volume/Tomer/aug_noise_gain"
#                          )
# ################################################################################################################################################################
# make_input_labels_timestretch(path_labels=path_labels,
#                               path_freqs="/Volumes/New Volume/Tomer/aug_timestretch_1/freqs",
#                               path_names="/Volumes/New Volume/Tomer/sorted_songs/timestretch/group_1.txt",
#                               path_save="/Volumes/New Volume/Tomer/aug_timestretch_1",
#                               rate=0.94)
# make_input_labels_timestretch(path_labels=path_labels,
#                               path_freqs="/Volumes/New Volume/Tomer/aug_timestretch_1/freqs",
#                               path_names="/Volumes/New Volume/Tomer/sorted_songs/timestretch/group_2.txt",
#                               path_save="/Volumes/New Volume/Tomer/aug_timestretch_1",
#                               rate=1.06)
#
# make_input_labels_timestretch(path_labels=path_labels,
#                               path_freqs="/Volumes/New Volume/Tomer/aug_timestretch_2/freqs",
#                               path_names="/Volumes/New Volume/Tomer/sorted_songs/timestretch/group_1.txt",
#                               path_save="/Volumes/New Volume/Tomer/aug_timestretch_2",
#                               rate=1.12)
# make_input_labels_timestretch(path_labels=path_labels,
#                               path_freqs="/Volumes/New Volume/Tomer/aug_timestretch_2/freqs",
#                               path_names="/Volumes/New Volume/Tomer/sorted_songs/timestretch/group_2.txt",
#                               path_save="/Volumes/New Volume/Tomer/aug_timestretch_2",
#                               rate=0.88)
#
# make_input_labels_timestretch(path_labels=path_labels,
#                               path_freqs="/Volumes/New Volume/Tomer/aug_timestretch_3/freqs",
#                               path_names="/Volumes/New Volume/Tomer/sorted_songs/timestretch/group_1.txt",
#                               path_save="/Volumes/New Volume/Tomer/aug_timestretch_3",
#                               rate=0.82)
# make_input_labels_timestretch(path_labels=path_labels,
#                               path_freqs="/Volumes/New Volume/Tomer/aug_timestretch_3/freqs",
#                               path_names="/Volumes/New Volume/Tomer/sorted_songs/timestretch/group_2.txt",
#                               path_save="/Volumes/New Volume/Tomer/aug_timestretch_3",
#                               rate=1.18)
# ################################################################################################################################################################
# make_input_labels_pitch(path_labels=path_labels,
#                         path_freqs="/Volumes/New Volume/Tomer/aug_pitch_1/freqs",
#                         path_names="/Volumes/New Volume/Tomer/sorted_songs/pitch/up_final_1.txt",
#                         path_save="/Volumes/New Volume/Tomer/aug_pitch_1",
#                         pitch=1)
# make_input_labels_pitch(path_labels=path_labels,
#                         path_freqs="/Volumes/New Volume/Tomer/aug_pitch_1/freqs",
#                         path_names="/Volumes/New Volume/Tomer/sorted_songs/pitch/down_final_1.txt",
#                         path_save="/Volumes/New Volume/Tomer/aug_pitch_1",
#                         pitch=-1)
#
# make_input_labels_pitch(path_labels=path_labels,
#                         path_freqs="/Volumes/New Volume/Tomer/aug_pitch_2/freqs",
#                         path_names="/Volumes/New Volume/Tomer/sorted_songs/pitch/up_final_2.txt",
#                         path_save="/Volumes/New Volume/Tomer/aug_pitch_2",
#                         pitch=2)
# make_input_labels_pitch(path_labels=path_labels,
#                         path_freqs="/Volumes/New Volume/Tomer/aug_pitch_2/freqs",
#                         path_names="/Volumes/New Volume/Tomer/sorted_songs/pitch/down_final_2.txt",
#                         path_save="/Volumes/New Volume/Tomer/aug_pitch_2",
#                         pitch=-2)
#
# make_input_labels_pitch(path_labels=path_labels,
#                         path_freqs="/Volumes/New Volume/Tomer/aug_pitch_3/freqs",
#                         path_names="/Volumes/New Volume/Tomer/sorted_songs/pitch/up_final_3.txt",
#                         path_save="/Volumes/New Volume/Tomer/aug_pitch_3",
#                         pitch=3)
# make_input_labels_pitch(path_labels=path_labels,
#                         path_freqs="/Volumes/New Volume/Tomer/aug_pitch_3/freqs",
#                         path_names="/Volumes/New Volume/Tomer/sorted_songs/pitch/down_final_3.txt",
#                         path_save="/Volumes/New Volume/Tomer/aug_pitch_3",
#                         pitch=-3)
# ################################################################################################################################################################
# make_input_labels_timestretch_pitch(path_labels=path_labels,
#                                     path_freqs="/Volumes/New Volume/Tomer/aug_TS_P_N_G_1/freqs",
#                                     path_names="/Volumes/New Volume/Tomer/sorted_songs/timestretch/group_1.txt",
#                                     path_save="/Volumes/New Volume/Tomer/aug_TS_P_N_G_1",
#                                     rate=1.06,
#                                     pitch=1)
# make_input_labels_timestretch_pitch(path_labels=path_labels,
#                                     path_freqs="/Volumes/New Volume/Tomer/aug_TS_P_N_G_1/freqs",
#                                     path_names="/Volumes/New Volume/Tomer/sorted_songs/timestretch/group_2.txt",
#                                     path_save="/Volumes/New Volume/Tomer/aug_TS_P_N_G_1",
#                                     rate=0.94,
#                                     pitch=-1)
#
# make_input_labels_timestretch_pitch(path_labels=path_labels,
#                                     path_freqs="/Volumes/New Volume/Tomer/aug_TS_P_N_G_2/freqs",
#                                     path_names="/Volumes/New Volume/Tomer/sorted_songs/timestretch/group_1.txt",
#                                     path_save="/Volumes/New Volume/Tomer/aug_TS_P_N_G_2",
#                                     rate=0.88,
#                                     pitch=-2)
# make_input_labels_timestretch_pitch(path_labels=path_labels,
#                                     path_freqs="/Volumes/New Volume/Tomer/aug_TS_P_N_G_2/freqs",
#                                     path_names="/Volumes/New Volume/Tomer/sorted_songs/timestretch/group_2.txt",
#                                     path_save="/Volumes/New Volume/Tomer/aug_TS_P_N_G_2",
#                                     rate=1.12,
#                                     pitch=2)



