import pandas as pd
from preprocess.utility_functions import (make_audio, save_lines, read_ground_truth, read_freqs, plot_histogram,
                                          match_justing_freqs_to_ground_truth, analize_justin_with_ground_truth,
                                          make_from_many, make_from_csv)


if __name__ == '__main__':
    # path_freqs = "/Users/tomermassas/Desktop/AClassicEducation_NightOwl.freqs"
    # path_gt = "/Users/tomermassas/Desktop/project melody extraction/Datasets/MedlydB/freqs_labels_wavs/annotations_Melody2/MusicDelta_GriegTrolltog_MELODY2.csv"
    #
    # gt_df = read_ground_truth(path_gt)
    # freqs = read_freqs(path_freqs)
    #
    # matching_results = match_justing_freqs_to_ground_truth(freqs,gt_df)
    # analize_justin_with_ground_truth(matching_results)
    # plot_histogram(gt_df)

    song_names_path = "/Users/tomermassas/Desktop/project melody extraction/Datasets/MedlydB/freqs_labels_wavs/annotations_Melody2.txt"
    song_names = []
    with open(song_names_path, 'r') as file:
        for line in file:
            line = line.strip()
            song_names.append(line)

    path_freqs = "/Users/tomermassas/Desktop/project melody extraction/Datasets/MedlydB/freqs_labels_wavs/freqs"
    path_gt = "/Users/tomermassas/Desktop/project melody extraction/Datasets/MedlydB/freqs_labels_wavs/annotations_Melody2"
    s = 0
    for name in song_names:
        gt_df = read_ground_truth(f"{path_gt}/{name}_MELODY2.csv")
        freqs = read_freqs(f"{path_freqs}/{name}.freqs")
        for i,f in enumerate(freqs):
            s += round(f["time"] - gt_df.iloc[(2*i)+1]["time"], 4)
        # matching_results = match_justing_freqs_to_ground_truth(freqs, gt_df)
        # analize_justin_with_ground_truth(matching_results,name, path=f"/Users/tomermassas/Desktop/project melody extraction/Datasets/MedlydB/analyzing_preprocess/Error_distance/{name}")
        # plot_histogram(gt_df,name, path=f"/Users/tomermassas/Desktop/project melody extraction/Datasets/MedlydB/analyzing_preprocess/Melody_distribution/{name}")


    print()

