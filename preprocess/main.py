import pandas as pd
from scipy.io import wavfile
from preprocess.extract_freqs_amps import process_freqs_amps_extraction
from preprocess.utility_functions import (make_audio, save_lines, read_ground_truth, read_freqs, plot_histogram,
                                          match_justing_freqs_to_ground_truth, analize_justin_with_ground_truth,
                                          make_from_many, make_from_csv)
from Lifshits import prepare_dataset_worker



if __name__ == '__main__':
    # path_wav = "/Users/tomermassas/Desktop/project melody extraction/datasets/medleyDB/AlexanderRoss_GoodbyeBolero/AlexanderRoss_GoodbyeBolero_MIX.wav"
    # path_freqs_amps_justin = "/Users/tomermassas/Desktop/project melody extraction/tmppp/AClassicEducation_NightOwl_MIX.freqs"
    # path_ground_truth_csv = "/Users/tomermassas/Desktop/project melody extraction/datasets/annotations/Medly dB/Melody2/Auctioneer_OurFutureFaces_MELODY2.csv"
    # path_csv_analize = "/Users/tomermassas/Desktop/project melody extraction/datasets/medleyDB/AClassicEducation_NightOwl/compare_result.csv"

    # path_save = "/Users/tomermassas/Desktop/project melody extraction/datasets/medleyDB/Auctioneer_OurFutureFaces"
    # name = "AlexanderRoss_GoodbyeBolero_MELODY2"
    # make_from_csv(path_ground_truth_csv,path_save)
    result, sr = process_freqs_amps_extraaction(path_wav)
    # sr, x = wavfile.read(path_wav)
    # result = read_ground_truth(path_ground_truth_csv)
    # make_audio(result, sr, path_save, name, split_lines=False)
    # save_lines(result, sr, path_save, name)

    # # analize justin + save
    # gt = read_ground_truth(path_ground_truth_csv)
    # plot_histogram(gt, path_save)
    # justin = read_freqs(path_freqs_amps_justin)
    # res = match_justing_freqs_to_ground_truth(justin, gt)
    # res.to_csv(f"{path_save}/compare_result.csv")

    # plt histogram of analize
    # res = pd.read_csv(path_csv_analize)
    # analize_justin_with_ground_truth(res, path_save)
    p = "/Users/tomermassas/Desktop/project melody extraction/tmppp/AClassicEducation_NightOwl_MIX.freqs"
    prepare_dataset_worker(p)


    print()
