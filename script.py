import os
import subprocess
import shutil
import csv
from preprocess.utility_functions import make_from_csv
from preprocess.Lifshits import process

# input_file = '../AClassicEducation_NightOwl_MELODY2.csv'
# output_file = 'output.csv'
#
# with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
#     reader = csv.reader(infile)
#     writer = csv.writer(outfile)
#
#     for row in reader:
#         if row:
#             row[0] = str(float(row[0]) / 1.2)
#             writer.writerow(row)

# Specify the path to the script you want to run
# script_to_run = "/Users/tomermassas/Documents/GitHub/melody_extraction/preprocess/melodia.py"
# source_csv_directory = "/Users/tomermassas/Desktop/prep_orchset/annotations"
#
# # Get the list of sub-folders in the parent folder
# # parent_folder = os.path.dirname(os.path.abspath(__file__))
# parent_folder = "/Users/tomermassas/Desktop/medleyDB_prep/songs"
# sub_folders = [folder for folder in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, folder))]
#
# # Iterate through the sub-folders and run the script with the folder name as an argument
# not_found_songs = []
# for i,folder in enumerate(sub_folders):
#     print(f"starting \"{folder}\"")
#     folder_path = os.path.join(parent_folder, folder)
#     script_path = os.path.join(parent_folder, script_to_run)
#     csv_file_name = f"{folder}.csv"
#     source_csv_path = os.path.join(source_csv_directory, csv_file_name)
#     destination_csv_path = os.path.join(folder_path, csv_file_name)
#     try:
#         shutil.copy(source_csv_path, destination_csv_path)
#
#         # Construct the command to run the script with the folder name as an argument
#         command = ["python3", script_path, folder]
#
#         # Run the script in the sub-folder
#         subprocess.run(command, cwd=folder_path)
#     except FileNotFoundError:
#         print(f" Song \"{folder}\" does not exist !!!")
#         not_found_songs.append(folder)
#         continue
#
# if not_found_songs:
#     with open("not_found_paths.txt", 'w') as output_file:
#         output_file.write("\n".join(not_found_songs))
#
# print("Done")

# path = "C:\\Users\Shahar\Desktop\PROJ\TEST_OTHERS\Melody-extraction-with-melodic-segnet\shape_of_my_heart_short.txt"
# path_to_save = "C:\\Users\Shahar\Desktop\PROJ\TEST_OTHERS\Melody-extraction-with-melodic-segnet"
# make_from_csv(path=path,path_to_save=path_to_save,name='ShapeofMyHeart_CFP')

wav_file_path = '../sirduke_short.wav'
freq_file_path = wav_file_path + '.freqs'
process(wav_file_path, path_to_save='.', outputname=wav_file_path,output_wave=False)