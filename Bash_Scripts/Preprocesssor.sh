#!/bin/bash

python3 /fast/gmooers/preprocess_RG.py --config_file /fast/gmooers/SPCAM5.yml --in_dir /DFS-L/DATA/pritchard/gmooers/Workflow/SPCAM_DATA/SPCAM5/2_Degree_Res/ --aqua_names TimestepOutput_Neuralnet_SPCAM_216.cam.h1.20[01][012345679]-[01][0123456789]*-[0123][1]-* --out_dir /fast/gmooers/Preprocessed_Data/7_Years_Spaced/ --out_pref full_physics_essentials_train_month01

python3 /fast/gmooers/preprocess_RG.py --config_file /fast/gmooers/SPCAM5.yml --in_dir /DFS-L/DATA/pritchard/gmooers/Workflow/SPCAM_DATA/SPCAM5/2_Degree_Res_Valid_Data/ --aqua_names TimestepOutput_Neuralnet_SPCAM_216.cam.h1.2018-[01][0123456789]*-[0123][1]-* --out_dir /fast/gmooers/Preprocessed_Data/7_Years_Spaced/ --out_pref full_physics_essentials_valid_month02 --ext_norm /fast/gmooers/Preprocessed_Data/7_Years_Spaced/full_physics_essentials_train_month01_norm.nc

python3 /fast/gmooers/shuffle_ds.py --method fast --pref /fast/gmooers/Preprocessed_Data/7_Years_Spaced/full_physics_essentials_train_month01 --chunk_size 10000000

