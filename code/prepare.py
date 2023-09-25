# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Description: Prepare the experimental settings
"""
import torch

def prep_env():
    # type: () -> dict
    """
    Desc:
        Prepare the experimental settings
    Returns:
        The initialized arguments
    """
    settings = {
        "path_to_phase1_x": "./data/sdwpf_baidukddcup2022_test_toy/test_x/0001in.csv",
        "path_to_test_x": "./data/sdwpf_baidukddcup2022_test_toy/test_x",
        "path_to_test_y": "./data/sdwpf_baidukddcup2022_test_toy/test_y",
        "data_path": "./data",
        "path_to_mean": "mean_folder",
        "path_to_std": "std_folder",
        "filename": "wtbdata_245days.csv",
        "actual_filename": "wtbdata_259days.csv",
        "use_new_data": False,
        "task": "MS",
        "target": "Patv",
        "checkpoints": "checkpoints",
        "turbine_id": 0,
        "input_len": 36, #length of input sequence
        "step_size": 1, # we take 1 from every n observations, so the actual input length is input_len / step_size
        "output_len": 288, #length of output sequence
        "columns": ["Day", "Tmstamp", "Wspd", "Wdir", "Etmp", "Patv"],
        "column_pos": [1, 2, 3, 4, 5, 12], # for `predict.py` evaluation use, since column names cannot be indexed in np.ndarray
        "start_col": 1, #'Index of the start column of the meaningful variables'
        "scale_cols": ["Wspd", "Wdir", "Etmp", "Patv"],
        "in_var": 6, #'Number of the input variables'
        "out_var": 1,  #'Number of the output variables'
        "day_len": 144,  #'Number of observations in one day'
        "train_days": 214,  #'Number of days for training'  ##train_size
        "actual_train_days": 228,
        "val_days": 31,  #'Number of days for validation'
        "actual_val_days": 31,
        "total_days": 245,  #'Number of days for the whole dataset'
        "actual_total_days": 259,
        "gru_hidden_size": 12, #8
        "gru_layers": 4, #2
        "dropout": 0.05,
        "num_workers": 0,  #'#workers for data loader'
        "train_epochs": 32,  #10
        "batch_size": 64, #'Batch size for the input training data'
        "patience": 12,
        "lr": 0.25e-4,#5e-4,  #'Optimizer learning rate'
        "lr_adjust": "type1", 
        "device": "cpu",
        "capacity": 134, #"The capacity of a wind farm, i.e. the number of wind turbines in a wind farm"
        "turbine_id": 0,
        "pred_file": "predict.py",
        "framework": "pytorch",
        "is_debug": False
    }

    # Prepare the GPUs
    if torch.cuda.is_available():
        settings["device"] = 0
    else:
        settings["device"] = 'cpu'
    print("The experimental settings are: \n{}".format(str(settings)))
    return settings
