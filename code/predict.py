import os
import time
import numpy as np
import torch
from model_rnn import BaselineRNNModel
from model_gru import BaselineGRUModel
from test_data import TestData
from common import Experiment

def inverse_transform(data, mean, std):
        # type: (torch.tensor, torch.tensor, torch.tensor) -> torch.tensor
        """
        Desc:
            Restore to the original data
        Args:
            data: the transformed data
        Returns:
            The original data
        """
        mean = mean.numpy() if not torch.is_tensor(data) else mean
        std = std.numpy() if not torch.is_tensor(data) else std
        data = data * std[-1] + mean[-1]
        return data

def transform(data, mean, std):

    mean = mean.numpy() if not torch.is_tensor(data) else mean
    std = std.numpy() if not torch.is_tensor(data) else std
    return (data - mean) / std

def func_add_t(x):
    time_strip = 600
    time_obj = time.strptime(x, "%H:%M")
    time_e = ((
        (time_obj.tm_sec + time_obj.tm_min * 60 + time_obj.tm_hour * 3600)) //
              time_strip) % 288
    return time_e

def forecast_one(experiment, test_turbines):
    # type: (Experiment, TestData) -> np.ndarray
    """
    Desc:
        Forecasting the power of one turbine
    Args:
        experiment:
        test_turbines:
        train_data:
    Returns:
        Prediction for one turbine
    """
    args = experiment.get_args()
    device = args["device"]
    tid = args["turbine_id"]
    
    dirs = args["checkpoints"].split('/')
    tmp_dir = dirs[0:dirs.index('checkpoints')]
    res_dir = ""#'/'
    for i in tmp_dir:
        res_dir = os.path.join(res_dir, i)

    # initialize model
    model_gru = BaselineGRUModel(args)
    model_rnn = BaselineRNNModel(args)

    # model path
    model_gru_dir = 'gru'
    model_rnn_dir = 'rnn'
    path_to_gru_model = os.path.join(args["checkpoints"], model_gru_dir, "model_{}".format(str(tid)))
    path_to_rnn_model = os.path.join(args["checkpoints"], model_rnn_dir, "model_{}".format(str(tid)))

    # load model dict
    state_dict_gru = torch.load(path_to_gru_model)
    state_dict_rnn = torch.load(path_to_rnn_model)
    model_gru.load_state_dict(state_dict_gru)
    model_rnn.load_state_dict(state_dict_rnn)

    # to device and switch to eval() mode
    model_gru.to(device)
    model_rnn.to(device)
    model_gru.eval() 
    model_rnn.eval()

    test_x, test_df = test_turbines.get_turbine(tid)
    args["relative_pos"] = [i - 1 for i in args["column_pos"]]
    # step 1: get desired features for prediction
    test_x = test_x[:, args["relative_pos"]]
    
    # step 2: process `Tmstamp` and apply cos function
    test_x[:, 1] = list(map(func_add_t, test_x[:, 1]))
    test_x[:, 0] = np.cos(test_x[:, 0].astype(np.float32)*2*np.pi / 365)
    test_x[:, 1] = np.cos(test_x[:, 1].astype(np.float32)*2*np.pi / 144)

    test_x = np.array(test_x, dtype=np.float32)

    path_to_mean = os.path.join(res_dir, args["path_to_mean"], "mean{}.pt".format(tid))
    path_to_std = os.path.join(res_dir, args["path_to_std"], "std{}.pt".format(tid))

    mean = torch.load(path_to_mean)
    std = torch.load(path_to_std)

    test_x_Day = test_x[:, 0].reshape(-1, 1)
    test_x_time = test_x[:, 1].reshape(-1, 1)
    test_x_forstd = test_x[:, 2:]
    test_x = transform(test_x_forstd, mean, std)
    test_x = np.hstack((test_x_Day, np.hstack((test_x_time, test_x))))

    last_observ = test_x[-args["input_len"]::args["step_size"]]
    seq_x = torch.tensor(last_observ)
    sample_x = np.reshape(seq_x, [-1, seq_x.shape[-2], seq_x.shape[-1]])
    prediction_gru = experiment.inference_one_sample(model_gru, sample_x)
    prediction_rnn = experiment.inference_one_sample(model_rnn, sample_x)

    prediction_gru = inverse_transform(prediction_gru, mean, std)
    prediction_rnn = inverse_transform(prediction_rnn, mean, std)
 
    # clip
    prediction_gru[prediction_gru < 0] = 0
    prediction_gru[prediction_gru > 1500] = 1500
    prediction_rnn[prediction_rnn < 0] = 0
    prediction_rnn[prediction_rnn > 1500] = 1500

    prediction_std = np.std((prediction_gru-prediction_rnn).cpu().detach().numpy(), axis=0)
   
    prediction = prediction_gru * 0.5 + prediction_rnn * 0.5
  
    prediction = prediction.cpu().detach().numpy()
    return prediction

def forecast(settings):
    # type: (dict) -> np.ndarray
    """
    Desc:
        Forecasting the wind power in a naive distributed manner
    Args:
        settings:
    Returns:
        The predictions as a tensor \in R^{134 * 288 * 1}
    """
    start_time = time.time()
    cur_time = time.localtime()
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", cur_time)
    
    predictions = []
    settings["turbine_id"] = 0
    exp = Experiment(settings)

    test_x = Experiment.get_test_x(settings)
    if settings["is_debug"]:
        end_test_x_get_time = time.time()
        print("predict line103: get test x in {} secs".format(end_test_x_get_time - start_time))
        start_time = end_test_x_get_time
    for i in range(settings["capacity"]):
        settings["turbine_id"] = i
        print('\n>>>>>>>>> Testing Turbine {:3d} >>>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(i))
        prediction = forecast_one(exp, test_x)
        torch.cuda.empty_cache()
        predictions.append(prediction)
        if settings["is_debug"] and (i + 1) % 10 == 0:
            end_time = time.time()
            print("\nElapsed time for predicting 10 turbines is {} secs".format(end_time - start_time))
            start_time = end_time

    return np.array(predictions)
