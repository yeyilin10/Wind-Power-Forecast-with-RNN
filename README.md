# Wind Power Forecasting with Recurrent Neural Networks

The project was presented as part of the 2022 KDD Cup Spatial Dynamic Wind Power Forecast Challenge [Baidu KDD Cup 2022](https://aistudio.baidu.com/aistudio/competition/detail/152/0/introduction), which aimed to advance data-driven machine learning methods for wind power forecasting.

Baidu also offers a preliminary solution by a simple GRU model, baseline codes should refers to the [official baseline](https://github.com/PaddlePaddle/PGL/tree/main/examples/kddcup2022/wpf_baseline).

## Project Overview

The project involves predicting wind power production using Recurrent Neural Networks (RNNs) and Gated Recurrent Units (GRU), two powerful models for sequential data analysis. This is crucial for integrating renewable energy sources into the electrical grid due to their high variability. The models are used to forecast power production for 134 wind turbines over 2 days intervals, using 245 days of historical context data.

## Dataset

The dataset includes over 8 months of historical data for 134 wind turbines, including power production, wind speed, direction, temperature, and other turbine internal status variables. 

### Preparing the Data
Place the [data](https://aistudio.baidu.com/aistudio/competition/detail/152/0/datasets) in the `data/` directory.


### Configuring the Parameters
All the configurable parameters, such as the number of layers and features used, are defined in the prepare.py file as a dictionary. Modify this file as needed to align with your specific requirements or experimental setup.

## Training the Models
To train the RNN or GRU model, run the following command:

```shell
python train.py
```

By default, the train.py script specifies the model as a plain RNN. To use the GRU model, modify the prepare.py file as needed.

## Methodology

The methodology involved:

1. Data Preprocessing: The data preprocessing involved handling missing data using linear interpolation and kNN interpolation.

2. Model Training: Two recurrent neural network models, plain RNN and GRU, were used for the task. These models were chosen because they are effective at capturing the temporal dependencies in sequential data, including wind power data. 

3. Model Ensembling: After training the plain RNN and GRU models separately, the models were ensembled to obtain the final prediction scores. A 50% weight was assigned to each of the two models in the ensemble.

## Results

The model achieved an RMSE of 53.58 and an MAE of 44.03, demonstrating the effectiveness of the approach in wind power forecasting.

## Note

This project was conducted as part of the 2022 KDD Cup Spatial Dynamic Wind Power Forecast Challenge. The goal of the challenge was to facilitate progress in data-driven machine learning methods for wind power forecasting. The project aimed to learn more about the latest trends and techniques in the field of wind power forecasting and their potential applications in the renewable energy industry.

## Contact

For any queries, feel free to reach out to Zoe Ye at ylye@connect.ust.hk
