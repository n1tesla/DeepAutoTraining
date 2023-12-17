import pandas as pd
from data.preprocessing import PREPARATION
from util import utils
from configs.config import call_config,Parser
from train.train import DeepRegressor
from train.train_NAS import train_nas_lstm

def start_search(config_all,observation):
    global time_path
    start_time=utils.create_start_time()
    df_overall_result=pd.DataFrame([])
    for network,input_cfg in config_all.items():
        config=Parser(input_cfg)
        print(f"Network_name: {network}")
        observation_name = f"{config.network_name}_{observation}"
        for window_size in config.window_size:
            for stride_size in config.stride_size:
                for scaler_type in config.scaler:
                    for val_ratio in config.validation_ratio:

                            time_path, data_path, models_dir = utils.make_paths(observation_name, start_time)
                            parameters_dict = {"window_size": window_size, "stride_size": stride_size,
                                               "scaler_type": scaler_type,"features": config.features, "val_ratio": val_ratio}

                            preparation = PREPARATION(config, window_size, stride_size, scaler_type)
                            dataset_dict, df_train, df_test, scaler = preparation.create_dataset(data_path)

                            if network=="LSTM_NAS":
                                train_nas_lstm(config,dataset_dict)

                            else:

                                regressor=DeepRegressor(dataset_dict,config,models_dir,parameters_dict,scaler,df_test)
                                tuner, best_hps=regressor.fit()

                                regressor.save(tuner, best_hps)

