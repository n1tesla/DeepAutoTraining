import os
from json import dump
import numpy as np
import pandas as pd
import pickle


class PREPARATION:
    def __init__(self,
                 config,
                 window_size,
                 stride_size,
                 scaler_type,
                 ):

        self.config=config
        self.window_size=window_size
        self.stride_size=stride_size
        self.scaler_type=scaler_type


    def create_dataset(self,data_path):

        df=self.read_data()
        train,test=self.split_data(df)
        df_train,df_test,scaler=self.scale_data(train,test,data_path)
        dataset_dict= {'df_train': df_train, 'df_test': df_test}

        for k, v in dataset_dict.items():
            #putting X and y into dictionary.
            # {'df_train': [X_train_array, y_train_array], 'df_test': [X_test_array,y_test_array]}
            dataset_dict[k] = self.SlidingWindow(v)
        return dataset_dict,df_train,df_test,scaler

    def read_data(self):
        df = pd.read_csv('data/spx.csv', index_col='date')
        print(f"df_head: {df.head()}")
        return df


    def split_data(self,df):
        train_size = int(len(df) * (1-self.config.test_ratio))
        df_train, df_test = df.iloc[0:train_size], df.iloc[train_size:]
        print(f"Train size: {train_size}, Test size: {len(df)-train_size}")
        return df_train, df_test


    def scale_data(self,df_train=None, df_test=None,data_path=None):

        scaler_model = None

        if self.scaler_type == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler_model = RobustScaler()
        elif self.scaler_type == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler_model = MinMaxScaler()
        elif self.scaler_type == 'standard':
            from sklearn.preprocessing import StandardScaler
            scaler_model = StandardScaler()
        assert scaler_model is not None, f"Choose a valid scaler by using 'robust', 'minmax' or 'standard' keywords"

        df_train.loc[:, ['close']] = scaler_model.fit_transform(df_train.loc[:,['close']].values)
        df_test.loc[:, ['close']] = scaler_model.transform(df_test.loc[:,['close']].values)
        # dump(scaler_model, open(os.path.join(data_path, "scaler.bin"), "wb"))

        with open(os.path.join(data_path, "scaler.bin"), "wb") as file:
            pickle.dump(scaler_model, file)
        return df_train,df_test,scaler_model

    def SlidingWindow(self, df: pd.core.frame.DataFrame) -> list:
        """
        @param df: Dataframe samples will be converted to Windowed array.
        @return: list of X_train and y_train
        """
        WindowedX,WindowedY=[],[]
        arr=df.to_numpy()
        WindowedX.append(extract_window(arr, self.window_size, self.stride_size))
        WindowedY.append(extract_labels(arr, self.window_size, self.stride_size))

        WindowedX = np.concatenate(WindowedX, axis=0)
        WindowedY = np.concatenate(WindowedY, axis=0)
        print(f"WindowedX_Shape: {WindowedX.shape}, WindowedY_Shape: {WindowedY.shape}")
        # WindowedY=WindowedY.reshape(WindowedY.shape[0])
        #shuffling windows in list
        # idx = np.random.permutation(len(WindowedX))
        # WindowedX, WindowedY = WindowedX[idx], WindowedY[idx]
        return [WindowedX, WindowedY]

def extract_window(arr: np.ndarray, size: int, stride: int) -> np.ndarray:
    """
    @param arr: Unique Unit number
    @param size: Window size, also used in LSTM input shape (sample_size, feature_size,window_size)
    @param stride: Also called step size
    @return:
    """
    examples=[]

    min_len = size - 1
    max_len = len(arr) - size
    for i in range(0, max_len + 1, stride):
        example = arr[i:size + i]
        examples.append(np.expand_dims(example, 0))

    return np.vstack(examples)

def extract_labels(arr: np.ndarray, size: int, stride: int) -> np.ndarray:
    """
    @param arr:
    @param size:
    @param stride:
    @return:
    """
    examples = []
    max_len = len(arr) - size
    for i in range(-1, max_len, stride):
        examples.append(arr[size + i])

    return np.array(examples)