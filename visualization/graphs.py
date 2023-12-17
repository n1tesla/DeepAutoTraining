import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

def plot_distplot(loss,model_path):
    sns.distplot(loss,bins=50,kde=True)
    plot_path= model_path / f"train_mae_distplot.png"
    plt.savefig(plot_path)
    plt.close()

def plot_anomalies(df_test,window_size,test_mae_loss,THRESHOLD,scaler,model_path):

    test_score_df = pd.DataFrame(index=df_test[window_size:].index)
    test_score_df['test_MAE'] = test_mae_loss[:-1]
    test_score_df['threshold'] = THRESHOLD
    test_score_df['anomaly'] = test_score_df.test_MAE > test_score_df.threshold
    test_score_df['close'] = df_test[window_size:].close
    anomalies = test_score_df[test_score_df.anomaly == True]

    # plt.plot(test_score_df.index, test_score_df.test_MAE, label='test_MAE')
    # plt.plot(test_score_df.index, test_score_df.threshold, label='threshold')
    # plt.legend();

    # sns.lineplot(x=test_score_df.index,y=test_score_df['test_MAE'])
    # sns.lineplot(x=test_score_df.index,y=test_score_df['threshold'])
    sns.set(rc={'figure.figsize': (18,12)})

    y_1=scaler.inverse_transform(test_score_df['close'].to_numpy().reshape(len(test_score_df),1))
    y_2=scaler.inverse_transform(anomalies['close'].to_numpy().reshape(len(anomalies),1))
    y1_list=[value[0] for value in y_1]
    y2_list=[value[0] for value in y_2]
    sns.lineplot(x=test_score_df.index,y=y1_list,label='close_price')
    sns.scatterplot(x=anomalies.index,y=y2_list,color=sns.color_palette()[3],label='anomaly')
    plt.legend()
    plot_path_2 = model_path / f"anomalies.png"
    plt.savefig(plot_path_2)
    plt.close()

def plot_result(y_true: np.ndarray, y_pred: np.ndarray, model_path: str, index: int,
                result:pd.DataFrame):

    result_string = ""

    for column_name, value in result.iloc[0].items():
        result_string += f"{column_name} : {value}\n"

    plt.rcParams['figure.figsize'] = 12, 10
    plt.plot(y_pred)
    plt.plot(y_true)
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
    plt.figtext(x=0.1,y=0.94, s=result_string, bbox=bbox,ha='left',va = 'top')
    plt.ylabel('RUL')
    plt.xlabel('samples')
    plt.legend(('Predicted', 'True'), loc='upper right')
    plt.title('COMPARISION OF Real and Predicted values')
    plt.savefig(model_path + f"/TrueVSpred_{index}.png", format="png")
    plt.close()


def train_evaluation_graphs(history, model_path: str, index: int):

    # summarize history for Loss
    fig_acc = plt.figure(figsize=(10, 10))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plot_dir= model_path / f"loss_{index}.png"
    plt.savefig(plot_dir)
    plt.close()


