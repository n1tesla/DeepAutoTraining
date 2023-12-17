import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error



def model_test( model, X_train, X_test ):

    y_pred=None

    df_result = pd.DataFrame([])

    X_train_pred=model.predict(X_train)
    train_mae_loss=np.mean(np.abs(X_train_pred-X_train),axis=1)
    #more than train_mae_loss will be anomaly

    X_test_pred = model.predict(X_test)
    test_mae_loss=np.mean(np.abs(X_test_pred-X_test),axis=1)


    # (mae, r2,mse) = regression_evaluation(y, y_pred)
    # metrics = pd.DataFrame({'MAE': [mae], 'R2': [r2],'MSE':[mse]})
    # df_result.index=[config.network_name]
    # # df_result.loc[:,'model_name']=[config.network_name]
    # df_result.loc[:, 'model_no'] = [model_index]
    # df_result.loc[:, 'MAE'] = [mae]
    # df_result.loc[:, 'R2'] = [r2]
    # df_result.loc[:, 'MSE'] = [mse]
    # df_result=df_result.sort_values(by=['MAE'],ascending=True)

    return train_mae_loss, test_mae_loss






