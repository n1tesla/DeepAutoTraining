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


    return train_mae_loss, test_mae_loss






