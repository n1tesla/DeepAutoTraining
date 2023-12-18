import os
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import keras_tuner
from keras_tuner.tuners import BayesianOptimization
import tensorflow as tf
from metrics import evaluate
from pathlib import Path
from visualization import graphs
import mlflow

class DeepRegressor:
    def __init__(self,
                 dataset_dict:dict,
                 config:dict,
                 models_dir:str,
                 parameters_dict:dict,
                 scaler,
                 df_test):
        """
        Initialize the DeepRegressor.

        Parameters:
        - dataset_dict: Dictionary containing training and testing datasets.
        - config: Configuration parameters.
        - models_dir: Directory to save models.
        - parameters_dict: Dictionary containing additional parameters.
        - scaler: Scaler object.
        - df_test: Testing DataFrame.
        """

        self.dataset_dict=dataset_dict
        self.config=config
        self.models_dir=models_dir
        self.parameters_dict=parameters_dict
        self.scaler=scaler
        self.df_test=df_test


        self.problem_type='anomaly'
        self.cwd = Path.cwd()
        self.X_train, self.y_train = dataset_dict['df_train'][0], dataset_dict['df_train'][1]
        self.X_test, self.y_test = dataset_dict['df_test'][0],dataset_dict['df_test'][1]
        model=self.get_model_class(self.problem_type,config.module_name,config.network_name)
        self.network=model(self.X_train.shape[1:],1)
        self.parameters_dict['batch_size'] = 512

    def define_callback(self, early_stop: bool = False, reduceLR: bool = False, tensorboard: bool = False)->List[tf.keras.callbacks.Callback]:
        """
         Define a list of callbacks based on specified options.

         Parameters:
         - early_stop: Whether to include EarlyStopping callback.
         - reduceLR: Whether to include ReduceLROnPlateau callback.
         - tensorboard: Whether to include TensorBoard callback.

         Returns:
         - List of callback instances.
         """
        callback_list=[]
        if early_stop:
            early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6)
            callback_list.append(early_stop)
        if reduceLR:
            #Exponential Decay also is another option
            ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3,min_lr=1e-8)
            callback_list.append(ReduceLROnPlateau)

        if tensorboard:
            #to run cd to models directory and run this command while in the models directory:
            #tensorboard --logdir tensorboard/
            from keras.callbacks import TensorBoard
            tensorboard_dir = self.models_dir / 'tensorboard'
            tensorboard_callback = TensorBoard(log_dir=tensorboard_dir, histogram_freq=1)  # profile_batch='500,520'
            callback_list.append(tensorboard_callback)

        return callback_list

    def search_hyperparam(self)->BayesianOptimization:
        """
        Perform a hyperparameter search using Bayesian Optimization.

        Returns:
        - BayesianOptimization tuner instance.
        """
        hypermodel=self.network
        callbacks=self.define_callback(early_stop=True,reduceLR=True)
        tuner=BayesianOptimization(hypermodel,objective=keras_tuner.Objective("val_loss",direction="min")
                                   ,seed=1,max_trials=self.config.max_trials, directory=os.path.normpath(self.cwd),project_name='RS',overwrite=True)

        tuner.search_space_summary()
        tuner.search(self.X_train,self.y_train,validation_split=self.parameters_dict["val_ratio"],callbacks=callbacks,batch_size=self.parameters_dict['batch_size'],
                     verbose=1,epochs=self.config.epochs,use_multiprocessing=True)
        return tuner


    def fit(self)-> Tuple[BayesianOptimization, List[Dict[str, Any]]]:
        """
        Fit the model and return the tuner instance and the best hyperparameters.

        Returns:
        - Tuple containing the tuner instance and a list of best hyperparameters.
        """
        tuner=self.search_hyperparam()
        best_hps=tuner.get_best_hyperparameters(self.config.max_trials)

        return tuner, best_hps

    def save(self, tuner: BayesianOptimization, best_hps: List[Dict[str, Any]]):
        """
        Save the model and relevant information based on the best hyperparameters.

        Parameters:
        - tuner: BayesianOptimization tuner instance.
        - best_hps: List of best hyperparameters.
        """
        callbacks = self.define_callback(early_stop=True, reduceLR=True, tensorboard=True)

        for index, trial in enumerate(best_hps):
            model=tuner.hypermodel.build(trial)
            history=model.fit(self.X_train,self.y_train,validation_split=self.parameters_dict["val_ratio"],epochs=self.config.epochs,callbacks=callbacks,
                              batch_size=self.parameters_dict['batch_size'],verbose=1,use_multiprocessing=True)

            hp_config = {}
            hp_config.update(trial.values)
            lr = trial["learning_rate"]
            record_name = f"{self.parameters_dict['window_size']}ws_{self.parameters_dict['stride_size']}ss_" \
                          f"{self.parameters_dict['scaler_type']}_{self.parameters_dict['val_ratio']}vr_{index}"

            model_path= self.models_dir / record_name
            model_path.mkdir(exist_ok=True)
            model.save(model_path)

            train_mae_loss, test_mae_loss = evaluate.model_test(model, self.X_train, self.X_test)

            # graphs.plot_distplot(train_mae_loss, model_path)
            graphs.plot_anomalies(self.df_test,
                                  self.parameters_dict['window_size'],
                                  test_mae_loss,
                                  self.config.THRESHOLD,
                                  self.scaler,
                                  model_path)
            # graphs.train_evaluation_graphs(history, model_path, index)
            mlflow.set_experiment(f"{self.config.network_name}_{record_name}")
            with mlflow.start_run() as run:

                # mlflow.log_params(df_result)
                mlflow.log_params(trial.values)

                mlflow.tensorflow.log_model(model,'model')
                #mlflow.log_metrics(metrics.to_dict()) #dict istiyor.
                mlflow.log_metric('mae',np.mean(train_mae_loss))


    def get_model_class(self,problem_type:str,module_name:str,model_name:str):
        import  importlib

        imported_module=importlib.import_module(f"models.{problem_type}.{module_name}")
        model=getattr(imported_module,model_name)
        return model
