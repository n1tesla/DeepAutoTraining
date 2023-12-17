import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import tensorflow as tf
import shap
import timeshap
from timeshap.utils import calc_avg_event, calc_avg_sequence
from timeshap.explainer import local_report, local_pruning, feat_explain_all,local_event
from timeshap.plot import plot_temp_coalition_pruning,plot_event_heatmap
import keras.backend as K

def shap_evaluate_features(model, preparation, dataset_dict: dict):
    # Its need to be updated
    shap.initjs()
    tf.compat.v1.disable_v2_behavior()
    X = dataset_dict['df_train'][0]
    X_test = dataset_dict['df_test'][0]

    # trial
    df_train, df_test = preparation.create_shap_dataset(
        'C:/Users/xyualtinoz/PycharmProjects/nasa-pdm/saved_models/LSTM_code_dev/20230927_1537')
    filtered_df_train = df_train[preparation.features]

    background = X[np.random.choice(X.shape[0], 100, replace=False)]

    explainer = shap.DeepExplainer(model=model, data=filtered_df_train)
    random_test_X = X_test[np.random.choice(X_test.shape[0], 100, replace=False)]
    # shap_values = explainer.shap_values(X_test[40:100,:,:])
    shap_values = explainer.shap_values(random_test_X)
    shap.summary_plot(shap_values, background)

    # shap.bar_plot(explainer.expected_value,shap_values,X_test[40:100,:,:])


def timeshap_evaluate_features(model, dataset_dict, preparation):
    alt.renderers.enable('altair_viewer')

    #model = tf.keras.models.load_model(r"C:\Users\xyualtinoz\PycharmProjects\nasa-pdm\hp_models_for_test\30ws_2ss_minmax_0.25vr_13",custom_objects={'r2_keras':r2_keras})

    df_train, df_test = preparation.create_shap_dataset('C:/Users/xyualtinoz/PycharmProjects/nasa-pdm/saved_models/LSTM_code_dev/20230927_1537')

    filtered_df_train = df_train[preparation.features]
    filtered_df_train = np.expand_dims(filtered_df_train.to_numpy().copy(), axis=0)

    filtered_df_test =df_test[df_test['unit_number'] == 1]
    filtered_df_test = filtered_df_test.drop(filtered_df_test.index[0])
    filtered_df_test = filtered_df_test[preparation.features]
    filtered_df_test = np.expand_dims(filtered_df_test.to_numpy().copy(), axis=0)
    pos_x_pd = filtered_df_test

    f = lambda x: model.predict(x)

    average_event = calc_avg_event(df_train, numerical_feats=preparation.features, categorical_feats=[])

    # average sequence cannot be calculated due to sequence sizes inequality

    plot_feats = {'LPCompOT': 'LPCompOT', 'PhyCoreSpeed': 'PhyCoreSpeed',
                  'HPCOStaticPre': 'HPCOStaticPre', 'CorrectedFanSpeed': 'CorrectedFanSpeed',
                  'BleedEnthalpy': 'BleedEnthalpy',
                  'HPTurbineCoolAirFlow': 'HPTurbineCoolAirFlow', 'LPTurbineAirFlow': 'LPTurbineAirFlow',
                  'CorrectedCoreSpeed': 'CorrectedCoreSpeed'}

    # plot_feats = {'LPCompOT': 'LPCompOT', 'PhyCoreSpeed': 'PhyCoreSpeed', 'HPCOStaticPre': 'HPCOStaticPre',
    # 'CorrectedFanSpeed':'CorrectedFanSpeed', 'BleedEnthalpy': 'BleedEnthalpy', 'HPTurbineCoolAirFlow':
    # 'HPTurbineCoolAirFlow','LPTurbineAirFlow':'LPTurbineAirFlow','CorrectedCoreSpeed':'CorrectedCoreSpeed',
    # 'HPCompOT':'HPCompOT','LPTurbineOT':'LPTurbineOT', 'HPCompOP':'HPCompOP', 'PhyFanSpeed':'PhyFanSpeed',
    # 'RFuelFlow':'RFuelFlow','BypassRatio':'BypassRatio'}

   #pruning_dict = {'tol': 0.05}
   #event_dict = {'rs': 42, 'nsamples': 20}
   #feature_dict = {'rs': 42, 'nsamples': 30000, 'feature_names': preparation.features, 'plot_features': plot_feats}
   #cell_dict = {'rs': 42, 'nsamples': 20, 'top_x_feats': 2, 'top_x_events': 2}

   #local_report_plot = local_report(f, pos_x_pd, pruning_dict, event_dict, feature_dict, cell_dict=cell_dict,
   #                                 entity_uuid="unit number 5", entity_col='unit_number', baseline=average_event)
   #local_report_plot.show()

    ## Local pruning
    #coal_plot_data, coal_prun_idx = local_pruning(f, filtered_df_train, pruning_dict, average_event, entity_uuid = "unit number 5",
    #entity_col = "unit_number",verbose= False)
    #pruning_idx = filtered_df_train.shape[1] + coal_prun_idx
    # pruning_plot = plot_temp_coalition_pruning(coal_plot_data, coal_prun_idx, plot_limit=40)
    # pruning_plot.save("chart.png",engine="altair_saver")
    # pruning_plot.show()

    #feat_data = feat_explain_all(f, )

    #Event-level plots
    #event_data = local_event(f, filtered_df_train, event_dict,  "unit number 5", "unit_number", average_event,
    #                         pruning_idx)
    #event_plot = plot_event_heatmap(event_data)
    #event_plot.show()
    a = 5


def r2_keras(y_true, y_pred):
    """Coefficient of Determination
    """
    SS_res =  K.sum(K.square( y_true - y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )
