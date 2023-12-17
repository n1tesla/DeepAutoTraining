from keras.models import Model
from keras.layers import Input, Dense, Conv2D, GlobalAveragePooling1D,LSTM, Dropout, RepeatVector,TimeDistributed,Conv1D

def model_fn(actions):

    input_layer=Input(shape=(30,1))
    units_1, units_2 = actions

    x = LSTM(units=units_1)(input_layer)
    x = Dropout(0.2)(x)
    x = RepeatVector(n=30)(x)
    x=LSTM(units=units_2)(x)
    x = Dropout(0.2)(x)
    output_layer = Dense(units=1)(x)

    model=Model(inputs=input_layer,outputs=output_layer)


    # unpack the actions from the list
    # kernel_1, filters_1, kernel_2, filters_2, kernel_3, filters_3, kernel_4, filters_4 = actions
    #
    # ip = Input(shape=(30, 1))
    # x = Conv1D(filters_1, kernel_1, strides=2, padding='same', activation='relu')(ip)
    # x = Conv1D(filters_2, kernel_2, strides= 1, padding='same', activation='relu')(x)
    # x = Conv1D(filters_3, kernel_3, strides= 2 ,padding='same', activation='relu')(x)
    # x = Conv1D(filters_4, kernel_4, strides= 1, padding='same', activation='relu')(x)
    # x = GlobalAveragePooling1D()(x)
    # x = Dense(1, activation='softmax')(x)
    # model=Model(inputs=ip, outputs=x)

    return model