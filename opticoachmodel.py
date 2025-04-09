from keras.src import Model
from keras.src.callbacks import ReduceLROnPlateau
from keras.src.layers import Masking, LSTM, Dense
from keras.src.optimizers import Adam
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from utilities import save_pkl, load_pkl

class OpticoachModel:
    '''
    ### class OpticoachModel
    This `class` predicts label values using a recurrent neural network.

    ### dict __preprocessedFiles
    This `dict` variable stores `string` keys of all preprocessed files and `string` values of file names.

    ### dict modelFiles
    This `dict` stores `string` keys of all model related files and `string` values of file names.

    ### dict predictedFiles
    This `dict` stores `string` keys of prediction files and `string` values of file names.

    ### void __init__(self, Preprocessor preprocessor)
    This `void` function is called as the constructor for an OpticoachModel object, initializing the variable
    __preprocessedFiles from a Preprocessor instance.

    ### void __build(self)
    This private `void` function is called by the constructor to build and save the untrained model
    according to the prescribed architecture.

    ### void train()
    This `void` function trains the model(s).

    ### void predict()
    This `void` function makes predictions, updating the files referenced by predictedFiles.
    '''

    def __init__(self, preprocessor):
        self.modelFiles = {}
        self.predictedFiles = {}
        self.__preprocessedFiles = preprocessor.preprocessedFiles
        self.build()

    def __build(self):
        '''
        Build the recurrent neural network.
        https://www.tensorflow.org/guide/keras/working_with_rnns
        https://keras.io/api/layers/recurrent_layers/gru/
        https://chatgpt.com/share/67f4a398-42f8-8012-9c56-9538846a97b0
        '''

        timeStepCount = 75
        metricCount = 30

        # We first accept an input batch of coaches. For each coach, a time-ordered sequence of
        # coaching metrics will be provided. In order to accomodate gaps in the data, a
        # masking is used to cover missing time steps and missing metrics. There will be gaps in the
        # time sequence in which a coach was not a head coach, and gaps in the metrics if data is
        # not available.
        maskedInput = Masking(
            mask_value=0.0,
            input=(timeStepCount, metricCount)
        )
        
        # In order to handle long-term dependencies we will utilize a Long Short Term-Memory (LSTM)
        # layer. Dropout and regularization are also supplemented in order to avoid overfitting.
        # We use chat's rule of thumb, lstm_units = min(128, max(32, features * 2)).
        lstm = LSTM(
            160,
            dropout=0.2,
            recurrent_dropout=0.2,
            kernel_regularizer='l2'
        )(maskedInput)
        
        # In order to interpret the LSTM output, a Dense layer is added. Dropout is ommited
        # following this layer because that adds imprecision to regression tasks.
        hidden = Dense(
            64,
            activation='relu',
            kernel_regularizer='l2'
        )(lstm)
        
        # Finally, a few key coaching success metrics are trained on and predicted. For the
        # purpose of precise regression, linear activation is used.
        denseOutput = Dense(
            metricCount,
            activation='linear'
        )(hidden)

        model = Model(inputs=maskedInput, outputs=denseOutput)
        save_pkl(model, 'model.pkl')
        self.modelFiles['model'] = 'model.pkl'

    def train(self):
        '''
        Train the recurrent neural network.
        '''

        xScaler, yScaler = MinMaxScaler(), MinMaxScaler()
        tX = load_pkl(self.__preprocessedFiles['trainX'])
        tY = load_pkl(self.__preprocessedFiles['trainY'])
        vX = load_pkl(self.__preprocessedFiles['validX'])
        vY = load_pkl(self.__preprocessedFiles['validY'])
        tXs = xScaler.fit_transform(tX)
        vXs = xScaler.transform(vX)
        tYs = yScaler.fit_transform(tY)
        vYs = yScaler.transform(vY)

        save_pkl(xScaler, 'xScaler.pkl')
        self.modelFiles['xScaler'] = 'xScaler.pkl'
        save_pkl(yScaler, 'yScaler.pkl')
        self.modelFiles['yScaler'] = 'yScaler.pkl'

        learningRateReducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
        model = Model(load_pkl(self.modelFiles['model']))
        model.compile(
            optimizer=Adam(learning_rate=1e-2),
            loss='mse',
            metrics=['mse', 'mae']
        )
        model.fit(
            tXs, tYs,
            batch_size=16,
            epochs=100,
            verbose=2,
            callbacks=learningRateReducer,
            validation_data=(vXs, vYs)
        )
        save_pkl(model, 'model.pkl')
        return

    def predict(self):
        model = load_pkl(self.modelFiles['model'])
        # Make prediction, save pkl, and update predictedFiles
        self.predictedFiles = {}
        return