from keras.src import Model
from keras.src.callbacks import ReduceLROnPlateau
from keras.src.layers import Masking, GRU, Dense
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from utilities import save_pkl, load_pkl

class OpticoachModel:
    '''
    ### class OpticoachModel
    This `class` predicts label values using a recurrent neural network.

    ### dict modelFiles
    This `dict` stores `string` keys of all model related files and `string` values of file names.

    ### dict predictedFiles
    This `dict` stores `string` keys of prediction files and `string` values of file names.

    ### dict __preprocessedFiles
    This `dict` variable stores `string` keys of all preprocessed files and `string` values of file names.

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

        # We first accept an input batch of coaches. For each coach, a time-ordered sequence of
        # coaching metrics will be provided. In order to accomodate gaps in the data, a
        # masking is used to cover missing time steps and missing metrics.
        maxTimeStepCount = 75
        maxMetricCount = 30
        input = Masking(
            mask_value=0.0,
            input=(maxTimeStepCount, maxMetricCount)
        )
        
        # In order to handle long-term dependencies and avoid overfitting on out small data set
        # we will utilize a Gated Recurrent Unit (GRU). Dropout and regularization are also
        # supplemented in order to avoid overfitting.
        gru = GRU(
            64,
            dropout=0.2,
            recurrent_dropout=0.2,
            kernel_regularizer='l2'
        )(input)
        
        # In order to interpret the GRU output, a Dense layer is added. Dropout is ommited
        # following this layer because that adds imprecision to regression tasks.
        hidden = Dense(
            64,
            activation='relu',
            kernel_regularizer='l2'
        )
        
        # Finally, a few key coaching success metrics are trained on and predicted. For the
        # purpose of precise regression, linear activation is used.
        outputMetricCount = 10
        output = Dense(
            outputMetricCount,
            activation='linear'
        )

        model = Model(inputs=input, outputs=output)
        save_pkl(model, 'model.pkl')
        self.modelFiles['model'] = 'model.pkl'

    def train(self):
        '''
        Train the recurrent neural network.
        '''

        featureScaler, labelScaler = MinMaxScaler(), MinMaxScaler()
        # TO-DO: fit_transform training data, transform validation data

        save_pkl(featureScaler, 'featureScaler.pkl')
        self.modelFiles['featureScaler'] = 'featureScaler.pkl'
        save_pkl(labelScaler, 'labelScaler.pkl')
        self.modelFiles['labelScaler'] = 'labelScaler.pkl'

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
        model = load_pkl(self.modelFiles['model'])
        model.compile()
        model.fit()
        save_pkl(model, 'model.pkl')
        return

    def predict(self):
        model = load_pkl(self.modelFiles['model'])
        # Make prediction, save pkl, and update predictedFiles
        self.predictedFiles = {}
        return