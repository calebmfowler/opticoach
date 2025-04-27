from copy import deepcopy
from keras import Model
from keras.src.callbacks import ReduceLROnPlateau
from keras.src.layers import Input, Embedding, Concatenate, Masking, Lambda, LSTM, Dense, TextVectorization, Normalization
from keras._tf_keras.keras.models import load_model
from keras.src.optimizers import Adam
from numpy import array as nparr, isnan, isinf, nan, newaxis, shape, unique
from preprocess import Preprocessor
from slicer import Slicer
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

    def __init__(self, arg):
        if type(arg) == Preprocessor:
            self.predictedFiles = {
                'trainP' : 'files/trainP.pkl',
                'validP' : 'files/validP.pkl'
            }
            self.__modelFiles = {
                'model' : 'files/model.keras'
            }
            self.__preprocessedFiles = Preprocessor(arg).preprocessedFiles
            self.__backgroundYears = Preprocessor(arg).backgroundYears
            self.__predictionYears = Preprocessor(arg).predictionYears
            self.__build()
        elif type(arg) == OpticoachModel:
            self.predictedFiles = deepcopy(arg.predictedFiles)
            self.__modelFiles = deepcopy(arg.__modelFiles)
            self.__preprocessedFiles = deepcopy(arg.__preprocessedFiles)
            self.__backgroundYears = deepcopy(arg.__backgroundYears)
            self.__predictionYears = deepcopy(arg.__predictionYears)
        else:
            raise Exception("Incorrect arguments for OpticoachModel.__init__(self, preprocessor)")
        return

    def __build(self):
        '''
        Build the recurrent neural network.
        https://www.tensorflow.org/guide/keras/working_with_rnns
        https://keras.io/api/layers/recurrent_layers/gru/
        https://chatgpt.com/share/67f4a398-42f8-8012-9c56-9538846a97b0
        '''
        
        tY = load_pkl(self.__preprocessedFiles['trainY'])
        XEmbeds = load_pkl(self.__preprocessedFiles['XEmbeds'])
        XVocabs = load_pkl(self.__preprocessedFiles['XVocabs'])

        # We first accept an input batch of coaches. For each coach, a time-ordered sequence of
        # coaching metrics will be provided. However, some of these metrics are categorical with
        # string values, and so we must use a TextVectorization layer on these.
        inputLayers, numerizedLayers = [], []
        for i, embed in enumerate(XEmbeds):
            if embed:
                inputLayer = Input((self.__backgroundYears,), name=f"input_{i}_cat")
                inputLayers.append(inputLayer)
                vocabSize = len(XVocabs[i])
                embeddingLayer = Embedding(
                    vocabSize + 1,
                    min(50, (vocabSize + 1) // 2)
                )(inputLayer)
                numerizedLayers.append(embeddingLayer)
            else:
                inputLayer = Input((self.__backgroundYears, 1), name=f"input_{i}_num")
                inputLayers.append(inputLayer)
                numerizedLayers.append(inputLayer)
        
        numericalConcatenation = Concatenate()(numerizedLayers)

        normalizedLayer = Normalization()(numericalConcatenation)

        # In order to accomodate gaps in the data, a masking is used to cover missing time steps 
        # and missing metrics. There will be gaps in the time sequence in which a coach was not 
        # a head coach, and gaps in the metrics if data is not available.
        maskLayer = Masking(mask_value=0)(normalizedLayer)
        
        # In order to handle long-term dependencies we will utilize a Long Short Term-Memory (LSTM)
        # layer. Dropout and regularization are also supplemented in order to avoid overfitting.
        # We use chat's rule of thumb, lstm_units = min(128, max(32, features * 2)).
        lstmLayer = LSTM(
            128,
            dropout=0.3,
            recurrent_dropout=0.25,
            kernel_regularizer='l2',
            return_sequences=True  # Ensure the LSTM outputs sequences for each time step
        )(normalizedLayer)

        # Slice the LSTM output to keep only the last 3 time steps
        slicedLayer = Slicer(num_steps=3)(lstmLayer)

        # In order to interpret the LSTM output, a Dense layer is added. Dropout is omitted
        # following this layer because that adds imprecision to regression tasks.
        hiddenLayer = Dense(
            64,
            activation='relu',
            kernel_regularizer='l2'
        )(slicedLayer)

        # revert the normalization
        unnormalizedLayer = Normalization(invert=True)(hiddenLayer)
        
        # Finally, a few key coaching success metrics are trained on and predicted. For the
        # purpose of precise regression, linear activation is used.
        outputLayer = Dense(
            len(tY[0][0]),
            activation='linear'
        )(unnormalizedLayer)

        model = Model(inputs=inputLayers, outputs=outputLayer)
        model.save(self.__modelFiles['model'])

    def train(self):
        '''
        Train the recurrent neural network.
        '''

        tX = load_pkl(self.__preprocessedFiles['trainX'])
        vX = load_pkl(self.__preprocessedFiles['validX'])
        tY = load_pkl(self.__preprocessedFiles['trainY'])
        vY = load_pkl(self.__preprocessedFiles['validY'])
        XEmbeds = load_pkl(self.__preprocessedFiles['XEmbeds'])

        def split_features(X):
            xList = []
            for i, embed in enumerate(XEmbeds):
                metric = X[:, :, i, newaxis]
                xList.append(metric)
            return xList
        
        tXS = split_features(tX)
        vXS = split_features(vX)

        learningRateReducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
        
        model = load_model(self.__modelFiles['model'])
        
        model.compile(
            optimizer=Adam(learning_rate=1e-2),
            loss='mse',
            metrics=['mse', 'mae']
        )

        model.fit(
            tXS, tY,
            batch_size=8,
            epochs=50,
            verbose=2,
            callbacks=learningRateReducer,
            validation_data=(vXS, vY)
        )

        model.save(self.__modelFiles['model'])

        return

    def predict(self):
        model = load_model(self.__modelFiles['model'])

        tX = load_pkl(self.__preprocessedFiles['trainX'])
        vX = load_pkl(self.__preprocessedFiles['validX'])
        tY = load_pkl(self.__preprocessedFiles['trainY'])
        vY = load_pkl(self.__preprocessedFiles['validY'])
        XEmbeds = load_pkl(self.__preprocessedFiles['XEmbeds'])

        def split_features(X):
            xList = []
            for i, embed in enumerate(XEmbeds):
                metric = X[:, :, i, newaxis]
                xList.append(metric)
            return xList

        tXS = split_features(tX)
        vXS = split_features(vX)

        tP = model.predict(tXS)
        vP = model.predict(vXS)

        save_pkl(tP, self.predictedFiles['trainP'])
        save_pkl(vP, self.predictedFiles['validP'])

        return