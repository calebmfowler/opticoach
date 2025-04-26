from copy import deepcopy
from keras.src import Model
from keras.src.callbacks import ReduceLROnPlateau
from keras.src.layers import Input, Embedding, Concatenate, Masking, LSTM, Dense
from keras._tf_keras.keras.models import load_model
from keras.src.optimizers import Adam
from numpy import nan, newaxis, shape, unique
from preprocess import Preprocessor
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

    def __init__(self, arg):
        if type(arg) == Preprocessor:
            self.modelFiles = {}
            self.predictedFiles = {}
            self.__preprocessedFiles = Preprocessor(arg).preprocessedFiles
            self.__backgroundYears = Preprocessor(arg).backgroundYears
            self.__predictionYears = Preprocessor(arg).predictionYears
            self.__build()
        elif type(arg) == OpticoachModel:
            self.modelFiles = deepcopy(arg.modelFiles)
            self.predictedFiles = deepcopy(arg.predictedFiles)
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
        tX = load_pkl(self.__preprocessedFiles['trainX'])
        vX = load_pkl(self.__preprocessedFiles['validX'])
        tY = load_pkl(self.__preprocessedFiles['trainY'])
        vY = load_pkl(self.__preprocessedFiles['validY'])
        XTypes = load_pkl(self.__preprocessedFiles['XTypes'])
        YTypes = load_pkl(self.__preprocessedFiles['YTypes'])

        # We first accept an input batch of coaches. For each coach, a time-ordered sequence of
        # coaching metrics will be provided. However, some of these metrics are categorical with
        # string values, and so embedding layers are necessary
        inputLayers, numericalInputs, embeddingLayers = [], [], []
        for i, type in enumerate(XTypes):
            if type == str:
                inputLayer = Input((self.__backgroundYears,), name=f"input_{i}_cat")
                inputLayers.append(inputLayer)
                categoryCount = len(unique(tX[:, :, i]))
                embeddingLayer = Embedding(
                    categoryCount + 1,
                    min(50, (categoryCount + 1) // 2)
                )(inputLayer)
                embeddingLayers.append(embeddingLayer)
            else:
                inputLayer = Input((self.__backgroundYears, 1), name=f"input_{i}_num")
                inputLayers.append(inputLayer)
                numericalInputs.append(inputLayer)
        
        numericalConcatenation = Concatenate()(numericalInputs) if numericalInputs else None
        embeddingConcatenation = Concatenate()(embeddingLayers) if embeddingLayers else None
        
        if numericalConcatenation is not None and embeddingConcatenation is not None:
            mergeConcatenation = Concatenate()([numericalConcatenation, embeddingConcatenation])
        else:
            mergeConcatenation = numericalConcatenation or embeddingConcatenation
        
        # In order to accomodate gaps in the data, a masking is used to cover missing time steps 
        # and missing metrics. There will be gaps in the time sequence in which a coach was not 
        # a head coach, and gaps in the metrics if data is not available.
        maskLayer = Masking(mask_value=nan)(mergeConcatenation)
        
        # In order to handle long-term dependencies we will utilize a Long Short Term-Memory (LSTM)
        # layer. Dropout and regularization are also supplemented in order to avoid overfitting.
        # We use chat's rule of thumb, lstm_units = min(128, max(32, features * 2)).
        lstmLayer = LSTM(
            160,
            dropout=0.2,
            recurrent_dropout=0.2,
            kernel_regularizer='l2'
        )(maskLayer)
        
        # In order to interpret the LSTM output, a Dense layer is added. Dropout is ommited
        # following this layer because that adds imprecision to regression tasks.
        hiddenLayer = Dense(
            64,
            activation='relu',
            kernel_regularizer='l2'
        )(lstmLayer)
        
        # Finally, a few key coaching success metrics are trained on and predicted. For the
        # purpose of precise regression, linear activation is used.
        outputLayer = Dense(
            len(tY[0][0]),
            activation='linear'
        )(hiddenLayer)

        model = Model(inputs=inputLayers, outputs=outputLayer)
        model.save('files/model.keras')
        self.modelFiles['model'] = 'files/model.keras'

    def train(self):
        '''
        Train the recurrent neural network.
        '''

        tX = load_pkl(self.__preprocessedFiles['trainX'])
        vX = load_pkl(self.__preprocessedFiles['validX'])
        tY = load_pkl(self.__preprocessedFiles['trainY'])
        vY = load_pkl(self.__preprocessedFiles['validY'])
        XTypes = load_pkl(self.__preprocessedFiles['XTypes'])
        YTypes = load_pkl(self.__preprocessedFiles['YTypes'])

        print(f"tX\n{tX}")
        print(f"tY\n{tY}")

        '''
        xScaler, yScaler = MinMaxScaler(), MinMaxScaler()

        tXs = xScaler.fit_transform(tX)
        vXs = xScaler.transform(vX)
        tYs = yScaler.fit_transform(tY)
        vYs = yScaler.transform(vY)

        save_pkl(xScaler, 'files/xScaler.pkl')
        save_pkl(yScaler, 'files/yScaler.pkl')
        self.modelFiles['xScaler'] = 'files/xScaler.pkl'
        self.modelFiles['yScaler'] = 'files/yScaler.pkl'
        '''

        def split_features(X):
            xList = []
            for i in range(shape(X)[2]):
                metric = X[:, :, i]
                xList.append(metric[..., newaxis])
            return xList
        
        tXS = split_features(tX)
        vXS = split_features(vX)

        learningRateReducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
        
        model = load_model(self.modelFiles['model'])
        
        model.compile(
            optimizer=Adam(learning_rate=1e-2),
            loss='mse',
            metrics=['mse', 'mae']
        )
        
        print(f"tXS\n{tXS}")
        print(f"tY\n{tY}")
        model.fit(
            tXS, tY,
            batch_size=16,
            epochs=100,
            verbose=2,
            callbacks=learningRateReducer,
            validation_data=(vXS, vY)
        )

        save_pkl(model, 'files/model.pkl')
        self.modelFiles['model'] = 'files/model.pkl'

        return

    def predict(self):
        model = Model(load_pkl(self.modelFiles['model']))

        xScaler = MinMaxScaler(load_pkl(self.modelFiles['xScaler']))
        yScaler = MinMaxScaler(load_pkl(self.modelFiles['yScaler']))

        tX = load_pkl(self.__preprocessedFiles['trainX'])
        vX = load_pkl(self.__preprocessedFiles['validX'])

        tXs = xScaler.transform(tX)
        vXs = xScaler.transform(vX)

        tPs = model.predict(tXs)
        vPs = model.predict(vXs)

        tP = yScaler.inverse_transform(tPs)
        vP = yScaler.inverse_transform(vPs)

        save_pkl(tP, 'files/trainP.pkl')
        save_pkl(vP, 'files/validP.pkl')
        self.modelFiles['trainP'] = 'files/trainP.pkl'
        self.modelFiles['validP'] = 'files/validP.pkl'

        return