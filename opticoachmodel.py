from copy import deepcopy
import keras.src.backend as K
from keras import Model, utils
from keras.src.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.src.layers import Input, Embedding, Concatenate, Masking, RepeatVector, LSTM, Dense, TimeDistributed, Lambda, BatchNormalization
from keras._tf_keras.keras.models import load_model
from keras.src.optimizers import Adam
from keras_tuner import BayesianOptimization, HyperParameters
import tensorflow as tf
from numpy import array as nparr, isnan, isinf, nan, newaxis, shape, unique
from preprocess import Preprocessor
from Expander import Expander
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
            self.__build(hp=HyperParameters())  # Build the model with hyperparameters
        elif type(arg) == OpticoachModel:
            self.predictedFiles = deepcopy(arg.predictedFiles)
            self.__modelFiles = deepcopy(arg.__modelFiles)
            self.__preprocessedFiles = deepcopy(arg.__preprocessedFiles)
            self.__backgroundYears = deepcopy(arg.__backgroundYears)
            self.__predictionYears = deepcopy(arg.__predictionYears)
        else:
            raise Exception("Incorrect arguments for OpticoachModel.__init__(self, preprocessor)")
        return

    def __build(self, hp: HyperParameters):
        '''
        Build the recurrent neural network.
        https://www.tensorflow.org/guide/keras/working_with_rnns
        https://keras.io/api/layers/recurrent_layers/gru/
        https://chatgpt.com/share/67f4a398-42f8-8012-9c56-9538846a97b0
        '''

        # Define hyperparameters to tune
        lstm_units = hp.Int('lstm_units', min_value=128, max_value=256, step=32)
        dense_units = hp.Int('dense_units', min_value=125, max_value=256, step=32)
        dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
        
        tY = load_pkl(self.__preprocessedFiles['trainY'])
        tX = load_pkl(self.__preprocessedFiles['trainX'])
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
        
        # Concatenate and normalize the numerical features
        numericalConcatenation = Concatenate()(numerizedLayers)
        
        # In order to accomodate gaps in the data, a masking is used to cover missing time steps 
        # and missing metrics. There will be gaps in the time sequence in which a coach was not 
        # a head coach, and gaps in the metrics if data is not available.
        maskedInputs = Masking(mask_value=0)(numericalConcatenation)
        
        # In order to handle long-term dependencies we will utilize a Long Short Term-Memory (LSTM)
        # layer. Dropout and regularization are also supplemented in order to avoid overfitting.
        # We use chat's rule of thumb, lstm_units = min(128, max(32, features * 2)).
        encoderLSTM = LSTM(
            lstm_units,
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate,
            kernel_regularizer='l2',
            return_sequences=True  # Ensure the LSTM outputs sequences for each time step
        )(maskedInputs)
        lstmLayer = BatchNormalization()(encoderLSTM)

        # In order to interpret the LSTM output, a Dense layer is added. Dropout is omitted
        # following this layer because that adds imprecision to regression tasks.
        hiddenLayer = Dense(
            dense_units,
            activation='relu',
            kernel_regularizer='l2'
        )(lstmLayer)
        
        # Finally, a few key coaching success metrics are trained on and predicted. For the
        # purpose of precise regression, linear activation is used.
        outputLayer = TimeDistributed(Dense(
            len(tY[0][0]),
            activation='linear'
        ))(hiddenLayer)

        model = Model(inputs=inputLayers, outputs=outputLayer)

        model.compile(
            optimizer=Adam(),
            loss='mse',
            metrics=['mse', 'mae']
        )

        return model

    def train(self):
        '''
        Train the recurrent neural network.
        '''

        tuner = BayesianOptimization(
            self.__build,
            objective='val_loss',  # Minimize validation loss
            max_trials=5,  # Number of hyperparameter combinations to try
            directory='tuner_results',
            project_name='opticoach_tuning'
        )

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

        # Perform the search
        learningRateReducer = ReduceLROnPlateau(
            monitor='val_loss',  # Monitor validation loss
            factor=0.5,          # Reduce learning rate by a factor of 0.5
            patience=5,          # Wait for 3 epochs without improvement
            min_lr=1e-6          # Minimum learning rate
        )

        earlyStopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        tuner.search(
            tXS, tY,
            validation_data=(vXS, vY),
            epochs=50,
            batch_size=16,
            verbose=2,
            callbacks=[learningRateReducer, earlyStopping]
        )

        # Get the best hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(f"Best hyperparameters: {best_hps.values}")

        # Build the best model using the best hyperparameters
        best_model = tuner.hypermodel.build(best_hps)

        # Define the learning rate reducer callback

        # Train the best model with the learning rate reducer
        best_model.fit(
            tXS, tY,
            batch_size=8,
            epochs=100,
            verbose=2,
            callbacks=[learningRateReducer, earlyStopping],  # Include the learning rate reducer earlyStopping],
            validation_data=(vXS, vY)
        )

        # Save the trained model
        best_model.save(self.__modelFiles['model'])

        # visualize the model structure:
        utils.plot_model(
            best_model,
            to_file="files/Model_Visual.png",
            rankdir="TB",
            show_layer_names=False,
            expand_nested=True,
            dpi=200,
            show_layer_activations=True,
        )

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