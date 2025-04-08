from keras.src import Model
from keras.src.layers import Input, LSTM, GRU, SimpleRNN, TimeDistributed, Bidirectional
import numpy as np

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
        self.__preprocessedFiles = preprocessor.preprocessedFiles
        self.build()

    def __build(self):
        input = Input(np.shape())

        self.modelFiles = {}

    def train(self):
        return
    
    def predict(self):
        self.predictedFiles = {}
        return