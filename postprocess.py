from copy import deepcopy
from opticoachmodel import OpticoachModel
from numpy import sum, mean, shape, vstack
from preprocess import Preprocessor
from utilities import load_pkl
import matplotlib.pyplot as plt
import numpy as np

class Postprocessor:
    '''
    ## class Postprocessor
    This `class` postprocesses the predicted data into a format suitable for reporting results.

    ### dict __predictedFiles
    This private `dict` variable stores `string` keys of all prediction files and `string` values of 
    file names as sourced from the OpticoachModel instance

    ### dict postprocessedFiles
    This `dict` stores `string` keys of all postprocessed files and `string` values of file names

    ### void __init__(self, OpticoachModel model)
    This `void` function is called as the constructor for a Postprocessor object, initializing the variable 
    __preprocessedFiles from a Preprocessor instance and initializing the variable __predictedFiles from an
    OpticoachModel instance.

    ### void postprocess()
    This `void` function postprocesses all predicted data, updating the files referenced by postprocessedFiles
    '''

    def __init__(self, arg1, arg2=None):
        if type(arg1) == Preprocessor and type(arg2) == OpticoachModel:
            self.postprocessedFiles = {}
            self.__preprocessedFiles = Preprocessor(arg1).preprocessedFiles
            self.__predictedFiles = OpticoachModel(arg2).predictedFiles
        elif type(arg1) == Postprocessor and arg2 == None:
            self.__predictedFiles = deepcopy(arg1.__predictedFiles)
            self.postprocessedFiles = deepcopy(arg1.postprocessedFiles)
        else:
            raise Exception("Incorrect arguments for Postprocessor.__init__(self, preprocessor, model=None)")
        return

    def postprocess(self):
        """
        Postprocess the predicted data and calculate R² values for multiple outputs.
        """
        self.postprocessedFiles = {}

        tP = load_pkl(self.__predictedFiles['trainP'])
        vP = load_pkl(self.__predictedFiles['validP'])
        tY = load_pkl(self.__preprocessedFiles['trainY'])
        vY = load_pkl(self.__preprocessedFiles['validY'])

        R2Train = []
        R2Valid = []
        yTrain = []
        yPredict = []

        for i in range(shape(tP)[-1]):  # Iterate over each output
            def get_r2(Y, P):
                YSlice = Y[..., i].flatten()
                PSlice = P[..., i].flatten()
                # for j in range(len(YSlice)):
                #     print(YSlice[j], PSlice[j])
                ss_res_train = sum((YSlice - PSlice)**2)  # Residual Sum of Squares
                ss_tot_train = sum((YSlice - mean(YSlice))**2)  # Total Sum of Squares
                return 1 - ss_res_train / ss_tot_train
            YSlice = tY[..., i].flatten()
            PSlice = tP[..., i].flatten()
            for j in range(len(YSlice)):
                yTrain.append(YSlice[j])
                yPredict.append(PSlice[j])
            R2Train.append(get_r2(tP, tY))
            R2Valid.append(get_r2(vP, vY))
            print(f"R² value for output {i} (train): {R2Train[-1]}")
            print(f"R² value for output {i} (valid): {R2Valid[-1]}")
        

        plt.figure(figsize=(10, 5))
        plt.scatter(yTrain, yPredict, alpha=0.5)
        plt.savefig('my_plot.png')
        return
