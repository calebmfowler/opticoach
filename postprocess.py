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

    def get_r2(self, Y, P):
        """
        Calculate the R² score for predictions P and true values Y.
        """
        ss_res = sum((Y - P) ** 2)  # Residual Sum of Squares
        ss_tot = sum((Y - mean(Y)) ** 2)  # Total Sum of Squares
        return 1 - ss_res / ss_tot

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

        # Iterate over each output dimension
        for i in range(shape(tP)[-1]):
            # Extract slices for the current output dimension
            tYSlice = tY[..., i].flatten()
            tPSlice = tP[..., i].flatten()
            vYSlice = vY[..., i].flatten()
            vPSlice = vP[..., i].flatten()

            # Append true and predicted values for plotting
            yTrain.extend(tYSlice)
            yPredict.extend(tPSlice)

            # Calculate R² for training and validation
            R2Train.append(self.get_r2(tYSlice, tPSlice))
            R2Valid.append(self.get_r2(vYSlice, vPSlice))

            # Print R² values
            print(f"R² value for output {i} (train): {R2Train[-1]}")
            print(f"R² value for output {i} (valid): {R2Valid[-1]}")

        # Scatter plot of true vs predicted values
        plt.figure(figsize=(10, 5))
        plt.scatter(yTrain, yPredict, alpha=0.5)
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.title("True vs Predicted Values")
        plt.savefig('my_plot.png')
        return
