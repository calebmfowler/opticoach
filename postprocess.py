from copy import deepcopy
from opticoachmodel import OpticoachModel
from preprocess import Preprocessor
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

        # Load predictions and actual targets
        tP = np.load(self.__predictedFiles["trainP"])
        vP = np.load(self.__predictedFiles["validP"])
        tY = np.load(self.__preprocessedFiles["trainY"])
        vY = np.load(self.__preprocessedFiles["validY"])

        # Ensure shapes match
        if tP.shape != tY.shape or vP.shape != vY.shape:
            raise ValueError("Shape mismatch between predictions and targets.")

        # Initialize dictionaries to store R² values
        r2_train = []
        r2_valid = []

        # Check if there are multiple outputs
        if len(tP.shape) > 1 and tP.shape[1] > 1:
            for i in range(tP.shape[1]):  # Iterate over each output
                # Calculate R² for training set
                ss_res_train = np.sum((tY[:, i] - tP[:, i]) ** 2)  # Residual Sum of Squares
                ss_tot_train = np.sum((tY[:, i] - np.mean(tY[:, i])) ** 2)  # Total Sum of Squares
                r2_train.append(1 - (ss_res_train / ss_tot_train))

                # Calculate R² for validation set
                ss_res_valid = np.sum((vY[:, i] - vP[:, i]) ** 2)  # Residual Sum of Squares
                ss_tot_valid = np.sum((vY[:, i] - np.mean(vY[:, i])) ** 2)  # Total Sum of Squares
                r2_valid.append(1 - (ss_res_valid / ss_tot_valid))

                # Print R² values
                print(f"R² value for output {i} (train): {r2_train[-1]}")
                print(f"R² value for output {i} (valid): {r2_valid[-1]}")

        else:
            # Single output case
            # Calculate R² for training set
            ss_res_train = np.sum((tY - tP) ** 2)  # Residual Sum of Squares
            ss_tot_train = np.sum((tY - np.mean(tY)) ** 2)  # Total Sum of Squares
            r2_train = 1 - (ss_res_train / ss_tot_train)

            # Calculate R² for validation set
            ss_res_valid = np.sum((vY - vP) ** 2)  # Residual Sum of Squares
            ss_tot_valid = np.sum((vY - np.mean(vY)) ** 2)  # Total Sum of Squares
            r2_valid = 1 - (ss_res_valid / ss_tot_valid)

            # Print R² values
            print(f"R² value (train): {r2_train}")
            print(f"R² value (valid): {r2_valid}")

        # Store R² values in postprocessedFiles
        self.postprocessedFiles["r2_train"] = r2_train
        self.postprocessedFiles["r2_valid"] = r2_valid

        return