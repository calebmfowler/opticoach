from copy import deepcopy
from opticoachmodel import OpticoachModel
from preprocess import Preprocessor

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
        self.postprocessedFiles = {}
        return