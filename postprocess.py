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

    def __init__(self, preprocessor, model):
        self.postprocessedFiles = {}
        self.__preprocessedFiles = preprocessor.preprocessedFiles
        self.__predictedFiles = model.predictedFiles

    def postprocess(self):
        self.postprocessedFiles = {}
        return