class Model:
    '''
    # `class Model`
    This `class` provides interfaces for training and using the machine learning model.

    ## `dict modelFiles`
    This `dict` stores `string` keys of all model relate files and `string` values of file names.
    These files will include the model(s) themselves as well as any scalers. However, all scaling
    and unscaling should be done internal to this class.

    ## `dict predictedFiles`
    This `dict` stores `string` keys of predictions files and `string` values of file names.
    These prediction files will corresponding to the file names provided by the Preprocessor.

    ## `void train()`
    This `void` function trains the model(s)

    ## `void predict()`
    This `void` function makes predictions, updating the files referenced by predictedFiles.
    '''

    modelFiles = {
        "model": "model.pkl",
        "scalerX": "scalerX.pkl",
        "scalerY": "scalerY.pkl",
    }

    predictedFiles = {
        "trainP": "trainP.pkl",
        "validP": "validP.pkl",
    }

    def train(preprocessedFiles):
        return
    
    def predict(preprocessedFiles):
        return