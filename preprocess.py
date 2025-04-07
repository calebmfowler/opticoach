class Preprocessor:
    '''
    # `class Preprocessor`
    This `class` preprocesses the aggregated data into a format suitable for model training.

    ## `dict preprocessedFiles`
    This `dict` stores `string` keys of all preprocessed files and `string` values of file names

    ## `void preprocess()`
    This `void` function preprocesses all data, updating the files referenced by preprocessedFiles
    '''

    preprocessedFiles = {
        "trainX": "trainX.pkl",
        "trainY": "trainY.pkl",
        "validX": "validX.pkl",
        "validy": "validY.pkl",
    }

    def preprocess(aggregatedFiles):
        return