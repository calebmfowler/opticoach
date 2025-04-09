class Preprocessor:
    '''
    ### class Preprocessor
    This `class` preprocesses the aggregated data into a format suitable for model training.

    ### dict __aggregatedFiles
    This private `dict` variable stores `string` keys of all aggregated files and `string` values of 
    file names as sourced from the Aggregator instance.

    ### dict preprocessedFiles
    This `dict` variable stores `string` keys of all preprocessed files and `string` values of file names.

    ### void __init__(self, Aggregator aggregator)
    This `void` function is called as the Preprocessor constructor, initializing the variable
    __aggregatedFiles from an Aggregator instance.

    ### void preprocess()
    This `void` function preprocesses all data, updating the files referenced by preprocessedFiles
    '''
    
    def __init__(self, aggregator):
        self.preprocessedFiles = {
            "trainX": "trainX.pkl",
            "trainY": "trainY.pkl",
            "validX": "validX.pkl",
            "validY": "validY.pkl"
        }
        self.__aggregatedFiles = aggregator.aggregatedFiles
        return

    def preprocess(self):
        return