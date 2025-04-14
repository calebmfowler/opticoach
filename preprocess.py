from aggregate import Aggregator
from copy import deepcopy
from pandas import Series, DataFrame, json_normalize, read_json
from utilities import load_json

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
    This `void` function is called as the Preprocessor constructor, initializing the variable __aggregatedFiles
    from an Aggregator instance. If provided a Preprocessor, it operates as a copy constructor.

    ### void preprocess()
    This `void` function preprocesses all data, updating the files referenced by preprocessedFiles
    '''
    
    def __init__(self, arg):
        if type(arg) == Aggregator:
            self.__aggregatedFiles = Aggregator(arg).aggregatedFiles
            self.preprocessedFiles = {}
        elif type(arg) == Preprocessor:
            self.__aggregatedFiles = deepcopy(arg.__aggregatedFiles)
            self.preprocessedFiles = deepcopy(arg.preprocessedFiles)
        else:
            raise Exception("Incorrect arguments for Preprocessor.__init__(self, aggregator)")
        return

    def preprocess(self):
        # Here we load in the data compiled by our Aggregator
        testCoachHistoryJSON = load_json(self.__aggregatedFiles['testCoachHistory'])
        testCoachHistoryDF = DataFrame(testCoachHistoryJSON).T
        testCoachHistoryDF.rename_axis('school.year.role')
        testCoachHistoryDF.reset_index().rename(columns={0: 'name'})
        
        testCoachHistoryDF = testCoachHistoryDF.stack().apply(Series)


        testGameHistoryDF = load_json(self.__aggregatedFiles['testGameHistory'])
        testGameHistoryDF = json_normalize(testGameHistoryDF, max_level=1)

        print(testCoachHistoryDF)
        print(testGameHistoryDF)

        self.preprocessedFiles = {
            "trainX": "files/trainX.pkl",
            "trainY": "files/trainY.pkl",
            "validX": "files/validX.pkl",
            "validY": "files/validY.pkl"
        }
        return