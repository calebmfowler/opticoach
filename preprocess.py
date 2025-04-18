from aggregate import Aggregator
from copy import deepcopy
from pandas import Series, DataFrame, json_normalize, read_json
from utilities import bound, load_json, serialize_dictionary, tabulate_dictionary

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
        startYear, endYear = 1935, 2024
        coachHistoryJSON = load_json('files/coach_history.json')
        
        school_coach_year = bound(tabulate_dictionary(coachHistoryJSON, columnDepth=3, indexDepth=1, valueDepth=0), startYear, endYear)
        print(school_coach_year)
        # Need to remap to schools to consistent names
        role_coach_year = bound(tabulate_dictionary(coachHistoryJSON, columnDepth=3, indexDepth=1, valueDepth=2), startYear, endYear)
        print(role_coach_year)
        # Need to remap to roles to consistent names

        heismanJSON = load_json('files/heismans.json')
        heismanSchool_year = bound(serialize_dictionary(heismanJSON, indexDepth=0, valueDepth=2), startYear, endYear)
        # Need to remap to schools to consistent names
        heismanSchool_year = heismanSchool_year.reindex(school_coach_year.index)
        heisman_coach_year = bound(school_coach_year.eq(heismanSchool_year, axis=0), startYear, endYear)
        print(heisman_coach_year)
        
        pollsJSON = load_json('files/polls.json')
        pollsJSON = { year: weeks.get('Final') for year, weeks in pollsJSON.items() if 'Final' in weeks }
        rank_school_year = tabulate_dictionary(pollsJSON, columnDepth=3, indexDepth=0, valueDepth=2).sort_index().loc[startYear:endYear]
        
        recordsJSON = load_json('files/records.json')
        record = tabulate_dictionary(recordsJSON, columnDepth=0, indexDepth=1, valueDepth=(2, None)).sort_index().loc[startYear:endYear]

        # convert all tables to numerical datatypes

        self.preprocessedFiles = {
            "trainX": "files/trainX.pkl",
            "trainY": "files/trainY.pkl",
            "validX": "files/validX.pkl",
            "validY": "files/validY.pkl"
        }
        return