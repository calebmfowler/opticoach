from aggregate import Aggregator
from copy import deepcopy
from numpy import array as nparr
from pandas import Series, to_numeric, DataFrame, json_normalize, read_json
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
        startYear, endYear = 1936, 2024
        coachHistoryJSON = load_json('files/coach_history.json')
        
        school_coach_year = bound(tabulate_dictionary(coachHistoryJSON, columnDepth=3, indexDepth=1, valueDepth=0), startYear, endYear)
        # print('school_coach_year\n', school_coach_year)
        # Need to remap to schools to consistent names
        role_coach_year = bound(tabulate_dictionary(coachHistoryJSON, columnDepth=3, indexDepth=1, valueDepth=2), startYear, endYear)
        # print('role_coach_year\n', role_coach_year)
        # Need to remap to roles to consistent names

        heismanJSON = load_json('files/heismans.json')
        heismanSchool_year = bound(serialize_dictionary(heismanJSON, indexDepth=0, valueDepth=2), startYear, endYear)
        # Need to remap to schools to consistent names
        heismanSchool_year = heismanSchool_year.reindex(school_coach_year.index)
        heismanCoach_year = bound(school_coach_year.eq(heismanSchool_year, axis=0), startYear, endYear).astype(int)
        # print('heismanCoach_year\n', heismanCoach_year)
        
        finalPollsJSON = load_json('files/final_polls.json')
        rank_school_year = bound(tabulate_dictionary(finalPollsJSON, columnDepth=(2, None), indexDepth=0, valueDepth=1), startYear, endYear)
        rank_school_year = rank_school_year.apply(to_numeric, errors='coerce').astype('Int64')
        # print('rank_school_year\n', rank_school_year)

        recordsJSON = load_json('files/records.json')
        recordsDF = bound(tabulate_dictionary(recordsJSON, columnDepth=0, indexDepth=1, valueDepth=(2, None)), startYear, endYear)
        offense = lambda record : [int(game[1].split('-')[0]) for game in record]
        defense = lambda record : [int(game[1].split('-')[1]) for game in record]
        def offensiveScorer(record):
            try:
                return sum(nparr(offense(record), dtype=int))
            except:
                return 0
        def defensiveScorer(record):
            try:
                return sum(nparr(defense(record), dtype=int))
            except:
                return 0
        def winCounter(record):
            try:
                return sum(nparr(
                    [offense > defense for offense, defense in zip(offense(record), defense(record))]
                , dtype=int))
            except:
                return 0
        def lossCounter(record):
            try:
                return sum(nparr(
                    [offense < defense for offense, defense in zip(offense(record), defense(record))]
                , dtype=int))
            except:
                return 0
        
        offensiveScore_school_year = recordsDF.applymap(offensiveScorer)
        # print('offensiveScore_school_year\n', offensiveScore_school_year)
        defensiveScore_school_year = recordsDF.applymap(defensiveScorer)
        # print('defensiveScore_school_year\n', defensiveScore_school_year)
        wins_school_year = recordsDF.applymap(winCounter)
        # print('wins_school_year\n', wins_school_year)
        losses_school_year = recordsDF.applymap(lossCounter)
        # print('losses_school_year\n', losses_school_year)

        self.preprocessedFiles = {
            "trainX": "files/trainX.pkl",
            "trainY": "files/trainY.pkl",
            "validX": "files/validX.pkl",
            "validY": "files/validY.pkl"
        }
        return