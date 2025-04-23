from aggregate import Aggregator
from copy import deepcopy
from numpy import array as nparr
from pandas import DataFrame, Series, to_numeric
from sklearn.model_selection import train_test_split
from utilities import bound_data, load_json, recolumnate_df, save_pkl, serialize_dict, tabulate_dict

class Preprocessor:
    '''
    ### class Preprocessor
    This `class` preprocesses the aggregated data into a format suitable for model training.

    ### dict __aggregatedFiles
    This private `dict` variable stores `string` keys of all aggregated files and `string` values of 
    file names as sourced from the Aggregator instance.

    ### int startYear
    This `int` variable stores the first year to be included.

    ### int endYear
    This  `int` variable stores the first year to be disincluded.

    ### int backgroundYears
    Thi `int` variable stores the number of years prior to a move to be considerd.

    ### int predictionYears
    Thi `int` variable stores the number of years after a move to be considered.

    ### dict preprocessedFiles
    This `dict` variable stores `string` keys of all preprocessed files and `string` values of file names.

    ### void __init__(self, Aggregator aggregator)
    This `void` function is called as the Preprocessor constructor, initializing the variable __aggregatedFiles
    from an Aggregator instance. If provided a Preprocessor, it operates as a copy constructor.

    ### void preprocess()
    This `void` function preprocesses all data, updating the files referenced by preprocessedFiles
    '''
    
    def __init__(self, arg, startYear=1936, endYear=2024, backgroundYears=15, predictionYears=5):
        if type(arg) == Aggregator:
            self.__aggregatedFiles = Aggregator(arg).aggregatedFiles
            self.startYear = startYear
            self.endYear = endYear
            self.backgroundYears = backgroundYears
            self.predictionYears = predictionYears
            self.preprocessedFiles = {}
        elif type(arg) == Preprocessor:
            self.__aggregatedFiles = deepcopy(arg.__aggregatedFiles)
            self.startYear = deepcopy(arg.startYear)
            self.endYear = deepcopy(arg.endYear)
            self.backgroundYears = deepcopy(arg.backgroundYears)
            self.predictionYears = deepcopy(arg.predictionYears)
            self.preprocessedFiles = deepcopy(arg.preprocessedFiles)
        else:
            raise Exception("Incorrect arguments for Preprocessor.__init__(self, aggregator)")
        return

    def preprocess(self):
        # === UTILITIES ===
        bound_years = lambda df : bound_data(df, self.startYear, self.endYear)
        tabulate = lambda *args, **kwargs : DataFrame(bound_years(tabulate_dict(*args, **kwargs)))
        serialize = lambda *args, **kwargs : Series(bound_years(serialize_dict(*args, **kwargs)))
        recolumnate = lambda *args, **kwargs : DataFrame(bound_years(recolumnate_df(*args, **kwargs)))

        # === FILE IMPORTS ===
        coachJSON = load_json('files/trimmed_coach_dictionary.json')
        schoolMapJSON = load_json('files/mapping_schools.json')
        heismanJSON = load_json('files/heismans.json')
        pollsJSON = load_json('files/final_polls.json')
        recordsJSON = load_json('files/records.json')

        # === MAPS ===
        def schoolMap(school):
            if school in schoolMapJSON:
                return schoolMapJSON[school]
            elif school != school:
                return ""
            else:
                return str(school)
            
        def roleMap(role):
            if role != role:
                return ""
            else:
                role = str(role)
                i = role.find('/')
                if i != -1:
                    return role[:i]
                else:
                    return role
            
        def rankMap(num):
            if num != num:
                return 30
            else:
                return int(num)
        
        def offensiveScores(record):
            return [int(game[1].split('-')[0]) for game in record]
        
        def defensiveScores(record):
            return [int(game[1].split('-')[1]) for game in record]
        
        def totalOffensiveScoreMap(record):
            try:
                return sum(nparr(offensiveScores(record), dtype=int))
            except:
                return 0

        def totalDefensiveScoreMap(record):
            try:
                return sum(nparr(defensiveScores(record), dtype=int))
            except:
                return 0

        def lossCountMap(record):
            try:
                return sum(nparr(
                    [offense < defense for offense, defense in zip(offensiveScores(record), defensiveScores(record))]
                , dtype=int))
            except:
                return 0
    
        def winCountMap(record):
            try:
                return sum(nparr(
                    [offense > defense for offense, defense in zip(offensiveScores(record), defensiveScores(record))]
                , dtype=int))
            except:
                return 0

        # === METRICS COMPILATION ===
        # --- "school" by "coach" by int(year) ---
        school_coach_year = tabulate(coachJSON, columnDepth=3, indexDepth=0, valueDepth=1)
        school_coach_year = school_coach_year.map(schoolMap)
        print('\nschool_coach_year\n', school_coach_year)
        metrics = []
        
        # --- "role" by "coach" by int(year) ---
        role_coach_year = tabulate(coachJSON, columnDepth=3, indexDepth=0, valueDepth=2)
        role_coach_year = role_coach_year.map(roleMap)
        metrics.append(role_coach_year)
        print('\nrole_coach_year\n', role_coach_year)

        # --- int(heisman) by "coach" by int(year) ---
        heismanSchool_year = serialize(heismanJSON, indexDepth=0, valueDepth=2)
        heismanSchool_year = heismanSchool_year.map(schoolMap)
        heismanSchool_year = heismanSchool_year.reindex(school_coach_year.index)
        heisman_coach_year = bound_years(school_coach_year.eq(heismanSchool_year, axis=0)).astype(int)
        metrics.append(heisman_coach_year)
        print('\nheisman_coach_year\n', heisman_coach_year)
        
        # --- int(rank) by "coach" by int(year) ---
        rank_school_year = tabulate(pollsJSON, columnDepth=(2, None), indexDepth=0, valueDepth=1)
        rank_school_year = rank_school_year.apply(to_numeric, errors='coerce').astype('Int64')
        rank_coach_year = bound_years(recolumnate(rank_school_year, school_coach_year))
        rank_coach_year = rank_coach_year.map(rankMap)
        metrics.append(rank_coach_year)
        print('\nrank_coach_year\n', rank_coach_year)

        # --- int(offensiveScore) by coach by year ---
        recordsDF = tabulate(recordsJSON, columnDepth=0, indexDepth=1, valueDepth=(2, None))
        records_coach_year = bound_years(recolumnate(recordsDF, school_coach_year))
        offensiveScore_coach_year = records_coach_year.map(totalOffensiveScoreMap)
        metrics.append(offensiveScore_coach_year)
        print('\noffensiveScore_coach_year\n', offensiveScore_coach_year)

        # --- int(defensiveScore) by coach by year ---
        defensiveScore_coach_year = records_coach_year.map(totalDefensiveScoreMap)
        metrics.append(defensiveScore_coach_year)
        print('\ndefensiveScore_coach_year\n', defensiveScore_coach_year)
        
        # --- int(winCount) by coach by year ---
        wins_coach_year = records_coach_year.map(winCountMap)
        metrics.append(wins_coach_year)
        print('\nwins_coach_year\n', wins_coach_year)
        
        # --- int(lossCount) by coach by year ---
        losses_coach_year = records_coach_year.map(lossCountMap)
        metrics.append(losses_coach_year)
        print('\nlosses_coach_year\n', losses_coach_year)

        # === PACKAGING METRICS ===
        X, Y = [], []
        for coach, i in zip(school_coach_year.columns, range(len(school_coach_year.columns))):
            print(f"coach {i} = {coach}")
            allSchools = school_coach_year[coach]
            allSchoolChanges = allSchools != allSchools.shift()
            changeYears = allSchools.index[allSchoolChanges][1:]

            for changeYear in changeYears:
                if (changeYear - self.__backgroundYears < self.__startYear or
                    changeYear + self.__predictionYears > self.__endYear):
                    continue

                backgroundYears = range(changeYear - self.__backgroundYears, changeYear)
                predictionYears = range(changeYear, changeYear + self.__predictionYears)
                backgroundSchools = school_coach_year.loc[backgroundYears, coach]
                predictionSchools = school_coach_year.loc[predictionYears, coach]

                newSchool = predictionSchools.iloc[0]
                if (predictionSchools.nunique() != 1 or not isinstance(newSchool, str) and newSchool != newSchool):
                    continue

                featureSet = [backgroundSchools.values]
                for metric in metrics:
                    featureSet.append(metric.loc[backgroundYears, coach].values)
                featureSet = nparr(featureSet).T
                featureSet = nparr([*featureSet.flatten(), newSchool])

                labelSet = []
                for metric in metrics:
                    labelSet.append(metric.loc[predictionYears, coach].values)
                labelSet = nparr(labelSet).T

                X.append(featureSet)
                Y.append(labelSet)

        X = nparr(X)
        Y = nparr(Y)

        trainX, validX, trainY, validY = train_test_split(X, Y, test_size=0.2)
        save_pkl(trainX, "files/trainX.pkl")
        save_pkl(validX, "files/validX.pkl")
        save_pkl(trainY, "files/trainY.pkl")
        save_pkl(validY, "files/validY.pkl")

        self.preprocessedFiles = {
            "trainX": "files/trainX.pkl",
            "trainY": "files/trainY.pkl",
            "validX": "files/validX.pkl",
            "validY": "files/validY.pkl"
        }
        return