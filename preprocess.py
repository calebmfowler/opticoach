from aggregate import Aggregator
from copy import deepcopy
import numpy as np
from numpy import array as nparr, nan, shape, unique
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
        
        self.preprocessedFiles = {
            "trainX" : "files/trainX.pkl",
            "trainY" : "files/trainY.pkl",
            "validX" : "files/validX.pkl",
            "validY" : "files/validY.pkl",
            "XTypes" : "files/XTypes.pkl",
            "YTypes" : "files/YTypes.pkl"
        }
        return

    def preprocess(self):
        metrics, metricTypes, backgroundMask, foresightMask, predictionMask = [], [], [], [], []

        # === UTILITIES ===

        bound_years = lambda data : bound_data(data, self.startYear, self.endYear)

        tabulate = lambda *args, **kwargs : DataFrame(bound_years(tabulate_dict(*args, **kwargs)))

        serialize = lambda *args, **kwargs : Series(bound_years(serialize_dict(*args, **kwargs)))

        recolumnate = lambda *args, **kwargs : DataFrame(bound_years(recolumnate_df(*args, **kwargs)))

        def map_columns(df, map):
            df = DataFrame(df)
            df.columns = df.columns.map(map)
            return df

        def add_metric(metric, metricType, backgroundInclusion, foresightInclusion, predictionInclusion, map=None, name=None):
            if map:
                metric = DataFrame(metric).map(map)

            if name:
                print(f"{name}\n{metric}")
            
            metrics.append(metric)
            metricTypes.append(metricType)
            backgroundMask.append(backgroundInclusion)
            foresightMask.append(foresightInclusion)
            predictionMask.append(predictionInclusion)

            return metric
        
        # === FILE IMPORTS ===

        coachJSON = load_json('files/trimmed_coach_dictionary.json')
        schoolMapJSON = load_json('files/mapping_schools.json')
        pollsJSON = load_json('files/polls.json')
        recordsJSON = load_json('files/records.json')

        # === MAPS ===

        def school_map(school):
            if school != school:
                return ""
            elif school in schoolMapJSON:
                return schoolMapJSON[school]
            else:
                return str(school)
            
        def role_map(role):
            if role != role:
                return ["", nan]
            else:
                role = str(role)
                i = role.find('/')
                if i != -1:
                    role = role[:i]
                if role == 'HC':
                    return [role, 0]
                elif role in ['OC', 'DC']:
                    return [role, 1]
                else:
                    return [role, 2]
            
        def rank_map(num):
            if num != num:
                return 30
            else:
                return int(num)

        def performance_map(record, year, coach):
            '''this function returns a list of statistical features for a given coach in a given year. It includes scoring offense,
            scoring defense, win percentage, talent level, and strength of schedule.'''
            if record != record or record == []:
                return [nan, nan, nan]
            
            elif not isinstance(record[0], list):
                game = record
                score = str(game[1]).split('-')
                offense, defense, win = int(score[0]), int(score[1]), 0

                if offense > defense:
                    win += 1
                elif offense == defense:
                    win += 0.5
                
                return [offense, defense, win]
            
            else:
                print(f"record\n{record}")
                gameCount = len(record)
                scoringOffense = 0.
                scoringDefense = 0.
                winRate = 0.

                for game in record:
                    print(f"game\n{game}")
                    score = str(game[1]).split('-')
                    offense, defense = int(score[0]), int(score[1])
                    scoringOffense += offense
                    scoringDefense += defense
                    if offense > defense:
                        winRate += 1
                    elif offense == defense:
                        winRate += .5
                
                scoringOffense /= gameCount
                scoringDefense /= gameCount
                winRate /= gameCount

                return [scoringOffense, scoringDefense, winRate]

        def annual_performance_map(season):
            season = Series(season)
            year = int(season.name)
            recordFeaturesDict = {}
            for coach in season.index:
                recordFeaturesDict[coach] = performance_map(season[coach], year, coach)
            return Series(recordFeaturesDict)

        def win_rate_map(record):
            return record[2]

        def avg_opponent_win_rate_map(record, year):
            gameCount = len(record)
            avgOpponentWinRate = 0
            for game in record:
                opponentSchool = str(game[0])
                opponentCoach = coach_school_year.at[year, opponentSchool]
                avgOpponentWinRate += winRate_coach_year.at[year, opponentCoach]
            avgOpponentWinRate /= gameCount
            return avgOpponentWinRate
        
        def annual_avg_opponent_win_rate_map(season):
            season = Series(season)
            year = int(season.name)
            recordFeaturesDict = {}
            for coach in season.index:
                recordFeaturesDict[coach] = avg_opponent_win_rate_map(season[coach], year)
            return Series(recordFeaturesDict)

        def sos_map(record, year, coach):
            teamSos = avgOpponentWinRate_coach_year.at[year, coach]

            avgOpponentSos = 0
            for game in record:
                opponentSchool = str(game[0])
                opponentCoach = coach_school_year.at[year, opponentSchool]
                avgOpponentSos += avgOpponentWinRate_coach_year.at[year, opponentCoach]
            
            return 2/3 * teamSos + 1/3 * avgOpponentSos

        def annual_sos_map(season):
            season = Series(season)
            year = int(season.name)
            sosDict = {}
            for coach in season.index:
                sosDict[coach] = sos_map(season[coach], year, coach)

        # === METRICS COMPILATION ===

        school_coach_year = tabulate(coachJSON, columnDepth=3, indexDepth=0, valueDepth=1)
        school_coach_year = add_metric(school_coach_year, str, True, True, False, school_map, "school_coach_year")
        
        role_coach_year = tabulate(coachJSON, columnDepth=3, indexDepth=0, valueDepth=2)
        role_coach_year = add_metric(
            role_coach_year,
            [str, int],                         # metricType
            [True, True],                       # backgroundMask    
            [False, False],                     # foresightMask
            [False, False],                     # predictionMask
            role_map,
            "role_coach_year"
        )
        
        rank_school_year = tabulate(pollsJSON, columnDepth=(2, None), indexDepth=0, valueDepth=1)
        rank_school_year = rank_school_year.apply(to_numeric, errors='coerce')
        rank_school_year = map_columns(rank_school_year, school_map)
        rank_coach_year = recolumnate(rank_school_year, school_coach_year)
        rank_coach_year = add_metric(rank_coach_year, int, True, False, True, rank_map, "rank_coach_year")

        record_school_year = tabulate(recordsJSON, columnDepth=0, indexDepth=1, valueDepth=(2, None))
        record_school_year = map_columns(record_school_year, school_map)
        record_coach_year = recolumnate(record_school_year, school_coach_year)
        performance_coach_year = record_coach_year.apply(annual_performance_map, axis=1)
        performance_coach_year = add_metric(
            performance_coach_year,
            [float, float, float],              # metricType
            [True, True, True],                 # backgroundMask
            [False, False, False],              # foresightMask
            [False, False, True],               # predictionMask
            name="record_coach_year"
        )

        coach_school_year = tabulate(coachJSON, columnDepth=1, indexDepth=0, valueDepth=3)
        winRate_coach_year = performance_coach_year.map(win_rate_map)
        avgOpponentWinRate_coach_year = record_coach_year.apply(annual_avg_opponent_win_rate_map, axis=1)
        sos_coach_year = record_coach_year.apply(annual_sos_map, axis=1)
        sos_coach_year = add_metric(sos_coach_year, float, True, False, False, name="sos_coach_year")

        # === PACKAGING METRICS ===

        # --- Listing feature and label types ---

        XTypes, YTypes = [], []

        for metricType, background, foresight, prediction in zip(metricTypes, backgroundMask, foresightMask, predictionMask):
            if isinstance(metricType, list):
                for subMetricType, subBackground, subForesight, subPrediction in zip(metricType, background, foresight, prediction):
                    if subBackground:
                        XTypes.append(subMetricType)
                    if subForesight:
                        XTypes.append(subMetricType)
                    if subPrediction:
                        YTypes.append(subMetricType)
            else:   
                if background:
                    XTypes.append(metricType)
                if foresight:
                    XTypes.append(metricType)
                if prediction:
                    YTypes.append(metricType)
        
        save_pkl(XTypes, "files/XTypes.pkl")
        save_pkl(YTypes, "files/YTypes.pkl")

        # --- Compiling features and labels ---

        X, Y = [], []

        for i, coach in enumerate(school_coach_year.columns):

            schools = school_coach_year[coach]
            schoolChanges = schools != schools.shift()
            changeYears = schools.index[schoolChanges][1:]

            for changeYear in changeYears:

                newSchool = school_coach_year.at[changeYear, coach]
                if (newSchool == "" or
                    changeYear - self.backgroundYears < self.startYear or
                    changeYear + self.predictionYears > self.endYear):
                    continue
                
                backgroundYears = range(changeYear - self.backgroundYears, changeYear)
                predictionYears = range(changeYear, changeYear + self.predictionYears)

                predictionRoles = Series(role_coach_year.loc[predictionYears, coach]).values
                predictionSchools = Series(school_coach_year.loc[predictionYears, coach]).values
                if (not all([role == 'HC' for role in predictionRoles]) or
                    not all([school == newSchool for school in predictionSchools])):
                    continue
                
                print(f"coach {i}, change {changeYear}, ({coach})")
                XSample, YSample = [], []
                for metric, metricType, background, foresight, prediction in zip(metrics, metricTypes, backgroundMask, foresightMask, predictionMask):
                    backgroundMetric = list(Series(DataFrame(metric).loc[backgroundYears, coach]).values)
                    predictionMetric = list(Series(DataFrame(metric).loc[predictionYears, coach]).values)

                    if background:
                        if isinstance(backgroundMetric[0], list):
                            for subMetric in [list(subMetric) for subMetric in zip(*backgroundMetric)]:
                                XSample.append(subMetric)
                        else:
                            XSample.append(backgroundMetric)
                    
                    if foresight:
                        if isinstance(predictionMetric[0], list):
                            for subMetric, subType in zip([list(subMetric) for subMetric in zip(*predictionMetric)], metricType):
                                foresightPadding = [subType()] * (self.backgroundYears - self.predictionYears)
                                XSample.append(foresightPadding + subMetric)
                        else:
                            foresightPadding = [metricType()] * (self.backgroundYears - self.predictionYears)
                            XSample.append(foresightPadding + predictionMetric)

                    if prediction:
                        if isinstance(predictionMetric[0], list):
                            for subMetric in [list(subMetric) for subMetric in zip(*predictionMetric)]:
                                YSample.append(subMetric)
                        else:
                            YSample.append(predictionMetric)

                X.append(nparr(XSample).T)
                Y.append(nparr(YSample).T)

        X = nparr(X)
        Y = nparr(Y)
        print(f"X (shape = {shape(X)})\n{X}")
        print(f"Y (shape = {shape(Y)})\n{Y}")

        trainX, validX, trainY, validY = train_test_split(X, Y, test_size=0.2)

        save_pkl(trainX, "files/trainX.pkl")
        save_pkl(validX, "files/validX.pkl")
        save_pkl(trainY, "files/trainY.pkl")
        save_pkl(validY, "files/validY.pkl")
        return