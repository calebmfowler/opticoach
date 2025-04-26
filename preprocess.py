from aggregate import Aggregator
from copy import deepcopy
from keras.src.layers import TextVectorization
import numpy as np
from numpy import array as nparr, hstack, insert, nan, shape, unique
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
            self.preprocessedFiles = {
                "trainX" : "files/trainX.pkl",
                "trainY" : "files/trainY.pkl",
                "validX" : "files/validX.pkl",
                "validY" : "files/validY.pkl",
                "XEmbeds" : "files/XEmbeds.pkl",
                "XVocabs" : "files/XVocabs.pkl"
            }
            self.__aggregatedFiles = Aggregator(arg).aggregatedFiles
            self.startYear = startYear
            self.endYear = endYear
            self.backgroundYears = backgroundYears
            self.predictionYears = predictionYears
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
        metrics, metricTypes, embedMask, backgroundMask, foresightMask, predictionMask, vocabularies = [], [], [], [], [], [], []

        # === UTILITIES ===

        bound_years = lambda data : bound_data(data, self.startYear, self.endYear)

        tabulate = lambda *args, **kwargs : DataFrame(bound_years(tabulate_dict(*args, **kwargs)))

        serialize = lambda *args, **kwargs : Series(bound_years(serialize_dict(*args, **kwargs)))

        recolumnate = lambda *args, **kwargs : DataFrame(bound_years(recolumnate_df(*args, **kwargs)))

        def map_columns(df, map):
            df = DataFrame(df)
            df.columns = df.columns.map(map)
            return df

        def add_metric(metric, metricType, metricEmbed, vocab, background, foresight, prediction, map=None, name=None):
            if map:
                metric = DataFrame(metric).map(map)

            if name:
                print(f"{name}\n{metric}")
            
            metrics.append(metric)
            metricTypes.append(metricType)
            embedMask.append(metricEmbed)
            backgroundMask.append(background)
            foresightMask.append(foresight)
            predictionMask.append(prediction)
            vocabularies.append(vocab)

            return metric
        
        # === FILE IMPORTS ===

        coachJSON = load_json('files/coach_history_regularized.json')
        schoolMapJSON = load_json('files/mapping_schools.json')
        pollsJSON = load_json('files/final_polls.json')
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
                return ["", -1]
            else:
                roleTitle = str(role)
                i = roleTitle.find('/')
                if i != -1:
                    roleTitle = roleTitle[:i]
                if roleTitle == 'HC':
                    return [roleTitle, 0]
                elif roleTitle in ['OC', 'DC']:
                    return [roleTitle, 1]
                else:
                    return [roleTitle, 2]
        
        def role_title_map(role):
            return role[0]
        
        def role_rank_map(role):
            return role[1]

        def rank_map(num):
            if isinstance(num, int):
                return num
            else:
                return 30

        def performance_map(record, year, coach):
            '''
            this function returns a list of statistical features for a given coach in a given year. It includes scoring offense,
            scoring defense, win percentage, talent level, and strength of schedule.
            '''
            if record != record or record == []:
                return [20, 30, 0.4]
            
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
                gameCount = len(record)
                scoringOffense = 0.
                scoringDefense = 0.
                winRate = 0.

                for game in record:
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
            if record != record or record == []:
                return 0.4
            
            elif not isinstance(record[0], list):
                game = record
                opponentSchool = str(game[0])
                if not opponentSchool in coach_school_year.columns:
                    return 0.4
                opponentCoach = coach_school_year.at[year, opponentSchool]
                if not opponentCoach in winRate_coach_year.columns:
                    return 0.4
                avgOpponentWinRate = winRate_coach_year.at[year, opponentCoach]
                return avgOpponentWinRate

            else:
                gameCount = len(record)
                avgOpponentWinRate = 0
                for game in record:
                    opponentSchool = str(game[0])
                    if not opponentSchool in coach_school_year.columns:
                        avgOpponentWinRate += 0.4
                        continue
                    opponentCoach = coach_school_year.at[year, opponentSchool]
                    if not opponentCoach in winRate_coach_year.columns:
                        avgOpponentWinRate += 0.4
                        continue
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
            if record != record or record == []:
                return 0.4
            
            teamSos = avgOpponentWinRate_coach_year.at[year, coach]

            avgOpponentSos = 0
            if not isinstance(record[0], list):
                game = record
                opponentSchool = str(game[0])
                if not opponentSchool in coach_school_year.columns:
                    avgOpponentSos = 0.4
                else:
                    opponentCoach = coach_school_year.at[year, opponentSchool]
                    if not opponentCoach in winRate_coach_year.columns:
                        avgOpponentSos = 0.4
                    else:
                        avgOpponentSos = avgOpponentWinRate_coach_year.at[year, opponentCoach]
            else:
                gameCount = len(record)
                for game in record:
                    opponentSchool = str(game[0])
                    if not opponentSchool in coach_school_year.columns:
                        avgOpponentSos += 0.4
                        continue
                    opponentCoach = coach_school_year.at[year, opponentSchool]
                    if not opponentCoach in winRate_coach_year.columns:
                        avgOpponentSos += 0.4
                        continue
                    avgOpponentSos += avgOpponentWinRate_coach_year.at[year, opponentCoach]
                avgOpponentSos /= gameCount
                
            return 2/3 * teamSos + 1/3 * avgOpponentSos

        def annual_sos_map(season):
            season = Series(season)
            year = int(season.name)
            sosDict = {}
            for coach in season.index:
                sosDict[coach] = sos_map(season[coach], year, coach)
            return Series(sosDict)

        # === METRICS COMPILATION ===

        # --- Vocabulary Generation ---

        school_coach_year = tabulate(coachJSON, columnDepth=3, indexDepth=0, valueDepth=1)
        school_coach_year = school_coach_year.map(school_map)
        rank_school_year = tabulate(pollsJSON, columnDepth=(2, None), indexDepth=0, valueDepth=1)
        rank_school_year = map_columns(rank_school_year, school_map)
        record_school_year = tabulate(recordsJSON, columnDepth=0, indexDepth=1, valueDepth=(2, None))
        record_school_year = map_columns(record_school_year, school_map)
        schoolVocabulary = unique(hstack((unique(school_coach_year), rank_school_year.columns, record_school_year.columns)))
        schoolVocabulary = insert(schoolVocabulary[1:], 0, ['', '[UNK]'])
        schoolVectorization = TextVectorization(standardize=None, split=None, vocabulary=schoolVocabulary)

        role_coach_year = tabulate(coachJSON, columnDepth=3, indexDepth=0, valueDepth=2)
        role_coach_year = role_coach_year.map(role_map)
        roleTitle_coach_year = role_coach_year.map(role_title_map)
        roleTitleVocabulary = insert(unique(roleTitle_coach_year)[1:], 0, ['', '[UNK]'])
        roleTitleVectorization = TextVectorization(standardize=None, split=None, vocabulary=roleTitleVocabulary)

        # --- Metric Enumeration ---

        schoolInt_coach_year = DataFrame(
            schoolVectorization(school_coach_year),
            columns=school_coach_year.columns,
            index=school_coach_year.index
        )
        schoolInt_coach_year = add_metric(
            schoolInt_coach_year,
            int,
            True,
            schoolVocabulary,
            True,
            True,
            False,
            name="schoolInt_coach_year"
        )

        roleTitleInt_coach_year = DataFrame(
            roleTitleVectorization(roleTitle_coach_year),
            columns=roleTitle_coach_year.columns,
            index=roleTitle_coach_year.index
        )
        roleTitleInt_coach_year = add_metric(
            roleTitleInt_coach_year,
            int,
            True,
            roleTitleVocabulary,
            True,
            False,
            False,
            name="roleTitleInt_coach_year"
        )

        roleRank_coach_year = role_coach_year.map(role_rank_map)
        roleRank_coach_year = add_metric(
            roleRank_coach_year,
            int,
            True,
            [-1, 0, 1, 2],
            True,
            False,
            False,
            name="roleRank_coach_year"
        )

        rank_school_year = rank_school_year.apply(to_numeric, errors='coerce')
        rank_coach_year = recolumnate(rank_school_year, school_coach_year)
        rank_coach_year = add_metric(
            rank_coach_year,
            int,
            False,
            [],
            True,
            False,
            True,
            rank_map,
            "rank_coach_year"
        )

        record_coach_year = recolumnate(record_school_year, school_coach_year)
        performance_coach_year = record_coach_year.apply(annual_performance_map, axis=1)
        performance_coach_year = add_metric(
            performance_coach_year,
            [float, float, float],              # metricType
            [False, False, False],              # metricEmbed
            [[], [], []],                       # vocabularies
            [True, True, True],                 # backgroundMask
            [False, False, False],              # foresightMask
            [False, False, True],               # predictionMask
            name="performance_coach_year"
        )

        coach_school_year = tabulate(coachJSON, columnDepth=1, indexDepth=0, valueDepth=3)
        winRate_coach_year = performance_coach_year.map(win_rate_map)
        avgOpponentWinRate_coach_year = record_coach_year.apply(annual_avg_opponent_win_rate_map, axis=1)
        sos_coach_year = record_coach_year.apply(annual_sos_map, axis=1)
        sos_coach_year = add_metric(
            sos_coach_year,
            float,
            False,
            [],
            True,
            False,
            False,
            name="sos_coach_year"
        )

        # === PACKAGING METRICS ===

        # --- Listing feature and label types ---

        XTypes, XEmbeds, XVocabs = [], [], []

        for i, metricType in enumerate(metricTypes):
            if isinstance(metricType, list):
                for j, subMetricType in enumerate(metricType):
                    if backgroundMask[i][j]:
                        XTypes.append(subMetricType)
                        XEmbeds.append(embedMask[i][j])
                        XVocabs.append(vocabularies[i][j])
                    if foresightMask[i][j]:
                        XTypes.append(subMetricType)
                        XEmbeds.append(embedMask[i][j])
                        XVocabs.append(vocabularies[i][j])
            else:
                if backgroundMask[i]:
                    XTypes.append(metricType)
                    XEmbeds.append(embedMask[i])
                    XVocabs.append(vocabularies[i])
                if foresightMask[i]:
                    XTypes.append(metricType)
                    XEmbeds.append(embedMask[i])
                    XVocabs.append(vocabularies[i])
        
        save_pkl(XEmbeds, self.preprocessedFiles['XEmbeds'])
        save_pkl(XVocabs, self.preprocessedFiles['XVocabs'])

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
                if (not all([role[0] == 'HC' for role in predictionRoles]) or
                    not all([school == newSchool for school in predictionSchools])):
                    continue
                
                print(f"coach {i}, change {changeYear}, ({coach})")
                XSample, YSample = [], []
                # metricType, background, foresight, prediction
                for i, metric in enumerate(metrics):
                    backgroundMetric = list(Series(DataFrame(metric).loc[backgroundYears, coach]).values)
                    predictionMetric = list(Series(DataFrame(metric).loc[predictionYears, coach]).values)

                    if isinstance(backgroundMetric[0], list):
                        for subBackgroundMetric, subPredictionMetric, subMetricType, subBackground, subForesight, subPrediction in zip(
                            [list(subMetric) for subMetric in zip(*backgroundMetric)],
                            [list(subMetric) for subMetric in zip(*predictionMetric)],
                            metricTypes[i], backgroundMask[i], foresightMask[i], predictionMask[i]
                        ):
                            if subBackground:
                                XSample.append(subBackgroundMetric)
                            if subForesight:
                                foresightPadding = [subMetricType()] * (self.backgroundYears - self.predictionYears)
                                XSample.append(foresightPadding + subPredictionMetric)
                            if subPrediction:
                                YSample.append(subPredictionMetric)
                    else:
                        if backgroundMask[i]:
                            XSample.append(backgroundMetric)
                        if foresightMask[i]:
                            foresightPadding = [metricTypes[i]()] * (self.backgroundYears - self.predictionYears)
                            XSample.append(foresightPadding + predictionMetric)
                        if predictionMask[i]:
                            YSample.append(predictionMetric)

                X.append([list(row) for row in zip(*XSample)])
                Y.append([list(row) for row in zip(*YSample)])

        X = nparr(X)
        Y = nparr(Y)

        trainX, validX, trainY, validY = train_test_split(X, Y, test_size=0.2)

        save_pkl(trainX, self.preprocessedFiles['trainX'])
        save_pkl(validX, self.preprocessedFiles['validX'])
        save_pkl(trainY, self.preprocessedFiles['trainY'])
        save_pkl(validY, self.preprocessedFiles['validY'])
        return