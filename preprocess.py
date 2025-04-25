from aggregate import Aggregator
from copy import deepcopy
import numpy as np
from numpy import array as nparr, nan, unique
from pandas import DataFrame, Series, to_numeric
from recordUtilities import BCS_sos, total_talent, success_level
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

        map_columns = lambda df, map : DataFrame(df).columns.map(map)

        def add_metric(metric, type, backgroundInclusion, foresightInclusion, predictionInclusion, map=None, name=None):
            if map:
                metric = DataFrame(metric).map(map)
            if name:
                print(f"{name}\n{metric}")
            metrics.append(metric)
            metricTypes.append(type)
            backgroundMask.append(backgroundInclusion)
            foresightMask.append(foresightInclusion)
            predictionMask.append(predictionInclusion)
        
        # === FILE IMPORTS ===
        coachJSON = load_json('files/trimmed_coach_dictionary.json')
        schoolMapJSON = load_json('files/mapping_schools.json')
        pollsJSON = load_json('files/polls.json')
        recordsJSON = load_json('files/records.json')
        mascotsJSON = load_json('files/total_no_mascots_inv.json')
        rostersJSON = load_json('files/rosters.json')
        school_links = load_json('files/school_links.json')
        DII_links = load_json('files/DII_links.json')
        DIII_links = load_json('files/DIII_links.json')
        naia_links = load_json('files/naia_links.json')
        FCS_links = load_json('files/FCS_links.json')

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
            
        def record_map(record, year, coach):
            '''this function returns a list of statistical features for a given coach in a given year. It includes scoring offense,
            scoring defense, win percentage, talent level, and strength of schedule.'''
            if record != record:
                return []
            else:
                print("record\n", record)
                gameCount = len(record)
                scoringOffense = []
                scoringDefense = []
                winCount = 0.
                for game in record:
                    print("game\n", game)
                    score = str(game[1]).split('-')
                    print("score\n", score)
                    offense, defense = int(score[0]), int(score[1])
                    scoringOffense.append(offense)
                    scoringDefense.append(defense)
                    if offense > defense:
                        winCount += 1
                    elif offense == defense:
                        winCount += .5
                scoringOffense = sum(nparr(scoringOffense)) / gameCount
                scoringDefense = sum(nparr(scoringDefense)) / gameCount
                winRate = winCount / gameCount
                school = school_coach_year.at(year, coach)
                strengthOfSchedule = BCS_sos(school, year)
                teamTalent = total_talent(school, year)
                # coachingSuccess = success_level(year, school)
                return [scoringOffense, scoringDefense, winRate, strengthOfSchedule, teamTalent]

        def annual_record_map(season):
            season = Series(season)
            year = int(season.name)
            recordFeaturesDict = {}
            for coach in season.index:
                recordFeaturesDict[coach] = record_map(season[coach], year, coach)
            return Series(recordFeaturesDict)

        # === METRICS COMPILATION ===
        school_coach_year = tabulate(coachJSON, columnDepth=3, indexDepth=0, valueDepth=1)
        add_metric(school_coach_year, str, True, True, False, school_map, "school_coach_year")
        
        role_coach_year = tabulate(coachJSON, columnDepth=3, indexDepth=0, valueDepth=2)
        add_metric(role_coach_year, [str, int], [True, True], [False, False], [False, False], role_map, "role_coach_year")
        
        rank_school_year = tabulate(pollsJSON, columnDepth=(2, None), indexDepth=0, valueDepth=1)
        rank_school_year = rank_school_year.apply(to_numeric, errors='coerce')
        map_columns(rank_school_year, school_map)
        rank_coach_year = recolumnate(rank_school_year, school_coach_year)
        add_metric(rank_coach_year, int, True, False, True, rank_map, "rank_coach_year")

        record_school_year = tabulate(recordsJSON, columnDepth=0, indexDepth=1, valueDepth=(2, None))
        record_school_year = map_columns(record_school_year, school_map)
        record_coach_year = recolumnate(record_school_year, school_coach_year)
        record_coach_year = record_coach_year.apply(annual_record_map, axis=1)
        add_metric(
            record_coach_year,
            [float, float, float, float, float, float],
            [True, True, True, True, True],
            [False, False, False, False, False],
            [False, False, True, False, False],
            name="record_coach_year"
        )

        # === PACKAGING METRICS ===
        # --- Listing feature and label types ---
        XTypes, YTypes = [], []
        for type, background, foresight, prediction in zip(metricTypes, backgroundMask, foresightMask, predictionMask):
            if background:
                XTypes.append(type)
            if foresight:
                XTypes.append(type)
            if prediction:
                YTypes.append(type)

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
                XElement, YElement = [], []
                for metric, type, background, foresight, prediction in zip(metrics, metricTypes, backgroundMask, foresightMask, predictionMask):
                    backgroundMetric = list(Series(DataFrame(metric).loc[backgroundYears, coach]).values)
                    predictionMetric = list(Series(DataFrame(metric).loc[predictionYears, coach]).values)

                    if background:
                        if isinstance(backgroundMetric[0], list):
                            for subMetric in [list(subMetric) for subMetric in zip(*backgroundMetric)]:
                                XElement.append(subMetric)
                        else:
                            XElement.append(backgroundMetric)
                    
                    if foresight:
                        if isinstance(predictionMetric[0], list):
                            for subMetric, subType in zip([list(subMetric) for subMetric in zip(*predictionMetric)], type):
                                foresightPadding = [subType()] * (self.backgroundYears - self.predictionYears)
                                XElement.append(foresightPadding + subMetric)
                        else:
                            foresightPadding = [type()] * (self.backgroundYears - self.predictionYears)
                            XElement.append(foresightPadding + predictionMetric)

                    if prediction:
                        if isinstance(predictionMetric[0], list):
                            for subMetric in [list(subMetric) for subMetric in zip(*predictionMetric)]:
                                YElement.append(subMetric)
                        else:
                            YElement.append(predictionMetric)

                X.append(nparr(XElement).T)
                Y.append(nparr(YElement).T)

        X = nparr(X)
        Y = nparr(Y)

        trainX, validX, trainY, validY = train_test_split(X, Y, test_size=0.2)
        save_pkl(trainX, "files/trainX.pkl")
        save_pkl(validX, "files/validX.pkl")
        save_pkl(trainY, "files/trainY.pkl")
        save_pkl(validY, "files/validY.pkl")
        save_pkl(XTypes, "files/XTypes.pkl")
        save_pkl(YTypes, "files/YTypes.pkl")
        return