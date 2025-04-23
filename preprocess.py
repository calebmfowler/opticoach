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
        metrics = []

        # === UTILITIES ===
        bound_years = lambda data : bound_data(data, self.startYear, self.endYear)
        tabulate = lambda *args, **kwargs : DataFrame(bound_years(tabulate_dict(*args, **kwargs)))
        serialize = lambda *args, **kwargs : Series(bound_years(serialize_dict(*args, **kwargs)))
        recolumnate = lambda *args, **kwargs : DataFrame(bound_years(recolumnate_df(*args, **kwargs)))
        def add_metric(metric, name=None, map=None):
            if name:
                print(f"{name}\n{metric}")
            if map:
                metric = DataFrame(metric).map(map)
            metrics.append(metric)

        # === FILE IMPORTS ===
        coachJSON = load_json('files/trimmed_coach_dictionary.json')
        schoolMapJSON = load_json('files/mapping_schools.json')
        heismanJSON = load_json('files/heismans.json')
        pollsJSON = load_json('files/final_polls.json')
        recordsJSON = load_json('files/records.json')

        # === MAPS ===
        def school_map(school):
            if school in schoolMapJSON:
                return schoolMapJSON[school]
            elif school != school:
                return ""
            else:
                return str(school)
            
        def role_map(role):
            if role != role:
                return ""
            else:
                role = str(role)
                i = role.find('/')
                if i != -1:
                    return role[:i]
                else:
                    return role
            
        def rank_map(num):
            if num != num:
                return 30
            else:
                return int(num)
        
        def offensive_scores(record):
            return [int(game[1].split('-')[0]) for game in record]
        
        def defensive_scores(record):
            return [int(game[1].split('-')[1]) for game in record]
        
        def scoring_offense_map(record):
            try:
                return sum(nparr(offensive_scores(record), dtype=int))
            except:
                return 0

        def scoring_defense_map(record):
            try:
                return sum(nparr(defensive_scores(record), dtype=int))
            except:
                return 0

        def loss_count_map(record):
            try:
                return sum(nparr(
                    [offense < defense for offense, defense in zip(offensive_scores(record), defensive_scores(record))]
                , dtype=int))
            except:
                return 0
    
        def win_count_map(record):
            try:
                return sum(nparr(
                    [offense > defense for offense, defense in zip(offensive_scores(record), defensive_scores(record))]
                , dtype=int))
            except:
                return 0

        def record_map(record):
            gameCount = len(record)
            scoringOffense = []
            scoringDefense = []
            winCount = 0.
            lossCount = 0.
            for game in record:
                score = game[1].split('-')
                offense, defense = int(score[0]), int(score[1])
                scoringOffense.append(offense)
                scoringDefense.append(defense)
                if offense > defense:
                    winCount += 1
                elif offense < defense:
                    lossCount += 1
            scoringOffense = sum(nparr(scoringOffense)) / gameCount
            scoringDefense = sum(nparr(scoringDefense)) / gameCount
            winRate = winCount / gameCount
            lossRate = lossCount / gameCount
            return [scoringOffense, scoringDefense, winRate, lossRate]

        # === METRICS COMPILATION ===
        school_coach_year = tabulate(coachJSON, columnDepth=3, indexDepth=0, valueDepth=1)
        add_metric(school_coach_year, name="school_coach_year", map=school_map)
        
        role_coach_year = tabulate(coachJSON, columnDepth=3, indexDepth=0, valueDepth=2)
        add_metric(role_coach_year, name="role_coach_year", map=role_map)
        
        rank_school_year = tabulate(pollsJSON, columnDepth=(2, None), indexDepth=0, valueDepth=1)
        rank_school_year = rank_school_year.apply(to_numeric, errors='coerce').astype('Int64')
        rank_coach_year = recolumnate(rank_school_year, school_coach_year)
        add_metric(rank_coach_year, name="rank_coach_year", map=rank_map)

        record_school_year = tabulate(recordsJSON, columnDepth=0, indexDepth=1, valueDepth=(2, None))
        record_coach_year = recolumnate(record_school_year, school_coach_year)
        recordFeatures_coach_year = record_coach_year.map(record_map)
        add_metric(recordFeatures_coach_year, name="recordFeatures_coach_year")

        # === PACKAGING METRICS ===
        X, Y = [], []
        for coach, i in zip(school_coach_year.columns, range(len(school_coach_year.columns))):
            print(f"coach {i} = {coach}")
            schools = school_coach_year[coach]
            schoolChanges = schools != schools.shift()
            changeYears = schools.index[schoolChanges][1:]

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