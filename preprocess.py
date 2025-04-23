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
        mascotsJSON = load_json('files/total_no_mascots_inv.json')

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
        
      


        def team_avg(team, year):
            year = str(year)
            season = recordsJSON[mascotsJSON[team]][year]
            wins = 0
            for game in season:
                score = game[1]
                home_score = int(score.split('-')[0])
                away_score = int(score.split('-')[1])
                if home_score > away_score:
                    wins += 1
                elif home_score == away_score:
                    wins += .5
            return wins/len(season)


        def sos(team, year):
            year = str(year)
            season = recordsJSON[mascotsJSON[team]][year]
            team_list = []
            avg = 0
            for game in season:
                avg += team_avg(game[0], int(year))
            avg = avg/len(season)
            return avg
                
                
        def BCS_sos(team, year):
            year = str(year)
            season = recordsJSON[mascotsJSON[team]][year]
            team_sos = sos(team, int(year))
            opponent_sos = 0
            for game in season:
                opponent_sos += sos(game[0], int(year))
            opponent_sos = opponent_sos/len(season)
            total_sos = (2 * team_sos + opponent_sos)/3
            return total_sos

        def sos_regular(team, year):
            year = str(year)
            season = recordsJSON[mascotsJSON[team]][year]
            team_list = []
            avg = 0
            game_count = 0
            for game in season:
                if game[2] != 'BOWL' and game[2] != 'FBS':
                    avg += team_avg(game[0], int(year))
            avg = avg/len(season)
            return avg

        def BCS_sos_regular(team, year):
            year = str(year)
            season = recordsJSON[mascotsJSON[team]][year]
            team_sos = sos_regular(team, int(year))
            opponent_sos = 0
            for game in season:
                opponent_sos += sos(game[0], int(year))
            opponent_sos = opponent_sos/len(season)
            total_sos = (2 * team_sos + opponent_sos)/3
            return total_sos
            

        # def sos_top25(team, year):
        #     return .5 * BCS_sos(team, year) + .5 * top25_score(team, year)


        def record_map(record, year):
            gameCount = len(record)
            scoringOffense = []
            scoringDefense = []
            winCount = 0.
            for game in record:
                score = game[1].split('-')
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
            return [scoringOffense, scoringDefense, winRate]

        def annual_record_map(season):
            year = season.name
            recordFeaturesDict = {}
            for coach in season.columns:
                recordFeaturesDict[coach] = record_map(season[coach], year)
            return Series(recordFeaturesDict)

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
        recordFeatures_coach_year = record_coach_year.apply(annual_record_map, axis=1)
        add_metric(recordFeatures_coach_year, name="recordFeatures_coach_year")

        # === PACKAGING METRICS ===
        X, Y = [], []
        for coach, i in zip(school_coach_year.columns, range(len(school_coach_year.columns))):
            print(f"coach {i} = {coach}")
            schools = school_coach_year[coach]
            schoolChanges = schools != schools.shift()
            changeYears = schools.index[schoolChanges][1:]

            for changeYear in changeYears:
                newSchool = school_coach_year.at(changeYear, coach)
                if (newSchool != newSchool or
                    changeYear - self.backgroundYears < self.startYear or
                    changeYear + self.predictionYears > self.endYear):
                    continue

                backgroundYears = range(changeYear - self.backgroundYears, changeYear)
                predictionYears = range(changeYear, changeYear + self.predictionYears)
                backgroundSchools = Series(school_coach_year.loc[backgroundYears, coach])
                predictionSchools = Series(school_coach_year.loc[predictionYears, coach])
                padding = self.backgroundYears - self.predictionYears
                predictionSchoolsPadding = Series([""] * padding, index=[None] * padding)

                featureSet = [
                    backgroundSchools.values,
                    predictionSchoolsPadding.append(predictionSchools)                    
                ]
                for metric in metrics:
                    metric = DataFrame(metric)
                    feature = list(Series(metric.loc[backgroundYears, coach]).values)
                    if isinstance(feature[0], list):
                        for subfeature in [list(subfeature) for subfeature in zip(*feature)]:
                            featureSet.append(subfeature)
                    else:
                        featureSet.append(feature)

                labelSet = []
                for metric in metrics:
                    metric = DataFrame(metric)
                    feature = list(Series(metric.loc[predictionYears, coach]).values)
                    if isinstance(feature[0], list):
                        for subfeature in [list(subfeature) for subfeature in zip(*feature)]:
                            featureSet.append(subfeature)
                    else:
                        featureSet.append(feature)

                X.append(nparr(featureSet).T)
                Y.append(nparr(labelSet).T)

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