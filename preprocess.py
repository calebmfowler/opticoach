from aggregate import Aggregator
from copy import deepcopy
from numpy import array as nparr
import numpy as np
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

        def position_weight(string):
            if string == 'QB':
                weight = 3
            elif string in ['RB', 'FB', 'HB']:
                weight = 2.5
            elif string in ['TE', 'WR', 'DE', 'EDGE', 'CB', 'S']:
                weight = 2
            elif string in ['LB', 'DB', 'ILB', 'OLB']:
                weight = 1.5
            elif string in ['OG', 'OL', 'C', 'OT', 'DL', 'DT', 'NT']:
                weight = 1
            else:
                weight = 1
            return weight


        def trim_string(s):
            new_str = ''
            for i in s:
                if i.isnumeric() == True:
                    new_str+=i
                else:
                    break
            return new_str


        def smooth_blend(x, x0, width):
            """Smooth transition function centered at x0 with given width."""
            return 1 / (1 + np.exp(-(x - x0) / width))

        def hybrid_function_smooth_slope(x,
                                        exp_decay_rate=0.4,
                                        exp_drop=0.05,
                                        sigmoid_center=128,
                                        sigmoid_steepness=20,
                                        plateau_start=0.95,
                                        plateau_end=0.93,
                                        final_value=0.1,
                                        exp_end=10,
                                        sigmoid_start=64,
                                        sigmoid_end=192,
                                        blend_width=5):
            # Region 1: exponential decay from 1 to plateau_start
            exp_part = 1 - exp_drop * (1 - np.exp(-exp_decay_rate * x))

            # Region 2: light decreasing function from plateau_start to plateau_end
            slope = (plateau_end - plateau_start) / (sigmoid_start - exp_end)
            plateau_part = plateau_start + slope * (x - exp_end)

            # Region 3: sigmoid decline from plateau_end to final_value
            sigmoid_part = (plateau_end - final_value) / (1 + np.exp((x - sigmoid_center) / sigmoid_steepness)) + final_value

            # Region 4: final flat value after sigmoid
            final_part = np.full_like(x, final_value)

            # Smooth transitions
            blend_1 = smooth_blend(x, exp_end, blend_width)         # exp -> slope
            blend_2 = smooth_blend(x, sigmoid_start, blend_width)   # slope -> sigmoid
            blend_3 = smooth_blend(x, sigmoid_end, blend_width)     # sigmoid -> flat

            # Combine smoothly
            y = ((1 - blend_1) * exp_part +
                (blend_1 * (1 - blend_2)) * plateau_part +
                (blend_2 * (1 - blend_3)) * sigmoid_part +
                blend_3 * final_part)

            return y


        def calc_senior(number):
            if number == 1:
                weight = 1
            elif number == 2:
                weight = .75
            elif number >= 3:
                weight = .5
            return weight
            
  
            
        def talent_composite(year, team):
            year = str(year)
            college_list = list(school_links.keys()) + list(DII_links.keys()) + list(DIII_links.keys()) + list(naia_links.keys()) + list(FCS_links.keys())
            try:
                team = maps[team]
            except:
                team = team
            try:
                no_mascot = no_mascots[team]
            except:
                no_mascot = team
            if no_mascot in college_list:
                roster = roster_dict[year][team]
                total = 0
                for player in roster:
                    position = player[1]
                    pick = int(trim_string(player[2]))
                    seniority = calc_senior(int(player[-1])-int(year))
                    value = position_weight(position) * hybrid_function_smooth_slope(pick-1) * seniority
                    total+=value
            else:
                total = 0
            return total


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
            school = school_coach_year.at(year, coach)
            try:
                school = schoolMapJSON[school]
            except:
                school = school
            BCS_strength = BCS_sos(school, year)
            talent_level = talent_composite(year, school)
            return [scoringOffense, scoringDefense, winRate, BCS_strength, talent_level]

        def annual_record_map(season):
            year = season.name
            recordFeaturesDict = {}
            for coach in season.columns:
                recordFeaturesDict[coach] = record_map(season[coach], year, coach)
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