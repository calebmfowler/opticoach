from aggregate import Aggregator
from copy import deepcopy
import numpy as np
from numpy import array as nparr, nan, unique
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
        pollsJSON = load_json('files/final_polls.json')
        recordsJSON = load_json('files/records.json')
        mascotsJSON = load_json('files/total_no_mascots_inv.json')

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

        def get_sos_utilities():
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
            
            '''def sos_top25(team, year):
                return .5 * BCS_sos(team, year) + .5 * top25_score(team, year)'''

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
            
            '''def talent_composite(year, team):
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
                return total'''

            return BCS_sos

        def record_map(record, year, coach):
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
                # BCS_strength = BCS_sos(school, year)
                # talent_level = talent_composite(year, school)
                # return [scoringOffense, scoringDefense, winRate, BCS_strength, talent_level]
                return [scoringOffense, scoringDefense, winRate]

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
        # BCS_sos = get_sos_utilities()
        record_coach_year = record_coach_year.apply(annual_record_map, axis=1)
        '''add_metric(
            record_coach_year,
            recordFeatureTypes,
            recordBackgroundInclusions,
            recordForesightInclusions,
            recordPredictionInclusions,
            name="record_coach_year"
        )'''

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
        for coach, i in zip(school_coach_year.columns, range(len(school_coach_year.columns))):

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