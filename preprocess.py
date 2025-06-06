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
        metrics, metricTypes, defaultValues, embedMask, backgroundMask, foresightMask, predictionMask, vocabularies = [], [], [], [], [], [], [], []

        # === UTILITIES ===

        bound_years = lambda data : bound_data(data, self.startYear, self.endYear)

        tabulate = lambda *args, **kwargs : DataFrame(bound_years(tabulate_dict(*args, **kwargs)))

        serialize = lambda *args, **kwargs : Series(bound_years(serialize_dict(*args, **kwargs)))

        recolumnate = lambda *args, **kwargs : DataFrame(bound_years(recolumnate_df(*args, **kwargs)))

        def map_columns(df, map):
            df = DataFrame(df)
            df.columns = df.columns.map(map)
            return df

        def add_metric(metric, metricType, defaultValue, metricEmbed, vocab, background, foresight, prediction, map=None, name=None):
            if map:
                metric = DataFrame(metric).map(map)

            if name:
                print(f"{name}\n{metric}")
            
            metrics.append(metric)
            metricTypes.append(metricType)
            defaultValues.append(defaultValue)
            embedMask.append(metricEmbed)
            vocabularies.append(vocab)
            backgroundMask.append(background)
            foresightMask.append(foresight)
            predictionMask.append(prediction)

            return metric
        
        calc_senior = lambda num : float(1 - (min(num, 3) - 1) / 4)

        def numerize_string(s):
            '''this function trims a string to only include the first number in the string.'''
            new_str = ''
            for c in s:
                if c.isnumeric() == True:
                    new_str += c
                else:
                    break
            return new_str

        def position_weight(string):
            '''this function computes the weight of a position based on its importance in football.'''
            if string == 'QB': #QB tier
                weight = 3
            elif string in ['RB', 'FB', 'HB']: #RB tier
                weight = 2.5
            elif string in ['TE', 'WR', 'DE', 'EDGE', 'CB', 'S', 'DT']: #WR/TE tier, DE/CB tier
                weight = 2
            elif string in ['LB', 'DB', 'ILB', 'OLB']: #LB tier
                weight = 1.5
            elif string in ['OG', 'OL', 'C', 'OT', 'DL', 'DT', 'NT']: #OL/DL tier
                weight = 1
            else: #other positions
                weight = 1
            return weight

        def smooth_blend(x, x0, width):
            """Smooth transition function centered at x0 with given width. Precursor to hybrid function below."""
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
                                        blend_width=5
        ):
            '''mildly arbitrary function for computing talent level of a player based on their draft pick number. 
            It is a hybrid of an exponential decay, a light decreasing function, and a sigmoid decline.'''
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

        # === FILE IMPORTS ===

        coachJSON = load_json('files/coach_history_regularized.json')
        schoolMapJSON = load_json('files/mapping_schools.json')
        pollsJSON = load_json('files/final_polls.json')
        recordsJSON = load_json('files/records.json')
        rostersJSON = load_json('files/rosters.json')
        d1_links = load_json('files/school_links.json')
        d2_links = load_json('files/DII_links.json')
        d3_links = load_json('files/DIII_links.json')
        naia_links = load_json('files/naia_links.json')
        fcs_links = load_json('files/FCS_links.json')
        nfl_links = load_json('files/nfl_links.json')
        cfl_links = load_json('files/cfl_links.json')
        arenafl_links = load_json('files/arenafl_links.json')
        ufl_links = load_json('files/ufl_links.json')
        usfl_links = load_json('files/usfl_links.json')

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
                elif roleTitle in ['OC', 'DC', 'ST', 'PGC', 'RGC', 'co-OC', 'co-DC']:
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

        def performance_map(record):
            '''
            this function returns a list of statistical features for a given coach in a given year. It includes scoring offense,
            scoring defense, win percentage, talent level, and strength of schedule.
            '''
            if record != record or record == [] or not isinstance(record, list):
                return [20, 30, 0.4]
            
            if not isinstance(record[0], list):
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

        def offense_map(performance):
            return performance[0]
        
        def defense_map(performance):
            return performance[1]

        def win_rate_map(performance):
            return performance[2]

        def avg_opponent_win_rate_map(record, school, year):
            if record != record or record == [] or not isinstance(record, list):
                return 0.4

            if not isinstance(record[0], list):
                game = record
                opponentSchool = str(game[0])
                if not opponentSchool in winRate_school_year.columns:
                    return 0.4
                avgOpponentWinRate = winRate_school_year.at[year, opponentSchool]
                return avgOpponentWinRate

            else:
                gameCount = len(record)
                avgOpponentWinRate = 0
                for game in record:
                    opponentSchool = str(game[0])
                    if not opponentSchool in winRate_school_year.columns:
                        avgOpponentWinRate += 0.4
                        continue
                    avgOpponentWinRate += winRate_school_year.at[year, opponentSchool]
                avgOpponentWinRate /= gameCount
                return avgOpponentWinRate
        
        def annual_avg_opponent_win_rate_map(season):
            season = Series(season)
            year = int(season.name)
            recordFeaturesDict = {}
            for school in season.index:
                recordFeaturesDict[school] = avg_opponent_win_rate_map(season[school], school, year)
            return Series(recordFeaturesDict)

        # def head_coach_map(coach, year):
        #     try:
        #         head_coach = coachJSON[year][school]['HC']
        #         if type(head_coach) == list:
        #             head_coach = head_coach[0]
        #     except:
        #         head_coach = 'NaN'
        #     return head_coach
        
        # def annual_head_coach_map(coach):
        #     season = Series(season)
        #     year = int(season.name)
        #     headcoachDict = {}
        #     for coach in season.index:
        #         headcoachDict[coach] = head_coach_map(season[coach], year)
        #     return Series(headcoachDict)

        def sos_map(record, year, coach):
            if record != record or record == [] or not isinstance(record, list):
                return 0.4
            
            school = school_coach_year.at[year, coach]

            teamSos = avgOpponentWinRate_school_year.at[year, school]

            avgOpponentSos = 0
            if not isinstance(record[0], list):
                game = record
                opponentSchool = str(game[0])
                if not opponentSchool in avgOpponentWinRate_school_year.columns:
                    avgOpponentSos = 0.4
                else:
                    avgOpponentSos = avgOpponentWinRate_school_year.at[year, opponentSchool]
            else:
                gameCount = len(record)
                for game in record:
                    opponentSchool = str(game[0])
                    if not opponentSchool in avgOpponentWinRate_school_year.columns:
                        avgOpponentSos += 0.4
                    else:
                        avgOpponentSos += avgOpponentWinRate_school_year.at[year, opponentSchool]
                avgOpponentSos /= gameCount
            return 2/3 * teamSos + 1/3 * avgOpponentSos

        def annual_sos_map(season):
            season = Series(season)
            year = int(season.name)
            sosDict = {}
            for coach in season.index:
                sosDict[coach] = sos_map(season[coach], year, coach)
            return Series(sosDict)

        def talent_map(roster, year, coach):
            if not isinstance(roster, list) or roster == []:
                return 0
            
            school = school_coach_year.at[year, coach]
            if school in proTeams:
                return 0.5
            elif school in FBSSchools:
                maxSkill = FBSMaxSkill_year.at[year]
                skill = skill_school_year.at[year, school]
                return skill / maxSkill
            elif school in FCSSchools:
                maxSkill = FCSMaxSkill_year.at[year]
                skill = skill_school_year.at[year, school]
                return skill / maxSkill
            elif school in otherSchools:
                maxSkill = otherSchoolMaxSkill_year.at[year]
                skill = skill_school_year.at[year, school]
                return skill / maxSkill
            else:
                return 0

        def annual_talent_map(annualRoster):
            annualRoster = Series(annualRoster)
            year = int(annualRoster.name)
            talentDict = {}
            for coach in annualRoster.index:
                talentDict[coach] = talent_map(annualRoster[coach], year, coach)
            return Series(talentDict)

        def skill_map(roster, year):
            if not isinstance(roster, list) or roster == []:
                return 0
            
            totalSkill = 0
            for player in roster:
                position = str(player[1])
                try:
                    pick = int(numerize_string(player[2]))
                except:
                    pick = 200
                seniority = calc_senior(int(player[-1]) - year)
                totalSkill += position_weight(position) * hybrid_function_smooth_slope(pick - 1) * seniority            
            return totalSkill

        def annual_skill_map(annualRoster):
            annualRoster = Series(annualRoster)
            year = int(annualRoster.name)
            talentDict = {}
            for school in annualRoster.index:
                talentDict[school] = skill_map(annualRoster[school], year)
            return Series(talentDict)

        def level_map(school):
            if school in proTeams:
                return 0
            elif school in FBSSchools:
                return 1
            elif school in FCSSchools:
                return 2
            elif school in otherSchools:
                return 3
            else:
                return -1
            
        # === METRICS COMPILATION ===

        # --- Vocabulary Generation ---

        school_coach_year = tabulate(coachJSON, columnDepth=(3, None), indexDepth=0, valueDepth=1)
        school_coach_year = tabulate(coachJSON, columnDepth=(3, None), indexDepth=0, valueDepth=1)
        school_coach_year = school_coach_year.map(school_map)

        rank_school_year = tabulate(pollsJSON, columnDepth=(2, None), indexDepth=0, valueDepth=1)
        rank_school_year = map_columns(rank_school_year, school_map)

        record_school_year = tabulate(recordsJSON, columnDepth=0, indexDepth=1, valueDepth=(2, None))
        record_school_year = map_columns(record_school_year, school_map)

        roster_school_year = tabulate(rostersJSON, columnDepth=1, indexDepth=0, valueDepth=(2, None))
        roster_school_year = roster_school_year.drop(['', 'fail'], axis=1)
        roster_school_year = map_columns(roster_school_year, school_map)

        schoolVocabulary = unique(hstack((
            unique(school_coach_year),
            rank_school_year.columns,
            record_school_year.columns,
            roster_school_year.columns
        )))
        schoolVocabulary = insert(schoolVocabulary[1:], 0, ['', '[UNK]'])
        schoolVectorization = TextVectorization(standardize=None, split=None, vocabulary=schoolVocabulary)

        role_coach_year = tabulate(coachJSON, columnDepth=(3, None), indexDepth=0, valueDepth=2)
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
        schoolInt_coach_year = add_metric( # x0, X1
            schoolInt_coach_year,
            int,
            0,
            True,
            schoolVocabulary,
            True,
            True,
            False,
            name="schoolInt_coach_year"
        )
        print(schoolInt_coach_year['Jimbo Fisher'])

        roleTitleInt_coach_year = DataFrame(
            roleTitleVectorization(roleTitle_coach_year),
            columns=roleTitle_coach_year.columns,
            index=roleTitle_coach_year.index
        )
        roleTitleInt_coach_year = add_metric( # x2
            roleTitleInt_coach_year,
            int,
            0,
            True,
            roleTitleVocabulary,
            True,
            False,
            False,
            name="roleTitleInt_coach_year"
        )
        print(roleTitleInt_coach_year['Jimbo Fisher'])

        roleRank_coach_year = add_metric( # x3
            role_coach_year,
            int,
            -1,
            True,
            [-1, 0, 1, 2],
            True,
            False,
            False,
            map=role_rank_map,
            name="roleRank_coach_year"
        )
        print(roleRank_coach_year['Jimbo Fisher'])

        '''rank_school_year = rank_school_year.apply(to_numeric, errors='coerce')
        rank_coach_year = recolumnate(rank_school_year, school_coach_year)
        rank_coach_year = add_metric( # x4
            rank_coach_year,
            int,
            30,
            False,
            [],
            True,
            False,
            False,
            rank_map,
            "rank_coach_year"
        )'''

        record_coach_year = recolumnate(record_school_year, school_coach_year)
        performance_coach_year = add_metric( # x4, x5, x6
            record_coach_year,
            [float, float, float],              # metricType
            [20., 30., 0.4],                    # defaultValue
            [False, False, False],              # metricEmbed
            [[], [], []],                       # vocabularies
            [True, True, True],                 # backgroundMask
            [False, False, False],              # foresightMask
            [False, False, True],               # predictionMask
            map=performance_map,
            name="performance_coach_year"
        )
        print(performance_coach_year['Jimbo Fisher'])

        performance_school_year = record_school_year.map(performance_map)
        winRate_school_year = performance_school_year.map(win_rate_map)
        avgOpponentWinRate_school_year = record_school_year.apply(annual_avg_opponent_win_rate_map, axis=1)
        sos_coach_year = record_coach_year.apply(annual_sos_map, axis=1)
        sos_coach_year = add_metric( # x7
            sos_coach_year,
            float,
            0.4,
            False,
            [],
            True,
            False,
            False,
            name="sos_coach_year"
        )
        print(sos_coach_year['Jimbo Fisher'])

        proTeams = Series(
            list(nfl_links.keys()) + list(cfl_links.keys()) + list(arenafl_links.keys()) + list(ufl_links.keys()) + list(usfl_links.keys())
        ).map(school_map).values
        FBSSchools = Series(list(d1_links.keys())).map(school_map).values
        FCSSchools = Series(list(fcs_links.keys())).map(school_map).values
        otherSchools = Series(
            list(d2_links.keys()) + list(d3_links.keys()) + list(naia_links.keys())
        ).map(school_map).values

        skill_school_year = roster_school_year.apply(annual_skill_map, axis=1)
        skilledFBSSchools = [school for school in FBSSchools if school in skill_school_year.columns]
        FBSMaxSkill_year = skill_school_year[skilledFBSSchools].max(axis=1)
        skilledFCSSchools = [school for school in FCSSchools if school in skill_school_year.columns]
        FCSMaxSkill_year = skill_school_year[skilledFCSSchools].max(axis=1)
        skilledOtherSchools = [school for school in otherSchools if school in skill_school_year.columns]
        otherSchoolMaxSkill_year = skill_school_year[skilledOtherSchools].max(axis=1)
        roster_coach_year = recolumnate(roster_school_year, school_coach_year)
        talent_coach_year = roster_coach_year.apply(annual_talent_map, axis=1)
        talent_coach_year = add_metric( # x8
            talent_coach_year,
            float,
            0.,
            False,
            [],
            True,
            False,
            False,
            name='talent_coach_year'
        )
        print(talent_coach_year['Jimbo Fisher'])

        level_coach_year = add_metric( # x9, x10
            school_coach_year,
            int,
            -1,
            True,
            [-1, 0, 1, 2, 3],
            True,
            True,
            False,
            map=level_map,
            name='level_coach_year'
        )
        print(level_coach_year['Jimbo Fisher'])
        
        winRate_coach_year = performance_coach_year.map(win_rate_map)
        offense_coach_year = performance_coach_year.map(offense_map)
        defense_coach_year = performance_coach_year.map(defense_map)
        difference_coach_year = offense_coach_year - defense_coach_year
        maxDifference = difference_coach_year.max().max()
        print(f"{maxDifference} ({type(maxDifference)})")
        normalizedDifference_coach_year = difference_coach_year.map(lambda diff : diff / maxDifference)
        success_coach_year = ((0.5 * winRate_coach_year + 0.5 * normalizedDifference_coach_year) * sos_coach_year / (1 + 0.125 * talent_coach_year)).map(lambda x : x + 1)
        success_coach_year = add_metric(
            success_coach_year,
            float,
            0.16,
            False,
            [],
            True,
            False,
            True,
            name='success_coach_year'
        )
        print(success_coach_year['Jimbo Fisher'])

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
        relevantSchools = Series(
            list(d1_links.keys()) + 
            list(fcs_links.keys()) + 
            list(d2_links.keys())
        ).map(school_map).values
        testCoaches = ['Jimbo Fisher']

        for i, coach in enumerate(school_coach_year.columns):

            schools = school_coach_year[coach]

            for year in schools.index:

                if (year - self.backgroundYears < self.startYear or year + self.predictionYears > self.endYear):
                    continue
                
                backgroundYearList = range(year - self.backgroundYears, year)
                predictionYearList = range(year, year + self.predictionYears)

                predictionRoleTitles = Series(roleTitle_coach_year.loc[predictionYearList, coach]).values
                predictionSchools = Series(school_coach_year.loc[predictionYearList, coach]).values
                if (not all([role == 'HC' for role in predictionRoleTitles]) or
                    not all([school == predictionSchools[0] for school in predictionSchools]) or
                    not predictionSchools[0] in relevantSchools):
                    continue
                
                print(f"coach {i}, {year}, ({coach})")
                XSample, YSample = [], []
                for j, metric in enumerate(metrics):
                    backgroundMetric = list(Series(DataFrame(metric).loc[backgroundYearList, coach]).values)
                    predictionMetric = list(Series(DataFrame(metric).loc[predictionYearList, coach]).values)

                    if isinstance(backgroundMetric[0], list):
                        for subBackgroundMetric, subPredictionMetric, subMetricType, subDefaultValue, subBackground, subForesight, subPrediction in zip(
                            [list(subMetric) for subMetric in zip(*backgroundMetric)],
                            [list(subMetric) for subMetric in zip(*predictionMetric)],
                            metricTypes[j], defaultValues[j], backgroundMask[j], foresightMask[j], predictionMask[j]
                        ):
                            if subBackground:
                                XSample.append(subBackgroundMetric)
                            if subForesight:
                                foresightPadding = [subDefaultValue] * (self.backgroundYears - self.predictionYears)
                                XSample.append(foresightPadding + subPredictionMetric)
                            if subPrediction:
                                YSample.append(subPredictionMetric)
                    else:
                        if backgroundMask[j]:
                            XSample.append(backgroundMetric)
                        if foresightMask[j]:
                            foresightPadding = [defaultValues[j]] * (self.backgroundYears - self.predictionYears)
                            XSample.append(foresightPadding + predictionMetric)
                        if predictionMask[j]:
                            YSample.append(predictionMetric)

                XSample = [list(row) for row in zip(*XSample)]
                YSample = [list(row) for row in zip(*YSample)]

                if coach in testCoaches:
                    print(f"Unmasked XSample\n{nparr(XSample)}")
                    print(f"Unmasked YSample\n{nparr(YSample)}")

                maskedYearCount = 0
                for t in range(self.backgroundYears):
                    x = 0
                    default = []
                    for j, defaultValue in enumerate(defaultValues):
                        if isinstance(defaultValue, list):
                            for k, subDefaultValue in enumerate(defaultValue):
                                if backgroundMask[j][k]:
                                    default.append(XSample[t][x] == subDefaultValue)
                                    x += 1
                                if foresightMask[j][k]:
                                    default.append(XSample[t][x] == subDefaultValue)
                                    x += 1
                        else:
                            if backgroundMask[j]:
                                default.append(XSample[t][x] == defaultValue)
                                x += 1
                            if foresightMask[j]:
                                default.append(XSample[t][x] == defaultValue)
                                x += 1
                    if coach in testCoaches:
                        print(default)
                        print(any([default[i] for i in [0, 2, 3, 9]]))
                        print(all([default[i] for i in [5, 6, 7]]))
                        print(t >= self.backgroundYears - self.predictionYears and any(default[i] for i in [1, 10]))
                    if (
                        any([default[i] for i in [0, 2, 3, 9]]) # missing background school, role, or level (coaching enviroment)
                        or all([default[i] for i in [5, 6, 7]]) # missing background rank and performance (coaching performance)
                        or t >= self.backgroundYears - self.predictionYears and (
                            any(default[i] for i in [1, 10]) # missing foresight school or level (coaching enviroment)
                        )
                    ):
                        XSample[t] = [0 for y in range(len(XSample[t]))]
                        maskedYearCount += 1
                
                if coach in testCoaches:
                    print(f"Masked XSample\n{nparr(XSample)}")
                    print(f"Masked YSample\n{nparr(YSample)}")

                if maskedYearCount > self.backgroundYears / 3:
                    if coach in testCoaches:
                        print(f"Excessive Masking, Tossing Sample")
                    continue

                X.append(XSample)
                Y.append(YSample)

        X = nparr(X)
        Y = nparr(Y)

        print(f"X {shape(X)}\n{X}")
        print(f"Y {shape(Y)}\n{Y}")

        trainX, validX, trainY, validY = train_test_split(X, Y, test_size=0.2)

        save_pkl(trainX, self.preprocessedFiles['trainX'])
        save_pkl(validX, self.preprocessedFiles['validX'])
        save_pkl(trainY, self.preprocessedFiles['trainY'])
        save_pkl(validY, self.preprocessedFiles['validY'])
        return