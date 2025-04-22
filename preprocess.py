from aggregate import Aggregator
from copy import deepcopy
from numpy import array as nparr, nan, isnan
from pandas import Series, to_numeric, DataFrame, json_normalize, read_json
from sklearn.model_selection import train_test_split
from utilities import load_json, save_pkl, serialize_dictionary, tabulate_dictionary, recolumnate

class Preprocessor:
    '''
    ### class Preprocessor
    This `class` preprocesses the aggregated data into a format suitable for model training.

    ### dict __startYear
    This private `int` variable stores the first year to be included.

    ### dict __endYear
    This private `int` variable stores the first year to be disincluded.

    ### dict __aggregatedFiles
    This private `dict` variable stores `string` keys of all aggregated files and `string` values of 
    file names as sourced from the Aggregator instance.

    ### dict preprocessedFiles
    This `dict` variable stores `string` keys of all preprocessed files and `string` values of file names.

    ### void __init__(self, Aggregator aggregator)
    This `void` function is called as the Preprocessor constructor, initializing the variable __aggregatedFiles
    from an Aggregator instance. If provided a Preprocessor, it operates as a copy constructor.

    ### DataFrame __bound(DataFrame df)


    ### void preprocess()
    This `void` function preprocesses all data, updating the files referenced by preprocessedFiles
    '''
    
    def __init__(self, arg, startYear=1936, endYear=2024, backgroundYears=10, predictionYears=1):
        if type(arg) == Aggregator:
            self.__aggregatedFiles = Aggregator(arg).aggregatedFiles
            self.__startYear = startYear
            self.__endYear = endYear
            self.__backgroundYears = backgroundYears
            self.__predictionYears = predictionYears
            self.preprocessedFiles = {}
        elif type(arg) == Preprocessor:
            self.__aggregatedFiles = deepcopy(arg.__aggregatedFiles)
            self.__startYear = deepcopy(arg.__startYear)
            self.__endYear = deepcopy(arg.__endYear)
            self.__backgroundYears = deepcopy(arg.__backgroundYears)
            self.__predictionYears = deepcopy(arg.__predictionYears)
            self.preprocessedFiles = deepcopy(arg.preprocessedFiles)
        else:
            raise Exception("Incorrect arguments for Preprocessor.__init__(self, aggregator)")
        return
    
    def bound(self, df):
        '''
        Restrict a DataFrame to a specified range of integer index values.
        Also, sort by string casted columns and integer casted indices.

        Parameters:
        -----------
        df : pandas.DataFrame
            The DataFrame whose index will be filtered and sorted.
        
        start : int
            The starting index value (inclusive) of the desired range.
        
        end : int
            The ending index value (inclusive) of the desired range.

        Returns:
        --------
        pandas.DataFrame
            A new DataFrame containing only the rows within the specified index range, sorted by index.

        Notes:
        ------
        - Converts the index to integers before filtering.
        - Sorts the DataFrame by its index prior to slicing.
        '''

        df.index = df.index.astype(int)
        df.columns = df.columns.astype(str)
        df = df.sort_index().loc[self.__startYear:self.__endYear]
        df = df[sorted(df.columns)]
        return df

    def preprocess(self):
        # === COMPILING METRICS ===
        # --- "school" by "coach" by int(year) ---
        coachJSON = load_json('files/trimmed_coach_dictionary.json')
        school_coach_year = self.bound(tabulate_dictionary(coachJSON, columnDepth=3, indexDepth=0, valueDepth=1))
        schoolMapJSON = load_json('files/mapping_schools.json')
        def schoolMap(school):
            if school in schoolMapJSON:
                return schoolMapJSON[school]
            elif isnan(school):
                return ""
            else:
                return str(school)
        school_coach_year = school_coach_year.map(schoolMap)
        print('\nschool_coach_year\n', school_coach_year)
        
        # --- "role" by "coach" by int(year) ---
        role_coach_year = self.bound(tabulate_dictionary(coachJSON, columnDepth=3, indexDepth=0, valueDepth=2))
        def roleMap(role):
            if role != role:
                return ""
            i = role.find('/')
            if i != -1:
                return str(role[:i])
            else:
                return str(role)
        role_coach_year = role_coach_year.map(roleMap)
        metrics = [role_coach_year]
        print('\nrole_coach_year\n', role_coach_year)

        # --- int(heisman) by "coach" by int(year) ---
        heismanJSON = load_json('files/heismans.json')
        heismanSchool_year = self.bound(serialize_dictionary(heismanJSON, indexDepth=0, valueDepth=2))
        heismanSchool_year = heismanSchool_year.map(schoolMap)
        heismanSchool_year = heismanSchool_year.reindex(school_coach_year.index)
        heisman_coach_year = self.bound(school_coach_year.eq(heismanSchool_year, axis=0)).astype(int)
        metrics.append(heisman_coach_year)
        print('\nheisman_coach_year\n', heisman_coach_year)
        
        # --- int(rank) by "coach" by int(year) ---
        finalPollsJSON = load_json('files/final_polls.json')
        rank_school_year = self.bound(tabulate_dictionary(finalPollsJSON, columnDepth=(2, None), indexDepth=0, valueDepth=1))
        rank_school_year = rank_school_year.apply(to_numeric, errors='coerce').astype('Int64')
        rank_coach_year = self.bound(recolumnate(rank_school_year, school_coach_year))
        def rankMap(num):
            if isnan(num):
                return 30
            else:
                return int(num)
        rank_coach_year = rank_coach_year.map(rankMap)
        metrics.append(rank_coach_year)
        print('\nrank_coach_year\n', rank_coach_year)

        # --- int(offensiveScore) by coach by year ---
        recordsJSON = load_json('files/records.json')
        recordsDF = self.bound(tabulate_dictionary(recordsJSON, columnDepth=0, indexDepth=1, valueDepth=(2, None)))
        records_coach_year = self.bound(recolumnate(recordsDF, school_coach_year))
        offense = lambda record : [int(game[1].split('-')[0]) for game in record]
        defense = lambda record : [int(game[1].split('-')[1]) for game in record]
        def offensiveScoreMap(record):
            try:
                return sum(nparr(offense(record), dtype=int))
            except:
                return 0
        offensiveScore_coach_year = records_coach_year.map(offensiveScoreMap)
        metrics.append(offensiveScore_coach_year)
        print('\noffensiveScore_coach_year\n', offensiveScore_coach_year)

        # --- int(defensiveScore) by coach by year ---
        def defensiveScoreMap(record):
            try:
                return sum(nparr(defense(record), dtype=int))
            except:
                return 0
        defensiveScore_coach_year = records_coach_year.map(defensiveScoreMap)
        metrics.append(defensiveScore_coach_year)
        print('\ndefensiveScore_coach_year\n', defensiveScore_coach_year)
        
        # --- int(winCount) by coach by year ---
        def winCountMap(record):
            try:
                return sum(nparr(
                    [offense > defense for offense, defense in zip(offense(record), defense(record))]
                , dtype=int))
            except:
                return 0
        wins_coach_year = records_coach_year.map(winCountMap)
        metrics.append(wins_coach_year)
        print('\nwins_coach_year\n', wins_coach_year)
        
        # --- int(lossCount) by coach by year ---
        def lossCountMap(record):
            try:
                return sum(nparr(
                    [offense < defense for offense, defense in zip(offense(record), defense(record))]
                , dtype=int))
            except:
                return 0
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
                if (predictionSchools.nunique() != 1 or not isinstance(newSchool, str) and isnan(newSchool)):
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