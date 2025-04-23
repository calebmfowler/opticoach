'''this file computes various SOS metrics'''

import json as json

with open('records.json', 'r', encoding = 'utf-8') as file2read:
    records = json.load(file2read)
    
    
with open('mapping_schools.json', 'r', encoding = 'utf-8') as file2read:
    maps = json.load(file2read)
    
    
with open('polls.json', 'r', encoding = 'utf-8') as file2read:
    polls = json.load(file2read)
    

new_polls = {}

for year in list(polls.keys()):
    if int(year) >= 1975:
        new_polls[year] = {}
        for week in list(polls[year].keys()):
            new_polls[year][week] = {}
            values = list(polls[year][week].values())
            keys = list(polls[year][week].keys())
            for i in range(len(keys)):
                key = keys[i]
                value = values[i]
                print(key, value)
                if type(value) == list:
                    value_list = []
                    for i in value:
                        value_list.append(maps[i])
                    new_polls[year][week][key] = value_list
                else:
                    new_polls[year][week][key] = maps[value]
                
with open('standard_polls.json', 'w', encoding = 'utf-8') as file2write:
    json.dump(new_polls, file2write)
    
print(polls['2001']['Final'])
print(new_polls['2001']['Final'])


with open('total_no_mascots_inv.json', 'r') as file2read:
    no_mascots = json.load(file2read)
    
polls_inverted = {}

for year, weeks in new_polls.items():
    polls_inverted[year] = {}
    for week, places in weeks.items():
        polls_inverted[year][week] = {}
        for place, team_or_teams in places.items():
            if isinstance(team_or_teams, list):
                for team in team_or_teams:
                    polls_inverted[year][week][team] = place
            else:
                polls_inverted[year][week][team_or_teams] = place
                


def top25_score(team, year):
    year = str(year)
    season = records[no_mascots[team]][year]
    sum_ = 0
    for game in season:
        opponent = game[0]
        # print(opponent)
        try:
            rank = int(polls_inverted[year]['Final'][opponent])
            # print(rank)
            dummy = False
        except:
            dummy = True
        if dummy == False:
            sum_ += (26-rank)
        # print(sum_)
    return sum_/(26*12)

def team_avg(team, year):
    year = str(year)
    season = records[no_mascots[team]][year]
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
    season = records[no_mascots[team]][year]
    team_list = []
    avg = 0
    for game in season:
        avg += team_avg(game[0], int(year))
    avg = avg/len(season)
    return avg
        
        
def BCS_sos(team, year):
    year = str(year)
    season = records[no_mascots[team]][year]
    team_sos = sos(team, int(year))
    opponent_sos = 0
    for game in season:
        opponent_sos += sos(game[0], int(year))
    opponent_sos = opponent_sos/len(season)
    total_sos = (2 * team_sos + opponent_sos)/3
    return total_sos

def sos_regular(team, year):
    year = str(year)
    season = records[no_mascots[team]][year]
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
    season = records[no_mascots[team]][year]
    team_sos = sos_regular(team, int(year))
    opponent_sos = 0
    for game in season:
        opponent_sos += sos(game[0], int(year))
    opponent_sos = opponent_sos/len(season)
    total_sos = (2 * team_sos + opponent_sos)/3
    return total_sos
    

def sos_top25(team, year):
    return .5 * BCS_sos(team, year) + .5 * top25_score(team, year)



    
    

print(BCS_sos('Texas A&M', 2019))
print(top25_score('Texas A&M', 2019))
print(sos_top25('Texas A&M', 2019))

print(BCS_sos('Texas', 2019))
print(top25_score('Texas', 2019))
print(sos_top25('Texas', 2019))

print(BCS_sos('Louisiana State', 2019))
print(top25_score('Louisiana State', 2019))
print(sos_top25('Louisiana State', 2019))














        