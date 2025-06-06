import numpy as np
from utilities import load_json

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
nfl_links = load_json('files/nfl_links.json')
cfl_links = load_json('files/cfl_links.json')
euro_links = load_json('files/euro_links.json')
arenafl_links = load_json('files/arenafl_links.json')
ufl_links = load_json('files/ufl_links.json')
usfl_links = load_json('files/usfl_links.json')

D1 = list(school_links.keys())
pro = list(nfl_links.keys()) + list(cfl_links.keys()) + list(arenafl_links.keys()) + list(ufl_links.keys()) + list(usfl_links.keys())
other_schools = list(DII_links.keys()) + list(DIII_links.keys()) + list(naia_links.keys()) + list(FCS_links.keys())

def ranking_score(team, year):
    '''this function computes the ranking score for a given team in a given year, as computed by the final AP poll.'''
    year = str(year)
    try:
        team = schoolMapJSON[team] #convert team to standard name if possible
    except:
        team = team
    try:
        season = pollsJSON[year]['Final'] #pull the final season poll data for the year
        teams = list(season.values()) #pull the teams from the season data
        rankings = list(season.keys()) #pull the rankings from the season data
        for i in teams:
            if type(i) == list:
                for j in i:
                    if j == team:
                        return (26 - int(rankings[teams.index(i)]))/25
            else:
                if i == team:
                    return (26 - int(rankings[teams.index(i)]))/25
        return 0 #if the team is not in the dictionary, return 0 (not ranked)
    except: #if the year is not in the poll dictionary (some unknown error), return 0
        return 0

def team_avg(team, year):
    year = str(year)
    try:
        team = schoolMapJSON[team] #convert team to standard name if possible
    except:
        team = team
    try:
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
    except:
        return .5 #if the team is not in the dictionary, return .5 (average win percentage)

def sos(team, year):
    '''this function computes the strength of schedule for a given team in a given year.'''
    year = str(year) #convert year to string
    try:
        team = schoolMapJSON[team] #convert team to standard name if possible
    except:
        team = team
    try:
        season = recordsJSON[mascotsJSON[team]][year] #pull the season data for the team/year
        team_list = []
        avg = 0
        for game in season:
            avg += team_avg(game[0], int(year))
        avg = avg/len(season)
        return avg
    except:
        return .5 #if the team is not in the dictionary, return .5 (average SOS)
    
def BCS_sos(team, year):
    '''this function computes the BCS strength of schedule for a given team in a given year.
    The BCS formula is as follows: BCS SOS = (2 * normal SOS + SOS of opponents) / 3'''
    year = str(year) # convert year to string
    try:
        team = schoolMapJSON[team] #convert team to standard name if possible
    except:
        team = team
    try:
        season = recordsJSON[mascotsJSON[team]][year] #pull the season data for the team/year
        team_sos = sos(team, int(year))
        opponent_sos = 0
        for game in season:
            opponent_sos += sos(game[0], int(year))
        opponent_sos = opponent_sos/len(season)
        total_sos = (2 * team_sos + opponent_sos)/3
        return total_sos
    except:
        return .5 #if the team is not in the dictionary, return .5 (average SOS)

def sos_regular(team, year):
    '''this function computes the regular strength of schedule for a given team in a given year.'''
    year = str(year) #convert year to string
    try:
        team = schoolMapJSON[team] #convert team to standard name if possible
    except:
        team = team
    try:
        season = recordsJSON[mascotsJSON[team]][year] #pull the season data for the team/year
        team_list = []
        avg = 0
        game_count = 0
        for game in season:
            if game[2] != 'BOWL' and game[2] != 'FBS':
                avg += team_avg(game[0], int(year))
            else:
                break
        avg = avg/len(season)
        return avg
    except:
        return .5 #if the team is not in the dictionary, return .5 (average SOS)

def BCS_sos_regular(team, year):
    '''this function computes the BCS strength of schedule for the regular season of a team in a given year.
    The BCS formula is as follows: BCS SOS = (2 * normal SOS + SOS of opponents) / 3'''
    year = str(year) #convert year to string
    try:
        team = schoolMapJSON[team] #convert team to standard name if possible
    except:
        team = team
    try:
        season = recordsJSON[mascotsJSON[team]][year] #pull the season data for the team/year
        team_sos = sos_regular(team, int(year))
        opponent_sos = 0
        for game in season:
            opponent_sos += sos(game[0], int(year))
        opponent_sos = opponent_sos/len(season)
        total_sos = (2 * team_sos + opponent_sos)/3
        return total_sos
    except:
        total_sos = .5 #if the team is not in the dictionary, return .5 (average SOS)
        return total_sos

'''def sos_top25(team, year):
    return .5 * BCS_sos(team, year) + .5 * top25_score(team, year)'''

def win_percentage(team, year):
    '''this function computes the win percentage of a team in a given year.'''
    year = str(year)
    try:
        team = schoolMapJSON[team] #convert team to standard name if possible
    except:
        team = team
    try:
        season = recordsJSON[mascotsJSON[team]][year] #pull the season data for the year/team
        wins = 0
        games = len(season)
        for game in season:
            score = game[1]
            home_score = int(score.split('-')[0])
            away_score = int(score.split('-')[1])
            if home_score > away_score:
                wins += 1
            elif home_score == away_score:
                wins += .5
        return wins/games #return the win percentage for the year/school
    except: #print error if the team is not in the dictionary
        print('Error: win_percentage. Team not found in recordsJSON.')

def total_talent(team, year):
    '''this function returns a number between 0 and 1 that represents the talent level of a team. It is computed by dividing
    the team talent level by the maximum talent level in a given year.'''
    try:
        max_talent_value = max_talent(team, year) #compute associated max talent level
        talent = talent_composite(year, team) #compute team talent level
        return talent/max_talent_value #return the ratio of team talent to max talent
    except:
        return .5 #if the team is not in the dictionary, return .5 (average talent level)

def success_level(year, team):
    '''this function computes the success level of a team in a given year. It is regularized by talent level 
    and strength of schedule.'''
    sos = BCS_sos(team, year) #compute strength of schedule
    adjusted_talent = total_talent(team, year) #compute relative talent level
    rank_score = ranking_score(team, year) #compute final ranking score
    win_loss = win_percentage(team, year) #compute win percentage
    if rank_score == 0:
        success = win_loss
    else:
        success = .8 * win_loss + .2 * rank_score #compute success level
    score = success * sos/(2*adjusted_talent) #regularize success level by talent and SOS
    return score

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

def trim_string(s):
    '''this function trims a string to only include the first number in the string.'''
    new_str = ''
    for i in s:
        if i.isnumeric() == True:
            new_str+=i
        else:
            break
    return new_str

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
                                blend_width=5):
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

def calc_senior(number):
    '''
    this function calculates the seniority weight of a player based on their draft year and current year.
    '''
    if number == 1:
        return 1
    elif number == 2:
        return .75
    elif number >= 3:
        return .5

def talent_composite(year, team):
    '''this function computes the talent level of a team in a given year. It is computed by incorporating position, pick number, and seniority of the players.'''
    year = str(year)
    try:
        team = schoolMapJSON[team] #convert team to standard name if possible
    except:
        team = team
    try: #if the team is in the dictionary, compute the talent level
        roster = rostersJSON[year][team] #pull the roster for the year/team
        total = 0
        for player in roster: #sum the talent levels of the players
            position = player[1]
            pick = int(trim_string(player[2]))
            seniority = calc_senior(int(player[-1])-int(year))
            value = position_weight(position) * hybrid_function_smooth_slope(pick-1) * seniority
            total+=value
    except: #otherwise, return 0
        total = 0
    return total #return the talent level for the year/school

def max_talent(team, year):
    '''this function computes the maximum talent level in a given year. It depends on whether the team is D1, pro, or some
    other college division.'''
    try:
        team = schoolMapJSON[team] #convert team to standard name if possible
    except:
        team = team
    if team in D1: #if the team is D1, we need to find the maximum talent level in D1
        talent_composite_dict = []
        for i in D1:
            team = schoolMapJSON[i]
            talent_composite_dict.append(talent_composite(year, team))
        max_talent = max(talent_composite_dict)
    elif team in pro: #parity in the NFL means max talent = min talent = .5
        max_talent = .5
    else: #other college divisions
        talent_composite_dict = []
        for i in other_schools: #iterate through other schools
            try:
                team = schoolMapJSON[i] #convert team to standard name if possible
            except:
                team = i
            talent_composite_dict.append(talent_composite(year, team))
        max_talent = max(talent_composite_dict)
    return max_talent #return the maximum talent level for the year/school

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
