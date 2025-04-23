'''this program generates machine learning samples'''


import json as json


with open('records.json', 'r', encoding='utf-8') as file2read:
    records = json.load(file2read)
    
with open('mapping_schools.json', 'r', encoding='utf-8') as file2read:
    maps = json.load(file2read)
    
maps['William & Mary'] = 'William & Mary'
maps['William and Mary'] = 'William & Mary'
maps['St. Louis Rams'] = 'Los Angeles Rams'
maps['San Diego Chargers'] = 'Los Angeles Chargers'
maps['Los Angeles Raiders'] = 'Las Vegas Raiders'
maps['Phoenix Cardinals'] = 'Arizona Cardinals'
maps['Tennessee Oilers'] = 'Tennessee Titans'
maps['Concordia (IL)'] = 'Concordia-Chicago'
maps['Washington Football'] = 'Washington Commanders'



with open('mapping_schools.json', 'w', encoding='utf-8') as file2read:
    json.dump(maps, file2read)
    
with open('mapping_schools.json', 'r', encoding='utf-8') as file2read:
    maps = json.load(file2read)
with open('rosters.json', 'r', encoding='utf-8') as file2read:
    rosters = json.load(file2read)
with open('coach_history_regularized.json', 'r', encoding = 'utf-8') as file2read:
    coaches = json.load(file2read)
    
  
fbs_abbreviations = {
    "Air Force": "AF",
    "Akron": "AKR",
    "Alabama": "BAMA",
    "Appalachian State": "APP",
    "Arizona": "ARIZ",
    "Arizona State": "ASU",
    "Arkansas": "ARK",
    "Arkansas State": "ARKST",
    "Army": "ARMY",
    "Auburn": "AUB",
    "Ball State": "BALLST",
    "Baylor": "BU",
    "Boise State": "BOISE",
    "Boston College": "BC",
    "Bowling Green": "BGSU",
    "Buffalo": "BUFF",
    "Brigham Young": "BYU",
    "California": "CAL",
    "Central Florida": "UCF",
    "Central Michigan": "CMU",
    "Charlotte": "CHAR",
    "Cincinnati": "CIN",
    "Clemson": "CLEM",
    "Coastal Carolina": "CCU",
    "Colorado": "COLO",
    "Colorado State": "CSU",
    "Connecticut": "UConn",
    "Duke": "DUKE",
    "East Carolina": "ECU",
    "Eastern Michigan": "EMU",
    "Florida": "UF",
    "Florida Atlantic": "FAU",
    "Florida International": "FIU",
    "Florida State": "FSU",
    "Fresno State": "FRESNO",
    "Georgia": "UGA",
    "Georgia Southern": "GASO",
    "Georgia State": "GSU",
    "Georgia Tech": "GT",
    "Hawaii": "HAW",
    "Houston": "UH",
    "Illinois": "ILL",
    "Indiana": "IU",
    "Iowa": "IOWA",
    "Iowa State": "ISU",
    "Kansas": "KU",
    "Kansas State": "KSU",
    "Kent State": "KENT",
    "Kentucky": "UK",
    "Liberty": "LIB",
    "Louisiana": "ULL",  # Some call it just "Louisiana" now
    "Louisiana Tech": "LATech",
    "Louisiana State": "LSU",
    "Louisville": "LOU",
    "Marshall": "MRSH",
    "Maryland": "UMD",
    "Massachusetts": "UMass",
    "Memphis": "MEM",
    "Miami": "Miami (FL)",
    "Miami (OH)": "Miami (OH)",
    "Michigan": "MICH",
    "Michigan State": "MSU",
    "Middle Tennessee": "MTSU",
    "Minnesota": "MINN",
    "Mississippi": "Ole Miss",
    "Mississippi State": "MSST",
    "Missouri": "MIZZOU",
    "Navy": "NAVY",
    "Nebraska": "NEB",
    "Nevada": "NEV",
    "New Mexico": "UNM",
    "New Mexico State": "NMSU",
    "North Carolina": "UNC",
    "North Carolina State": "NC State",
    "North Texas": "UNT",
    "Northern Illinois": "NIU",
    "Northwestern": "NW",
    "Notre Dame": "ND",
    "Ohio": "OHIO",
    "Ohio State": "OSU",
    "Oklahoma": "OU",
    "Oklahoma State": "OKST",
    "Old Dominion": "ODU",
    "Oregon": "ORE",
    "Oregon State": "ORST",
    "Penn State": "PSU",
    "Pittsburgh": "PITT",
    "Purdue": "PUR",
    "Rice": "RICE",
    "Rutgers": "RUT",
    "San Diego State": "SDSU",
    "San Jose State": "SJSU",
    "Southern California": "USC",
    "Southern Methodist": "SMU",
    "Southern Mississippi": "USM",
    "South Alabama": "USA",
    "South Carolina": "SCAR",
    "South Florida": "USF",
    "Stanford": "STAN",
    "Syracuse": "SYR",
    "Temple": "TEM",
    "Tennessee": "TENN",
    "Texas": "TEX",
    "Texas A&M": "TAMU",
    "Texas Christian": "TCU",
    "Texas State": "TXST",
    "Texas Tech": "TTU",
    "Toledo": "TOL",
    "Troy": "TROY",
    "Tulane": "TULN",
    "Tulsa": "TLSA",
    "Alabama-Birmingham": "UAB",
    "UCLA": "UCLA",
    "Nevada-Las Vegas": "UNLV",
    "Utah": "UTAH",
    "Utah State": "USU",
    "Texas-El Paso": "UTEP",
    "Texas-San Antonio": "UTSA",
    "Vanderbilt": "VANDY",
    "Virginia": "UVA",
    "Virginia Tech": "VT",
    "Wake Forest": "WF",
    "Washington": "UW",
    "Washington State": "WSU",
    "West Virginia": "WVU",
    "Western Kentucky": "WKU",
    "Western Michigan": "WMU",
    "Wisconsin": "WISC",
    "Wyoming": "WYO"
}
    
def return_d1(year):
    coach_list = []
    for i in list(fbs_abbreviations.keys()):
        try:
            coach_list.append(coaches[str(year)][maps[i]]['HC'])
        except:
            print(i, str(year), 'failed')
    return coach_list

def find_all_matches(nested_dict, target_value):
    matches = []

    def recurse(d, path=[]):
        if isinstance(d, dict):
            for k, v in d.items():
                recurse(v, path + [k])
        else:
            if d == target_value:
                matches.append(path)

    recurse(nested_dict)
    return matches


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


import numpy as np
import matplotlib.pyplot as plt

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
    


with open('nfl_links.json', 'r', encoding = 'utf-8') as file2read:
    nfl = json.load(file2read)
with open('ufl_links.json', 'r', encoding = 'utf-8') as file2read:
    ufl = json.load(file2read)
with open('arenafl_links.json', 'r', encoding = 'utf-8') as file2read:
    arenafl = json.load(file2read)
with open('cfl_links.json', 'r', encoding = 'utf-8') as file2read:
    cfl = json.load(file2read)

def detect_pro(school):
    if school in list(nfl.keys()) or school in list(ufl.keys()) or school in list(arenafl.keys()) or school in list(cfl.keys()):
        return True
    else:
        return False
    
def talent_composite(year, team):
    year = str(year)
    try:
        team = maps[team]
    except:
        team = team
    try:
        roster = rosters[year][team]
    except:
        roster = []
    total = 0
    for player in roster:
        position = player[1]
        pick = int(trim_string(player[2]))
        seniority = calc_senior(int(player[-1])-int(year))
        value = position_weight(position) * hybrid_function_smooth_slope(pick-1) * seniority
        total+=value
    if detect_pro(team) == False:
        return total
    else:
        return 'pro'

with open('standard_polls.json', 'r', encoding='utf-8') as file2read:
    new_polls = json.load(file2read)

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
    # print(year)
    # print(team)
    try:
        team = maps[team]
    except:
        team = team
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
    try:
        team = maps[team]
    except:
        team = team
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


def win_loss(team, year):
    try:
        team = maps[team]
        team = no_mascots[team]
    except:
        team = team
        team = no_mascots[team]
    record = records[team][str(year)]
    wins = 0
    for i in record:
        score = i[1]
        scores = score.split('-')
        score1 = int(scores[0])
        score2 = int(scores[1])
        if score1 > score2:
            wins += 1
        elif score1 == score2:
            wins += .5
    return wins/len(record)


def final_ranking(team, year):
    try:
        team = maps[team]
    except:
        team = team
    try:
        ranking = 25 - polls_inverted[str(year)]['Final'][team]
    except:
        ranking = 0
    return ranking/25



def generate_coach_sample(coach_info):
    for entry in coach_info:
        try:
            sample = {}
            team = entry[1]
            year = entry[0]
            position = entry[2]
            talent = talent_composite(int(year), team)
            strength = (BCS_sos(team, int(year)) + top25_score(team, int(year)))/2
            success = .8 * win_loss(team, int(year)) + .2 * final_ranking(team, int(year))
            sample['Success'] = success
            sample['Strength'] = strength
            sample['Talent'] = talent
            print(sample)
        except:
            team = entry[1]
            year = entry[0]
            print(team,year,  'failed')



def generate_coach_samples(year):
    coach_list = return_d1(year)
    for coach in coach_list[1:5]:
        if type(coach) != list:
            coach_info = find_all_matches(coaches, coach)
            generate_coach_sample(coach_info)
        else:
            for individual in coach:
                coach_info = find_all_matches(coaches, individual)
                generate_coach_sample(coach_info)

        
        
generate_coach_samples(2019)


