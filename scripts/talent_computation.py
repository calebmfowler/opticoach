'''this file computes the talent level of a college team based on draft numbers and seniority'''

import json as json

with open('rosters.json', 'r', encoding='utf-8') as file2read:
    roster_dict = json.load(file2read)
    
with open('mapping_schools.json', 'r', encoding='utf-8') as file2read:
    maps = json.load(file2read)

    
    
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


print(talent_composite(2018, 'Alabama'))



for i in range(2016, 2021):
    print(i, 'TAMU talent composite score is', talent_composite(i, 'Texas A&M'))
    
