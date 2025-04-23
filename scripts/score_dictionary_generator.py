'''this program creates a dictionary of games corresponding to a season for a particular team'''

# link = 'https://pro-football-results.com/ufl.htm'

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Referer": "https://google.com",  # Spoofing as if coming from Google
    "Accept-Language": "en-US,en;q=0.9",
}

from bs4 import BeautifulSoup
import requests as requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json as json



def dict_assignment(d, keys, value):
    """
    Assigns a value to a nested dictionary using a sequence of keys.
    If any key does not exist, a new dictionary is created.
    If the final key already has a value, it appends the new value to a list,
    but only if it is not already present.

    :param d: dict - The preexisting dictionary.
    :param keys: list - A list of keys defining the path.
    :param value: any - The value to assign.
    """
    current = d
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    
    last_key = keys[-1]
    if last_key in current:
        if isinstance(current[last_key], list):
            if value not in current[last_key]:  # Prevent duplicate
                current[last_key].append(value)
        else:
            if current[last_key] != value:  # Convert to list only if different
                current[last_key] = [current[last_key], value]
    else:
        current[last_key] = value
#     """
#     Assigns a value to a nested dictionary using a sequence of keys.
#     If any key does not exist, a new dictionary is created.
#     If the final key already has a value, it appends the new value to a list,
#     but only if it is not already present.

#     :param d: dict - The preexisting dictionary.
#     :param keys: list - A list of keys defining the path.
#     :param value: any - The value to assign.
#     """
#     current = d  # Start at the root of the dictionary
#     for key in keys[:-1]:  # Traverse through all keys except the last one
#         if key not in current or not isinstance(current[key], dict):  # If key doesn't exist or isn't a dictionary
#             current[key] = {}  # Create a new dictionary
#         current = current[key]  # Move to the next level
    
#     last_key = keys[-1]  # Get the last key
#     if last_key in current:  # If the last key already exists
#         if isinstance(current[last_key], list):  # If the value is a list
#             if value not in current[last_key]:  # Prevent duplicates
#                 current[last_key].append(value)  # Append the value
#         else:  # If the value is not a list
#             if current[last_key] != value:  # Convert to a list only if the value is different
#                 current[last_key] = [current[last_key], value]
#     else:  # If the last key doesn't exist
#         current[last_key] = value  # Assign the value




# print(file_lines[0:10])
with open('arenafl_links.json', 'r') as file2read:
    schools = json.load(file2read)
    

with open('records.json', 'r') as file2read:
    seasons = json.load(file2read)
    
names = list(schools.keys())
# new_names = []
# for i in names:
#     if i not in list(seasons.keys()):
#         new_names.append(i)
new_names = names

# seasons = {}
for name in new_names:
    link = schools[name]
    data = requests.get(link, headers=headers)
    soup = BeautifulSoup(data.content, 'html.parser')
    tables = pd.read_html(str(soup))
    table = tables[0].to_csv('seasons_data_temp.csv', encoding = 'utf-8')
    with open('seasons_data_temp.csv', 'r', encoding = 'utf-8') as file2read:
        file_lines = file2read.readlines()
    game_list = []
    # season_dict = {}
    for line in file_lines[1:]:
        line = line.split(',')
        opponent = line[1]
        score = line[3]
        conference = line[5].strip()
        if conference == '':
            conference = 'NONCON'
        # game_list = []
        if opponent.isnumeric() == True:
            year = opponent
        elif opponent.strip() != '':
            if score.strip() != '':
                game_list.append([opponent, score, conference])
                # dict_assignment(season_dict, [opponent.strip()], [score, conference])
            else:
                dummy = True
        else:
            # dict_assignment(seasons, [name, int(year)], season_dict)
            print(game_list)
            dict_assignment(seasons, [name, int(year)], game_list)
            # season_dict = {}
            game_list = []
        
with open('records.json', 'w') as file2write:
    json.dump(seasons, file2write)
            
# with open('records.json', 'r') as file2read:
#     seasons = json.load(file2read)
    
# with open('school_links.json', 'r') as file2read:
#     D1 = json.load(file2read)
            
# def ppgame(year, team):
#     global seasons
#     year = str(year)
#     season = seasons[team][year]
#     home_score = 0
#     away_score = 0
#     game_count = 0
#     wins = 0
#     for game in season:
#         # print(game)
#         score_list = game[1].split('-')
#         game_count += 1
#         home_score += int(score_list[0])
#         away_score += int(score_list[1])
#         if score_list[0] > score_list[1]:
#             wins += 1
#     return (home_score/game_count, away_score/game_count, wins, game_count)

# with open('total_no_mascots_inv.json', 'r') as file2read:
#     no_mascots_inv = json.load(file2read)

# def strength_of_schedule(year, team):
#     global seasons
#     global no_mascots_inv
#     season = seasons[team][str(year)]
#     avg = 0
#     for i in range(len(season)):
#         # print(team)
#         team = season[i][0]
#         # print(team)
#         team = no_mascots_inv[team]
#         avg += ppgame(year, team)[2]/ppgame(year, team)[3]
#     return avg/len(season)

# def strength_of_schedule_D1(year, team):
#     global seasons
#     global no_mascots_inv
#     global D1
#     season = seasons[team][str(year)]
#     avg = 0
#     game_count = 0
#     for i in range(len(season)):
#         # print(team
#         team = season[i][0]
#         team = no_mascots_inv[team]
#         if team in list(D1.keys()):
#             avg += ppgame(year, team)[2]/ppgame(year, team)[3]
#             game_count += 1
#     return avg/game_count

# def strength_of_schedule_NONCON(year, team):
#     global seasons
#     global no_mascots_inv
#     global D1
#     season = seasons[team][str(year)]
#     avg = 0
#     game_count = 0
#     for i in range(len(season)):
#         # print(team
#         team = season[i][0]
#         # print(team)
#         team = no_mascots_inv[team]
#         if season[i][2] == 'NONCON':
#             avg += ppgame(year, team)[2]/ppgame(year, team)[3]
#             game_count += 1
#     return avg/game_count

# def total_SOS(year, team):
#     sos_whole = strength_of_schedule(year, team)
#     global seasons
#     global no_mascots_inv
#     season = seasons[team][str(year)]
#     avg = 0
#     for i in range(len(season)):
#         team = season[i][0]
#         team = no_mascots_inv[team]
#         avg += strength_of_schedule(year, team)
#     sos_second = avg/len(season)
#     sos_total = (2 * sos_whole + sos_second)/3
#     return sos_total


# def total_SOS_D1(year, team):
#     sos_whole = strength_of_schedule_D1(year, team)
#     global seasons
#     global D1
#     global no_mascots_inv
#     season = seasons[team][str(year)]
#     avg = 0
#     game_count = 0
#     for i in range(len(season)):
#         team = season[i][0]
#         team = no_mascots_inv[team]
#         if team in list(D1.keys()):
#             avg += strength_of_schedule_D1(year, team)
#             game_count += 1
#     sos_second = avg/game_count
#     sos_total = (2 * sos_whole + sos_second)/3
#     return sos_total

# def total_SOS_NONCON(year, team):
#     sos_whole = strength_of_schedule_NONCON(year, team)
#     global seasons
#     global D1
#     global no_mascots_inv
#     season = seasons[team][str(year)]
#     avg = 0
#     game_count = 0
#     for i in range(len(season)):
#         team = season[i][0]
#         team = no_mascots_inv[team]
#         if season[i][2]=='NONCON' or season[i][2] == 'BOWL':
#             avg += strength_of_schedule_NONCON(year, team)
#             game_count += 1
#     sos_second = avg/game_count
#     sos_total = (2 * sos_whole + sos_second)/3
#     return sos_total

# with open('polls.json', 'r') as file2read:
#     polls = json.load(file2read)


# def top25(year, team):
#     global polls
#     global seasons
#     season = seasons[team][str(year)]
#     opponents = []
#     top25 = list(polls[str(year)]['Final'].values())[0:25]
#     top25_keys = list(polls[str(year)]['Final'].keys())[0:25]
#     number = 0
#     games = 0
#     for i in season:
#         if i[0] in top25 and i[2] != 'BOWL':
#             games += 1
#             number += (26-int(top25_keys[top25.index(i[0])]))
#     return number/(12 * 25)

# # print(top25(2019, 'Texas A&M Aggies'))
# # print(top25(2019, 'Texas Longhorns'))

# # # print(strength_of_schedule(2019, 'Texas Longhorns'))
# # print(total_SOS(2019, 'Texas Longhorns'))
# # print(total_SOS_D1(2019, 'Texas Longhorns'))
# print(total_SOS(2019, 'Texas A&M Aggies'))
# # print(total_SOS_D1(2019, 'Texas A&M Aggies'))



# def conference_classifier(year):
#     conferences = {}
#     teams = list(no_mascots_inv.values())
#     for team in teams:
#         try:
#             season = seasons[team][str(year)]
#             for game in season:
#                 if game[2] != 'NONCON' and game[2] != 'BOWL':
#                     dict_assignment(conferences, [game[2].upper()], team)
#         except:
#             dummy = True
#     return conferences


# conferences = conference_classifier(2024)
# print(conferences['BIG10'])

# def strength_of_conference(year, team):
#     conferences = conference_classifier(year)
#     season = seasons[team][str(year)]
#     for game in season:
#         if game[2] != 'NONCON' and game[2] != 'BOWL':
#             conference = game[2].upper()
#             break
#     teams = conferences[conference]
#     SOS_sum = 0
#     for team in teams:
#         print('team is', team)
#         SOS_sum += total_SOS_NONCON(year, team)
#     print(conference, 'strength was', SOS_sum/len(teams))
    

# teams = list(no_mascots_inv.values())
# strength_of_conference(2018, 'Texas A&M Aggies')
# strength_of_conference(2018, 'Michigan Wolverines')

    



# # home_avg = []
# # away_avg = []
# # dif_avg = []
# # win_avg = []
# # years = list(range(1975, 2025))
# # for i in range(1975, 2025):
# #     i = str(i)
# #     home_avg.append(ppgame(i, 'Buffalo Bills')[0])
# #     away_avg.append(ppgame(i, 'Buffalo Bills')[1])
# #     dif_avg.append(ppgame(i, 'Clemson Tigers')[0]-ppgame(i, 'Clemson Tigers')[1])
# #     win_avg.append(ppgame(i, 'Clemson Tigers')[2])
    
# # plt.plot(years, home_avg)
# # plt.plot(years, away_avg)
# # plt.plot(years, 40 * np.array(win_avg))
# # plt.plot(years, dif_avg)
# # plt.scatter(dif_avg, win_avg)
# # plt.show()

