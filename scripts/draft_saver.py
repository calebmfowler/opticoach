'''this program creates dictionaries of draft data and rosters'''

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

with open('mapping_schools.json', 'r', encoding = 'utf-8') as file2read:
    maps = json.load(file2read)
maps['Concordia (Moorhead)'] = 'Concordia-Moorhead'
maps['Tennessee-Chattanooga'] = 'Chattanooga'
maps['Northeast Missouri State'] = 'Truman State'
maps['Southeast Missouri State'] = 'Southeast Missouri'
maps['Central State (OH)'] = 'Central State'
maps['Kearney State'] = 'Nebraska-Kearney'
maps['UC Davis'] = 'California-Davis'
maps['Louisiana-Lafayette'] = 'Louisiana'
maps['Northeast-Louisiana'] = 'Louisiana-Monroe'
maps['Northeast Louisiana'] = 'Louisiana-Monroe'
maps['Louisiville'] = 'Louisville'
maps['UT Martin'] = 'Tennessee-Martin'
maps['Cal Lutheran'] = 'California Lutheran'
maps['Mankato State'] = 'Minnesota State'
maps['Mesa'] = 'Colorado Mesa'
maps['Mesa State'] = 'Colorado Mesa'
maps['Penn St.'] = 'Penn State'
maps['Ul Lafayette'] = 'Louisiana'



import unicodedata as unicodedata

def is_allowed_char(c):
    return (
        c.isalpha()
        or c in {' ', '-', '–', '—', '‑', '‒', '‐', "'", '&', "."}
    )

def normalize_dashes(s):
    dash_variants = [
        '\u2010',  # Hyphen
        '\u2011',  # Non-breaking hyphen
        '\u2012',  # Figure dash
        '\u2013',  # En dash
        '\u2014',  # Em dash
        '\u2015',  # Horizontal bar
    ]
    for dash in dash_variants:
        s = s.replace(dash, '-')
    return s

import re
def trim_string(s):
    match = re.search(r"[^a-zA-Z\s.'&\-\u2013\u2014\u2012\u2011\u2010()]", s)
    if match:
        return s[:match.start()]
    return s  # return whole string if nothing matches

with open('mapping_schools.json', 'w', encoding = 'utf-8') as file2read:
    json.dump(maps, file2read)
    
def dict_assignment(d, keys, value):
    """
    Assigns a value to a nested dictionary using a sequence of keys.
    Ensures that if multiple lists are assigned to the same key, they are stored as a list of lists.

    :param d: dict - The preexisting dictionary.
    :param keys: list - A list of keys defining the path.
    :param value: any - The value to assign. May be a list or any other type.
    """
    current = d
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    
    last_key = keys[-1]

    # Handle value assignment at the final level
    if last_key in current:
        existing = current[last_key]
        # If the current value is not a list of lists, convert it
        if not isinstance(existing, list) or (existing and not isinstance(existing[0], list)):
            current[last_key] = [existing]
        # Avoid duplication if value already present
        if value not in current[last_key]:
            current[last_key].append(value)
    else:
        # Always wrap in a list if the value is a list
        current[last_key] = [value] if isinstance(value, list) else value
        
with open('mapping_schools.json', 'r', encoding = 'utf-8') as file2read:
    maps = json.load(file2read)




def dict_assignment_single(d, keys, value):
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

# for year in range(2015, 2025):
#     link = 'https://en.wikipedia.org/wiki/' + str(year) + '_NFL_draft'
#     data = requests.get(link, headers=headers)
#     soup = BeautifulSoup(data.content, 'html.parser')
    
#     name_dict = {}
#     for z in range(1, 2):
        
#         table = soup.find_all('table', {'class':'wikitable'})[z]
        
#         print(table)
        
#         rows = table.find_all('tr')
        
#         table_data = []
        
        
        
        
        
        
#         for row in rows:
#             cells = row.find_all(['td', 'th'])  # Get all header or data cells
#             for tag in cells:
#                 for hidden in tag.select('[style*="display:none"]'):
#                     hidden.decompose()  # Remove hidden elements
#             row_texts = [cell.get_text(strip=True) for cell in cells]
#             for cell in cells:
#                 link = cell.find('a')
#                 if link != None:
#                     if link.text == trim_string(row_texts[4]):
#                         link_ = link.get('href')
#                         break
#                     else:
#                         link_ = None
#                 else:
#                     link_=None
#             table_data.append([row_texts, link_])
            
#         for row in table_data[1:]:
#             link = row[1]
#             row = row[0]
#             if len(row) > 6:
#                 rnd = row[1] 
#                 pick = row[2]
#                 name = unicodedata.normalize('NFC', row[4])
#                 name = name.replace('\xa0', ' ')
#                 position = row[5]
#                 team = row[6]
#                 try:
#                     print(team)
#                     team = maps[normalize_dashes(team)]
#                     print(team)
#                 except:
#                     fail_list.append(team)
#                 print(trim_string(name), team, pick)
#                 dict_assignment(draft_dict, [str(year), team], [trim_string(name), trim_string(position), pick, rnd, link])
        

# with open('draft_dict.json', 'w', encoding = 'utf-8') as file2read:
#     json.dump(draft_dict, file2read)
    

        



from bs4 import BeautifulSoup, NavigableString, Tag

def extract_lines_from_td(td):
    lines = []

    # Handle plainlist or ul/ol format
    if td.find('ul') or td.find('ol'):
        for li in td.find_all('li'):
            buffer = ''
            for child in li.children:
                if isinstance(child, Tag) and child.name == 'a':
                    school = child.get_text(strip=True)
                    years = ''
                    if child.next_sibling and isinstance(child.next_sibling, NavigableString):
                        years = child.next_sibling.strip()
                    buffer = f"{school} {years}".strip()
                elif isinstance(child, Tag) and child.name == 'br':
                    if buffer:
                        lines.append(buffer)
                        buffer = ''
                elif isinstance(child, NavigableString):
                    text = child.strip()
                    if text and not buffer:
                        buffer = text
            if buffer:
                lines.append(buffer)

    else:
        # Handle inline a-tags and br-separated entries
        buffer = ''
        for element in td.children:
            if isinstance(element, Tag) and element.name == 'a':
                school = element.get_text(strip=True)
                years = ''
                if element.next_sibling and isinstance(element.next_sibling, NavigableString):
                    years = element.next_sibling.strip()
                buffer = f"{school} {years}".strip()
            elif isinstance(element, NavigableString):
                text = element.strip()
                if text and not buffer:
                    buffer = text
            elif isinstance(element, Tag) and element.name == 'br':
                if buffer:
                    lines.append(buffer)
                    buffer = ''
        if buffer:
            lines.append(buffer)

    return [line for line in lines if line]

    
def player_time(link, school):
    data = requests.get(link, headers=headers)
    soup = BeautifulSoup(data.content, 'html.parser')
    tables = soup.find_all('table', class_ = 'infobox vcard')
    # print(tables)
    if tables != []:
        table = tables[0]
        body = table.find('tbody')
        rows = body.find_all('tr')
        college = None
        personal = False
        for row in rows:
            if 'College:' in row.text and 'high school' not in row.text.lower() and personal == True:
                college = row
                break
            elif 'College' in row.text and 'high school' not in row.text.lower() and personal == True:
                college = row
                break
            elif 'career' in row.text.lower():
                personal = True
        if college != None:    
            td = college.find('td')
            # print(td)
            # print(td)
            colleges = extract_lines_from_td(td)
            # print(link, colleges)
            total_dates = []
            school_list = []
            # print(colleges)
            # print(link)
            for college in colleges:
                date_list = []
                number_str = ''
                school_str = ''
                start = False
                for letter in college:
                    if letter.isnumeric() == True:
                        start = True
                        number_str += letter
                    elif start == True:
                        start = False
                        date_list.append(number_str)
                        number_str = ''
                for letter in college:
                    if is_allowed_char(letter) == True:
                        school_str += letter
                    else:
                        break
                school_list.append(school_str.strip())
                total_dates.append(date_list)
            # print(total_dates, school_list)
            # return total_dates, school_list
            # print(school_list)
            try:
                check = maps[school]
            except:
                check = school
            try:
                check1 = maps[school_list[-1]]
            except:
                check1 = school_list[-1]
            if check1 != check:
                print('expected school:', check1)
                print('draft school:', check)
                print(link, 'is misassigned \n')
                return [[]], [school]
            else:
                return total_dates, school_list
        else:
            return [[]], [school]
    else:
        return [[]], [school]
        
    

roster_dict = {}
def roster_update(roster_dict, player_name, school_list, total_dates, draft_year):
    for i in range(len(school_list)):
        try:
            school = maps[school_list[i].strip()]
        except:
            school = school_list[i]
        dates = total_dates[i]
        if len(dates) == 2:
            date1 = dates[0]
            date2 = dates[1]
            for year in range(int(date1), int(date2)+1):
                dict_assignment(roster_dict, [str(year), school], player_name)
        elif dates == []:
            for year in range(int(draft_year)-3, int(draft_year)):
                dict_assignment(roster_dict, [str(year), school], player_name)
        elif len(dates) == 1:
            dict_assignment(roster_dict, [dates[0], school], player_name)

# hurt = player_time('https://en.wikipedia.org/wiki/Jalen_Hurts', 'Oklahoma')
# roster_update(roster_dict, 'Jalen Hurts', hurt[1], hurt[0], '2020')

# murray = player_time('https://en.wikipedia.org/wiki/Kenneth_Murray_(American_football)', 'Oklahoma')
# roster_update(roster_dict, 'Murray', murray[1], murray[0], '2020')

with open('draft_dict.json', 'r', encoding = 'utf-8') as file2read:
    draft_dict = json.load(file2read)


for year in list(draft_dict.keys()):
    for team in list(draft_dict[year].keys()):
        players = draft_dict[year][team]
        # print(players)
        for player in players:
            player_name = player[0:4]
            player_name.append(year)
            print(player)
            if player[4] != None:
                link = 'https://en.wikipedia.org' + player[4]
                schools = player_time('https://en.wikipedia.org' + player[4], team)[1]
                dates = player_time('https://en.wikipedia.org' + player[4], team)[0]
                print(schools, dates)
                roster_update(roster_dict, player_name, schools, dates, year)
            else:
                roster_update(roster_dict, player_name, [team], [[]], year)

with open('rosters.json', 'w', encoding = 'utf-8') as file2write:
    json.dump(roster_dict, file2write)

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

# # # def func(save_draft)

