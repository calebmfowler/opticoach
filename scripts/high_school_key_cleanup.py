'''this file identifies and cleans keys corresponding to high schools'''



with open('school_keys_text_edit.txt', 'r', encoding='utf-8') as file2read:
    file = file2read.readlines()
    high_schools = []
    not_HS = []
    for line in file:
        if 'HS' in line:
            high_schools.append(line.strip())
        else:
            not_HS.append(line.strip())
            
            
    print(high_schools)
    
import json as json

with open('coach_history.json', 'r', encoding= 'utf-8') as file2read:
    coaches = json.load(file2read)


list1 = not_HS
list2 = high_schools

results = []
for item in list1:
    found = any(item in s for s in list2)
    results.append((item, found))

# Output the results
for item, is_found in results:
    if is_found == True:
        print(f"'{item}' found in list2: {is_found}")
        
        
trimmed_hs = []
for i in high_schools:
    try:
        coach_dict = coaches[i]
        years= []
        for j in list(coach_dict.keys()):
            years.append(int(j))
        if max(years) >= 1975:
            trimmed_hs.append(i)
    except:
        print(i, 'failed')
        
        
state_abbreviations = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
]
        

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

hs_coaches = {}


with open('total_no_mascots_inv.json', 'r', encoding = 'utf-8') as file2read:
    no_mascots = json.load(file2read)
    
    
with open('mapping_schools.json', 'r', encoding = 'utf-8') as file2read:
    maps = json.load(file2read)


maps['UT-Martin'] = 'Tennessee-Martin'

fail_list = []
hs_list = []

coach_abbreviations = [
    "HC",  # Head Coach
    "OC",  # Offensive Coordinator
    "DC",  # Defensive Coordinator
    "QB",  # Quarterbacks Coach
    "RB",  # Running Backs Coach
    "WR",  # Wide Receivers Coach
    "OL",  # Offensive Line Coach
    "DL",  # Defensive Line Coach
    "LB",  # Linebackers Coach
    "DB",  # Defensive Backs Coach
    "ST",  # Special Teams Coach  
    "TE",  # Tight Ends Coach
    "SA",  # Student Assistant
    "QC",  # Quality Control
    "RC",  # Recruiting Coordinator
    "AHC",  # Assistant Head Coach
    "PGC",  # Pass Game Coordinator
    "RGC",  # Run Game Coordinator
    "S&C", # Strength & Conditioning
    "A" #assistant
]

'''fix John Settle'''


for i in coaches:
    if '(' not in i:
        coach_list = coaches[i]
        for year in list(coach_list.keys()):
            for position in list(coach_list[year].keys()):
                coach = coach_list[year][position]
                if position in state_abbreviations and int(year) > 1974:
                    if i not in list(maps.keys()):
                        dict_assignment(hs_coaches, [i + ' (' + position + ')', year, 'HC'], coach)
                        hs_list.append(i)
                elif position == 'GA' and int(year) > 1974 and 'HS' in i:
                    dict_assignment(hs_coaches, [i + ' (' + position + ')', year, 'HC'], coach)
                    hs_list.append(i)
                elif position in coach_abbreviations and int(year) > 1974:
                    dict_assignment(hs_coaches, [i, year, position], coach)
                    hs_list.append(i)
    else:
        coach_list = coaches[i]
        for year in list(coach_list.keys()):
            for position in list(coach_list[year].keys()):
                coach = coach_list[year][position]
                if position in state_abbreviations and int(year) > 1974:
                    print(coach, 'is sus')
                if int(year) > 1974 and position != 'CH':
                    dict_assignment(hs_coaches, [i, year, position], coach)
                    hs_list.append(i)

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


failed_keys = []
for i in list(coaches.keys()):
    if i not in hs_list and i not in list(maps.keys()):
        years = []
        for j in list(coaches[i].keys()):
            years.append(int(j))
        if i not in failed_keys and max(years) > 1974:
            failed_keys.append(i)


import re
from collections import defaultdict

# Sample dictionary
school_dict = hs_coaches

# Regex for keys with state abbreviation variants
state_abbr_pattern = re.compile(r"^(.*) \(([A-Z]{2})\)$")

# Group matches by base key
variants = defaultdict(list)

for key in school_dict:
    match = state_abbr_pattern.match(key)
    if match:
        base_name = match.group(1)
        variants[base_name].append(key)

# Construct the output
conflicts = []

for base, matches in variants.items():
    if len(matches) == 1:
        maps[base] = matches[0]
    else:
        conflicts.append((base, matches))

# Output
print("maps =", maps)
print("conflicts =", conflicts)

for i in conflicts:
    for j in i[1]:
        maps[j] = j