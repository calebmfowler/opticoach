'''this file maps alternative names of schools to standardized names'''


import json as json
import re as re


with open('total_no_mascots_inv.json', 'r') as file2read:
    no_mascots = json.load(file2read)
    
no_mascots_abr = {}
for i in (list(no_mascots.keys())):
    no_mascots_abr[i] = i
    

    
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



fcs_abbreviations = {
    "Abilene Christian": "ACU",
    "Alabama A&M": "AAMU",
    "Alabama State": "ALST",
    "Albany": "UALB",
    "Alcorn State": "ALCN",
    "Arkansas Pine Bluff": "UAPB",
    "Austin Peay": "APSU",
    "Bethune Cookman": "BCU",
    "Brown": "BROWN",
    "Bryant": "BRY",
    "Bucknell": "BUCK",
    "Butler": "BUT",
    "California Davis": "UCD",
    "California Poly": "CP",
    "Campbell": "CAMP",
    "Central Arkansas": "UCA",
    "Central Connecticut": "CCSU",
    "Charleston Southern": "CSU",
    "Chattanooga": "CHAT",
    "Clark Atlanta": "CAU",
    "Colgate": "COLG",
    "Columbia": "COLUM",
    "Cornell": "CORN",
    "Dartmouth": "DART",
    "Davidson": "DAV",
    "Dayton": "DAY",
    "Delaware": "DEL",
    "Delaware State": "DSU",
    "Drake": "DRAKE",
    "Duquesne": "DUQ",
    "East Tennessee State": "ETSU",
    "Eastern Illinois": "EIU",
    "Eastern Kentucky": "EKU",
    "Eastern Washington": "EWU",
    "Elon": "ELON",
    "Florida A&M": "FAMU",
    "Fordham": "FORD",
    "Furman": "FUR",
    "Gardner Webb": "GWU",
    "Georgetown": "GTWN",
    "Grambling State": "GRAM",
    "Hampton": "HAM",
    "Harvard": "HARV",
    "Howard": "HOW",
    "Houston Christian": "HCU",
    "Idaho": "IDHO",
    "Idaho State": "IDST",
    "Illinois State": "ILST",
    "Incarnate Word": "UIW",
    "Indiana State": "INST",
    "Jackson State": "JST",
    "Jacksonville State": "JSU",  # Now moving to FBS but including for now
    "James Madison": "JMU",        # Now FBS but historically FCS powerhouse
    "Kennesaw State": "KSU",
    "Lafayette": "LAF",
    "Lamar": "LAM",
    "Lehigh": "LEH",
    "Long Island": "LIU",
    "Maine": "MAINE",
    "Marist": "MAR",
    "McNeese State": "MCNS",
    "Merrimack": "MERR",
    "Mississippi Valley State": "MVSU",
    "Missouri State": "MOST",
    "Monmouth": "MONM",
    "Montana": "MONT",
    "Montana State": "MTSU",
    "Morehead State": "MOR",
    "Morgan State": "MSU",
    "Murray State": "MURR",
    "New Hampshire": "UNH",
    "Nicholls State": "NICH",
    "Norfolk State": "NSU",
    "North Alabama": "UNA",
    "North Carolina A&T": "NCAT",
    "North Carolina Central": "NCCU",
    "North Dakota": "UND",
    "North Dakota State": "NDSU",
    "Northern Arizona": "NAU",
    "Northern Colorado": "UNC",
    "Northern Iowa": "UNI",
    "Northwestern State": "NWST",
    "Pennsylvania": "Penn",
    "Portland State": "PSU",
    "Prairie View A&M": "PVAM",
    "Presbyterian": "PRES",
    "Princeton": "PRIN",
    "Rhode Island": "URI",
    "Richmond": "RICH",
    "Robert Morris": "RMU",
    "Sacramento State": "SAC",
    "Sacred Heart": "SHU",
    "Samford": "SAM",
    "Sam Houston State": "SHSU",  # Transitioning FBS too, but historic FCS
    "San Diego": "USD",
    "Savannah State": "SAV",
    "Southeast Missouri State": "SEMO",
    "Southeastern Louisiana": "SELA",
    "Southern": "SOU",
    "Southern Illinois": "SIU",
    "South Carolina State": "SCSU",
    "South Dakota": "USD",
    "South Dakota State": "SDSU",
    "Southern Utah": "SUU",
    "St Francis": "SFU",
    "Stetson": "STET",
    "Stonehill": "STONE",
    "Tennessee Martin": "UTM",
    "Tennessee State": "TSU",
    "Tennessee Tech": "TNTC",
    "Texas Southern": "TXSO",
    "Towson": "TOW",
    "Valparaiso": "VALPO",
    "Villanova": "NOVA",
    "Virginia Military Institute": "VMI",
    "Wagner": "WAG",
    "Weber State": "WEB",
    "Western Carolina": "WCU",
    "Western Illinois": "WIU",
    "William and Mary": "William & Mary",
    "Wofford": "WOFF",
    "Yale": "YALE",
    'Virginia Military Institute': "VMI"
}



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
        
        
for i in list(no_mascots_abr.keys()):
    dict_assignment(no_mascots, [i], no_mascots_abr[i])
        
for i in list(fbs_abbreviations.keys()):
    dict_assignment(no_mascots, [i], fbs_abbreviations[i])
    
for i in list(fcs_abbreviations.keys()):
    dict_assignment(no_mascots, [i], fcs_abbreviations[i])
    
with open('coach_history_inv.json', 'r') as file2read:
    coaches = json.load(file2read)  
    
    
coach_keys = list(coaches['2020'].keys())

dict_assignment(no_mascots, ['North Carolina State'], 'North Carolina St')
dict_assignment(no_mascots, ['Oklahoma Panhandle'], 'Oklahoma Panhandle State')
dict_assignment(no_mascots, ['Sam Houston'], 'Sam Houston State')
dict_assignment(no_mascots, ['East Texas A&M'], 'Texas A&M-Commerce')
dict_assignment(no_mascots, ['North Carolina-Pembroke'], 'UNC Pembroke')
dict_assignment(no_mascots, ['Winston Salem'], 'Winston Salem State')
dict_assignment(no_mascots, ['Winston Salem'], 'Winston-Salem State')
dict_assignment(no_mascots, ['McNeese State'], 'McNeese')
dict_assignment(no_mascots, ['Nicholls State'], 'Nicholls')
dict_assignment(no_mascots, ['Houston Christian'], 'Houston Baptist')
dict_assignment(no_mascots, ['Jacksonville'], 'Jacksonville University')
dict_assignment(no_mascots, ['Massachusetts-Dartmouth'], 'UMass Dartmouth')
dict_assignment(no_mascots, ['Utah Tech'], 'Dixie State')
dict_assignment(no_mascots, ["Hawai'i"], 'Hawaii')
dict_assignment(no_mascots, ['Citadel'], 'The Citadel')
dict_assignment(no_mascots, ['Southern Mississippi'], 'Southern Miss')
dict_assignment(no_mascots, ['East Tennessee'], 'East Tennessee State')
dict_assignment(no_mascots, ['Lewis & Clark'], 'Lewis & Clark College')
dict_assignment(no_mascots, ['Missouri State'], 'Southwest Missouri State')
dict_assignment(no_mascots, ['Texas State'], 'Southwest Texas State')
dict_assignment(no_mascots, ['Texas A&M-Kingsville'], 'Texas A&I')
dict_assignment(no_mascots, ['West Texas A&M'], 'West Texas State')
dict_assignment(no_mascots, ['Tennessee Titans'], 'Houston Oilers')
dict_assignment(no_mascots, ['Fullerton State'], 'Cal State Fullerton')
dict_assignment(no_mascots, ['Las Vegas Raiders'], 'Oakland Raiders')
dict_assignment(no_mascots, ['Washington Commanders'], 'Washington Redskins')
dict_assignment(no_mascots, ['Los Angeles Chargers'], 'San Diego Chargers')
dict_assignment(no_mascots, ['Minnesota State-Moorhead'], 'Moorhead State')
dict_assignment(no_mascots, ['Edmonton Elks'], 'Edmonton Eskimos')
dict_assignment(no_mascots, ['Arizona Cardinals'], 'St. Louis Cardinals')
dict_assignment(no_mascots, ['Louisiana'], 'Louisiana–Lafayette')

for i in list(no_mascots.keys()):
    if '-' in i:
        i_list = i.split('-')
        i_str = ''
        for j in i_list:
            i_str += j + ' '
        i_str = i_str[:-1]
        dict_assignment(no_mascots, [i], i_str)
    if ' state' in i.lower():
        i_list = i.lower().split(' state')
        i_str = i[0:len(i_list[0])]
        new_list = list(no_mascots.keys())
        new_list[new_list.index(i)] = ''
        if i_str not in new_list:
            dict_assignment(no_mascots, [i], i_str)   
    else:
        i_str = i.strip() + ' State'
        # print(i_str)
        if i_str not in list(no_mascots.keys()):
            dict_assignment(no_mascots, [i], i_str)  
    if ' university' not in i.lower():
        i_str = i.strip() + ' University'
        # print(i_str)
        if i_str not in list(no_mascots.keys()):
            dict_assignment(no_mascots, [i], i_str) 
    if 'university of' not in i.lower():
        i_str = 'University of '+i.strip()
        # print(i_str)
        if i_str not in list(no_mascots.keys()):
            dict_assignment(no_mascots, [i], i_str)
    if 'univ. of' not in i.lower():
        i_str = 'Univ. of '+i.strip()
        # print(i_str)
        if i_str not in list(no_mascots.keys()):
            dict_assignment(no_mascots, [i], i_str)
    if ' college' not in i.lower():
        i_str = i.strip() + ' College'
        # print(i_str)
        if i_str not in list(no_mascots.keys()):
            dict_assignment(no_mascots, [i], i_str) 
    # key = re.sub(r"[–—−]", "-", i)
    # dict_assignment(no_mascots, [i], key)

no_mascots['Washington'] = ['Washington Huskies', 'Washington', 'UW', 'University of Washington', 'Univ. of Washington', 'Washington College']


school_aliases = no_mascots
alias_to_school = {}

for school, aliases in school_aliases.items():
    for alias in aliases:
        alias_to_school[alias] = school

high_school = []
college = []
import re
for key in coach_keys:
    if '[' in key:
        new_key = ''
        start = False
        for j in key:
            if j == '[':
                start = True
            elif j== ']':
                start = False
            elif start == False:
                new_key += j
        key = new_key.strip()
    if 'HS' in key or 'school' in key.lower() or 'academy' in key.lower():
        high_school.append(key)
    else:
        text = key
        key = re.sub(r"[–—−]", "-", text)
        try:
            if '/' in key:
                key = key.split('/')[-1].strip()
            if ' and ' in key:
                key_list = key.split(' and ')
                key = key_list[0] + ' & ' + key_list[-1]
            college.append(alias_to_school[key])
        except:
            # print(key, 'failed')
            dummy = True
            
with open('mapping_schools.json', 'w') as file2write:
    json.dump(alias_to_school, file2write)
            
            
