'''This program looks at all categories of coaches on a Wikipedia page and constructs a dictionary of coaching staffs.'''

# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
import csv  # For handling CSV files
import os  # For interacting with the operating system
import numpy as np  # For numerical operations
import json as json  # For handling JSON data
from bs4 import BeautifulSoup  # For parsing HTML and XML documents
import requests as requests  # For making HTTP requests

# Function to assign a value to a nested dictionary using a sequence of keys
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
    current = d  # Start at the root of the dictionary
    for key in keys[:-1]:  # Traverse through all keys except the last one
        if key not in current or not isinstance(current[key], dict):  # If key doesn't exist or isn't a dictionary
            current[key] = {}  # Create a new dictionary
        current = current[key]  # Move to the next level
    
    last_key = keys[-1]  # Get the last key
    if last_key in current:  # If the last key already exists
        if isinstance(current[last_key], list):  # If the value is a list
            if value not in current[last_key]:  # Prevent duplicates
                current[last_key].append(value)  # Append the value
        else:  # If the value is not a list
            if current[last_key] != value:  # Convert to a list only if the value is different
                current[last_key] = [current[last_key], value]
    else:  # If the last key doesn't exist
        current[last_key] = value  # Assign the value

# List to store indices
index_list = []

# List of US state abbreviations
state_abbreviations = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", 
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
]

# Function to process date strings into a list of date ranges
def date_intr(dates):
    """
    Converts a list of date strings into a list of date ranges or single years.

    :param dates: list - A list of date strings.
    :return: list - A list of processed date ranges or single years.
    """
    total_dates = []  # Initialize the list to store processed dates
    for date_str in dates:  # Iterate through each date string
        for i in date_str:  # Remove spaces from the string
            if i == ' ':
                date_str.replace(i, '')
        for i in date_str:  # Replace parentheses and dashes with standard characters
            if i == '(' or i == ')':
                date_str = date_str.replace(i, '')
            elif i == '-':
                date_str = date_str.replace(i, '–')
        if len(date_str) == 9:  # If the string represents a date range
            date1 = int(date_str[0:4])  # Extract the start year
            date2 = int(date_str[5:9])  # Extract the end year
            date_list = [date1, date2]  # Create a list of the range
        elif 'present' in date_str:  # If the string includes "present"
            date1 = int(date_str[0:4])  # Extract the start year
            date2 = 2024  # Assume the current year is 2024
            date_list = [date1, date2]  # Create a list of the range
        else:  # If the string represents a single year
            date_list = int(date_str)  # Convert the string to an integer
        total_dates.append(date_list)  # Append the processed date to the list
    return total_dates  # Return the list of processed dates

# Function to standardize coaching roles into abbreviations
def role_trans(list_of_roles):
    """
    Converts a list of coaching job descriptions into standardized abbreviations.

    :param list_of_roles: list - A list of coaching job descriptions.
    :return: tuple - A tuple containing a list of standardized roles and a list of failed conversions.
    """
    new_roles = []  # List to store standardized roles
    for i in list_of_roles:  # Iterate through each role in the list
        fail_list = []  # List to store roles that couldn't be standardized
        if ('&' in i or 'and' in i) and i.isupper() == False:  # If the role contains multiple jobs
            i_list = i.split('&')  # Split the role into individual jobs
            j_list = []  # List to store standardized roles for this job
            for j in i_list:  # Iterate through each job
                role = ''  # Initialize the standardized role
                j = j.strip()  # Remove leading and trailing spaces
                assistant = False  # Flag to indicate if the role is an assistant position
                if 'head coach' in j.lower():  # Check for specific job descriptions and assign abbreviations
                    role = 'HC'
                elif 'intern' in j.lower():
                    role = 'INT'
                elif 'tight end' in j.lower():
                    role = 'TE'
                elif 'offensive consultant' in j.lower() or 'offensive assistant' in j.lower():
                    role = 'OA'
                    assistant = True
                elif 'defensive consultant' in j.lower() or 'defensive assistant' in j.lower():
                    role = 'DA'
                    assistant = True
                elif 'quarterbacks' in j.lower() or 'quarterback' in j.lower():
                    role = 'QB'
                elif 'defensive back' in j.lower() or 'defensive backs' in j.lower():
                    role = 'DB'
                elif 'defensive coordinator' in j.lower():
                    role = 'DC'
                elif 'offensive coordinator' in j.lower():
                    role = 'OC'
                elif 'defensive line' in j.lower():
                    role = 'DL'
                elif 'offensive line' in j.lower():
                    role = 'OL'
                elif 'linebackers' in j.lower():
                    if 'inside' in j.lower():
                        role = 'ILB'
                    if 'outside' in j.lower():
                        role = 'OLB'
                    else:
                        role = 'LB'
                elif 'running game' in j.lower() or 'run game' in j.lower():
                    role = 'RGC'
                elif 'passing game' in j.lower() or 'pass game' in j.lower():
                    role = 'PGC'
                elif 'wide receivers' in j.lower() or 'receiver' in j.lower():
                    role = 'WR'
                elif 'special teams' in j.lower():
                    role = 'ST'
                elif 'punter' in j.lower():
                    role = 'P'
                elif 'graduate assistant' in j.lower():
                    role = 'GA'
                    assistant = True
                elif 'secondary' in j.lower():
                    role = 'CB/S'
                elif 'safet' in j.lower():
                    role = 'S'
                elif 'cornerback' in j.lower():
                    role = 'CB'
                elif 'interim head' in j.lower():
                    role = 'IHC'
                elif 'kicker' in j.lower():
                    role = 'K'
                elif 'defensive analyst' in j.lower():
                    role = 'DFAL'
                elif 'offensive analyst' in j.lower():
                    role = 'OFAL'
                elif 'analyst' in j.lower():
                    role = 'AL'
                elif 'recruiting' in j.lower():
                    role = 'RC'
                elif 'running back' in j.lower() or 'running backs' in j.lower():
                    role = 'RB'
                elif 'defensive end' in j.lower() or 'ends' in j.lower():
                    role = 'DE'
                elif 'defensive tackle' in j.lower():
                    role = 'DT'
                elif 'offensive tackle' in j.lower():
                    role = 'OT'
                elif 'coach' == j.lower():
                    role = 'CH'
                elif 'line' in j.lower():
                    role = 'L'
                elif 'offensive quality control' in j.lower():
                    role = 'OQC'
                elif 'defensive quality control' in j.lower():
                    role = 'DQC'
                elif 'quality' in j.lower():
                    role = 'QC'
                elif 'strength' in j.lower() or 'conditioning' in j.lower():
                    role = 'S&C'
                elif 'consultant' in j.lower():
                    role = 'CON'
                elif 'scout' in j.lower():
                    role = 'SC'
                elif 'backs' in j.lower() or 'offensive backfield' in j.lower():
                    role = 'B'
                elif 'edge' in j.lower():
                    role = 'E'
                if assistant == False and 'ass' in j.lower():  # Add "A" prefix for assistant roles
                    role = 'A' + role
                if 'co-' in j.lower():  # Add "co-" prefix for co-roles
                    role = 'co-' + role
                j_list.append(role)  # Append the standardized role to the list
                if role == '':  # If the role couldn't be standardized, add it to the fail list
                    fail_list.append(j)
            new_roles.append(j_list)  # Append the list of standardized roles to the main list
        elif i.isupper() == False:  # If the role is a single job
            j = i
            role = ''
            j = j.strip()
            assistant = False
            if 'head coach' in j.lower():
                role = 'HC'
            elif 'intern' in j.lower():
                role = 'INT'
            elif 'tight end' in j.lower():
                role = 'TE'
            elif 'offensive consultant' in j.lower() or 'offensive assistant' in j.lower():
                role = 'OA'
                assistant = True
            elif 'defensive consultant' in j.lower() or 'defensive assistant' in j.lower():
                role = 'DA'
                assistant = True
            elif 'quarterbacks' in j.lower() or 'quarterback' in j.lower():
                role = 'QB'
            elif 'defensive back' in j.lower() or 'defensive backs' in j.lower():
                role = 'DB'
            elif 'defensive coordinator' in j.lower():
                role = 'DC'
            elif 'offensive coordinator' in j.lower():
                role = 'OC'
            elif 'defensive line' in j.lower():
                role = 'DL'
            elif 'offensive line' in j.lower():
                role = 'OL'
            elif 'linebackers' in j.lower():
                if 'inside' in j.lower():
                    role = 'ILB'
                if 'outside' in j.lower():
                    role = 'OLB'
                else:
                    role = 'LB'
            elif 'running game' in j.lower() or 'run game' in j.lower():
                role = 'RGC'
            elif 'passing game' in j.lower() or 'pass game' in j.lower():
                role = 'PGC'
            elif 'wide receivers' in j.lower() or 'receiver' in j.lower():
                role = 'WR'
            elif 'special teams' in j.lower():
                role = 'ST'
            elif 'punter' in j.lower():
                role = 'P'
            elif 'graduate assistant' in j.lower():
                role = 'GA'
                assistant = True
            elif 'secondary' in j.lower():
                role = 'CB/S'
            elif 'safet' in j.lower():
                role = 'S'
            elif 'cornerback' in j.lower():
                role = 'CB'
            elif 'interim head' in j.lower():
                role = 'IHC'
            elif 'kicker' in j.lower():
                role = 'K'
            elif 'defensive analyst' in j.lower():
                role = 'DFAL'
            elif 'offensive analyst' in j.lower():
                role = 'OFAL'
            elif 'analyst' in j.lower():
                role = 'AL'
            elif 'recruiting' in j.lower():
                role = 'RC'
            elif 'running back' in j.lower() or 'running backs' in j.lower():
                role = 'RB'
            elif 'defensive end' in j.lower() or 'ends' in j.lower():
                role = 'DE'
            elif 'defensive tackle' in j.lower():
                role = 'DT'
            elif 'offensive tackle' in j.lower():
                role = 'OT'
            elif 'coach' == j.lower():
                role = 'CH'
            elif 'line' in j.lower():
                role = 'L'
            elif 'offensive quality control' in j.lower():
                role = 'OQC'
            elif 'defensive quality control' in j.lower():
                role = 'DQC'
            elif 'quality' in j.lower():
                role = 'QC'
            elif 'strength' in j.lower() or 'conditioning' in j.lower():
                role = 'S&C'
            elif 'consultant' in j.lower():
                role = 'CON'
            elif 'scout' in j.lower():
                role = 'SC'
            elif 'backs' in j.lower() or 'offensive backfield' in j.lower():
                role = 'B'
            elif 'edge' in j.lower():
                role = 'E'
            if assistant == False and 'ass' in j.lower():
                role = 'A' + role
            if 'co-' in j.lower():
                role = 'co-' + role
            new_roles.append(role)
            if role == '':
                fail_list.append(j)
        else:
            new_roles.append(i)
            if i == '' :
                fail_list.append(i)
    return new_roles, fail_list

def split_roles(split_role):
    '''This function accepts a string corresponding to a coaching job abbreviation, and either converts it into a list of simultaneous role abbreviations or a string corresponding to a single role role abbreviation.'''
    for i in split_role:
        if i == '(' or i == ')':
            split_role = split_role.replace(i, '')
    if '/' in split_role:
        roles = split_role.split('/')
    else:
        roles = split_role
    return roles
        


def college_format(my_csv):
    '''This function returns a tuple of three lists, corresponding to the job history, location history, and year history of a coach's data when formatted in the college style.'''
    #initialize lists of job history, location history, and time history
    history = []
    schools = []
    years = []
    #identify the start point of coaching history by iterating through csv rows
    for i in range(len(my_csv)-1):
        #get the current and following rows as lists
        row = my_csv[i]
        next_row = my_csv[i+1]
        #identify if the row is a coaching career header and contains dates in the next row
        if 'Coaching career' in row[1] and next_row[1][0].isalpha()==False or 'Coaching career' in row[1] and next_row[1][0].islower()==True:
            start = i+1
            break
        #identify if the next row is a football header
        elif 'Coaching career' in row[1] and 'football' in next_row[1].lower():
            start = i+2
            break
        elif 'career history' in row[1].lower() and 'as coach' in next_row[1].lower():
            start = i+2
            break
    #iterate through the csv to get dates, jobs, and locations
    for row in my_csv[start:]:
        #break the loop when the football dates end
        if row[1] != '' and row[1][0].isalpha() == True and row[1][0].islower()==False:
            break
        elif row[1] == '':
            break
        else:
            #get the coaching location and job
            #split the data using spaces
            data = row[2].split(' ')
            #identify if the job is a head coaching job if there is not abbreviation in brackets
            #NOTE: can still be a head coach if a state abbreviation exists
            # print(data)
            if ')' not in data[len(data)-1]:
                #append 'HC' (head coach) to coaching job list
                history.append('HC')
                #create the name of the location
                school = ''
                for j in data:
                    school += j
                    school += ' '
                #append the location to the location list (remove final space)
                schools.append(school[:-1])
            elif ')' in data[len(data)-1]:
                position = ''
                index = len(data)-1
                while '(' not in data[index]:
                    position = data[index]+' '+position
                    index = index-1
                position = data[index]+' '+position
                school = ''
                for i in data[0:index]:
                    school += i + ' '
                schools.append(school.strip())
                history.append(position.strip()[1:-1])
                
                
                
                
            #if a position is included in parentheses
            else:
                #append the position to the job history list
                history.append(data[len(data)-1][1:-1])
                #determine the location
                length = len(data)-1
                school = ''
                for j in data[:length]:
                    school += j
                    school += ' '
                #append the location to the location list (remove final space)
                schools.append(school[:-1])
        #try to split the list range
        try:
            year = row[1].split('–')
            #NOTE: this code only works prior to the upcoming football season
            #convert the year 'present' to the string '2024' if needed
            if year[1] == 'present':
                year[1] = '2024'
            #append a list of the start and end dates for the position, as integers, to the overall year list
            years.append([int(year[0]), int(year[1])])
        #otherwise if the position is only held for one year
        except:
            #append the year, as an integer, to the overall year list
            #ensure it is just first four characters of string
            #account for circa
            if row[1][0] != 'c':
                years.append(int(row[1][0:4]))
            else:
                for k in range(len(row[1])):
                    if row[1][k].isnumeric() == True:
                        years.append(int(row[1][k:k+4]))
                        break
    #convert all roles to standardized abbreviations
    #determine split roles and convert them to lists of single roles
    for k in range(len(history)):
        if '/' in history[k]:
            history[k] = split_roles(history[k])
    #return tuple of lists
    # print(history)
    return (history, schools, years)


def pro_format(my_csv):
    '''This function returns a tuple of three lists, corresponding to the job history, location history, and year history of a coach's data when formatted in the pro style.'''
    index_list = []
    place_list = []
    job_list = []
    date_list = []
    for i in range(len(my_csv)):
        if 'as a coach' in my_csv[i][1].lower() and 'as a coach' in my_csv[i][2].lower():
            index_list.append(i)
    
    index = index_list[0]+1
    text = my_csv[index][1]
    text_list = text.split(' ')
    # print(text_list)
    new_list = []
    for i in range(len(text_list)):
        if text_list[i] != '':
            new_list.append(text_list[i])
    text_list = new_list
    index_list = []
    for j in range(len(text_list)):
        if j != len(text_list)-1:
            try: 
                number = text_list[j][2].isnumeric()
            except:
                number = False
            if number == True and text_list[j+1][0].isupper() == True and text_list[j+2][0].isupper()==True:
                index_list.append(j+1)
            elif text_list[j][0].islower() == True and text_list[j+1][0].isupper() == True:
                index_list.append(j)
        else:
            index_list.append(j)
    # print(text_list)
    # print(index_list)
    role_list = []
    start = 0

    for i in range(len(index_list)):
        end = index_list[i]+1
        role = text_list[start:end]
        role_list.append(role)
        start = end
    for i in role_list:
        # print(role_list)
        for j in range(len(i)):
            if '(' in i[j] and '(' not in i[j+1]:
                date_index = j
                break
        date_start = date_index + 1
        date = i[date_index]
        if ')' not in date:
            date += i[date_index+1]
            date_start = date_index + 2
        place = ''
        job = ''
        for k in i[0:date_index]:
            place += k + ' '
        for m in i[(date_start):]:
            job += m + ' '
        place = place.strip()
        job = job.strip()
        place_list.append(place)
        job_list.append(job)
        date_list.append(date)
    # print(date_list)
    # print(job_list)
    # print(place_list)
    return (role_trans(job_list), place_list, date_intr(date_list))



def pro_format(my_csv, link):
    '''This function returns a tuple of three lists, corresponding to the job history, location history, and year history of a coach's data when formatted in the pro style.'''
    index_list = []
    place_list = []
    job_list = []
    date_list = []
    for i in range(len(my_csv)):
        if 'as a coach' in my_csv[i][1].lower() and 'as a coach' in my_csv[i][2].lower():
            index_list.append(i)
    
    url = link
    data = requests.get(url)
    soup = BeautifulSoup(data.content, 'html.parser')

    table = soup.find_all("table", class_ = 'infobox vcard')[0]
    rows = table.find_all('tr')
    for i in range(len(rows)):
        row = rows[i]
        th_finder = row.find('th')
        if th_finder != None:
            if 'as a coach' in th_finder.text.lower():
                history_row = rows[i+1]
                break
    stops = history_row.find_all('li')
    for stop in stops:
        stop_list = stop.text.split(')')
        # print(stop_list)
        stop_list_0 = ''
        # print(stop_list)
        for i in stop_list[:-1]:
            stop_list_0 += i + ')'
        stop_list = [stop_list_0.strip(), stop_list[-1].strip()]
        if stop_list[0] == '':
            break
        role = stop_list[-1]
        stop_list = stop_list[0].split('(')
        date = '(' + stop_list[-1]
        og_date = date
        new_date = ''
        for j in date:
            if j == ' ':
                dummy = True
            else:
                new_date += j
        date = new_date
        place = stop_list[0]
        for i in stop_list[1:-1]:
            place += '(' + i
        if role.strip() == '':
            if stop_list[-1][-2].isnumeric() == True or 'present' in stop_list[-1]:
                role = 'coach'
            else:
                role = og_date.strip()[1:-1]
                stop_list = [place.strip(), role]
                stop_list = stop_list[0].split('(')
                date = '(' + stop_list[-1]
                new_date = ''
                for j in date:
                    if j == ' ':
                        dummy = True
                    else:
                        new_date += j
                date = new_date
                place = stop_list[0]
                for i in stop_list[1:-1]:
                    place += '(' + i
        # print(place, role, date)
        place_list.append(place.strip())
        job_list.append(role.strip())
        date_list.append(date.strip())
    print(job_list)
    return (role_trans(job_list)[0], place_list, date_intr(date_list))
        
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
  
def list_to_dict(dictionary, job_list, place_list, time_list, name):
    #iterate through list of dates
    for i in range(len(time_list)):
        # print('Place is', place_list[i])
        # print('Job is', job_list[i])
        # print('Time is', time_list[i])
        if type(time_list[i]) == list and time_list[i][0] != time_list[i][1]:
            time_range = np.arange(time_list[i][0], time_list[i][1]+1)
        elif type(time_list[i]) == list and time_list[i][0] == time_list[i][1]:
            time_range = [time_list[i][0]]
        else:
            time_range = [time_list[i]]
        for time in time_range:
            if type(job_list[i]) == list:
                for job in job_list[i]:
                    dict_assignment(dictionary, [place_list[i], str(time), job], name)
            else:
                # print([place_list[i], str(time), job_list[i]])
                dict_assignment(dictionary, [place_list[i], str(time), job_list[i]], name)
      
        
def coaches_to_dict(dictionary, url):
    schools_dict = dictionary
    data = requests.get(url)
    soup = BeautifulSoup(data.content, 'html.parser')
    #find all hyperlinks on the disambiguation website
    links = soup.find_all("a")  # Find all elements with the tag "a"
    
    
    
    # tables = pd.read_html(url, flavor = 'bs4')
    # print(tables[0])
    
    # tables = soup.find_all('<table', class_='wikitable sortable plainrowheaders jquery-tablesorter')
    for link in links:
        if 'This list may not' in link.text.strip():
            header = links.index(link)
            break
        # if link.text.strip() == 'next page':
        #     page_link = link.get('href')
        #     page_link_index = links.index(link)
        #     break
        # elif ''
    if links[header+1].text.strip() == 'next page':
        link = links[header+1]
        page_link = link.get('href')
        page_link_index = links.index(link)
        start = header + 3
    elif 'List of' in links[header+1].text.strip():
        start = header+2
        page_link = None
    else:
        page_link = None
        start = header + 1
    
    
    links_list = []
    for link in links[start:]:
        if 'index' in link.get('href'):
            break
        else:
            links_list.append('https://en.wikipedia.org'+link.get('href'))
            
    if page_link != None:
        url = 'https://en.wikipedia.org' + page_link
        data = requests.get(url)
        soup = BeautifulSoup(data.content, 'html.parser')
        #find all hyperlinks on the disambiguation website
        links = soup.find_all("a")  # Find all elements with the tag "a"
        for link in links:
            if 'This list may not' in link.text.strip():
                header = links.index(link)
                break
        start = header + 2
        for link in links[start:]:
            if 'index' in link.get('href') or link.text.strip() == 'previous page':
                break
            else:
                links_list.append('https://en.wikipedia.org'+link.get('href'))
    # print(page_link)
    # print(links_list)
    
    
    
    fail_list = []
    fail_list_missing = []
    # links_list = ['https://en.wikipedia.org/wiki/Greg_Mattison']
    for link in links_list:
        # print(link)
        try:
            data = requests.get(link)
            soup = BeautifulSoup(data.content, 'html.parser')
            #find all hyperlinks on the disambiguation website
            tables = soup.find_all("table", class_ = 'infobox vcard')
            table = tables[0] # Find all elements with the tag "a"
            coach_bio_tables = pd.read_html(str(table), flavor = 'bs4')
            table = coach_bio_tables[0].to_csv('temp coach file.csv')
            # coach_bio_tables = pd.read_html(link, flavor = 'bs4')
            # if 'citations' in coach_bio_tables[0].to_string():
            #     coach_bio = coach_bio_tables[1].to_csv('temp coach file.csv')
            # else:
            #     coach_bio = coach_bio_tables[0].to_csv('temp coach file.csv')
        except:
            # try:
            #     # data = requests.get(link)
            #     # soup = BeautifulSoup(data.content, 'html.parser')
            #     # #find all hyperlinks on the disambiguation website
            #     # tables = soup.find_all("table", class_ = 'infobox vcard')
            #     # table = tables[0] # Find all elements with the tag "a"
            #     # coach_bio_tables = pd.read_html(str(table), flavor = 'bs4')
            #     # table = coach_bio[0].to_csv('temp coach file.csv')
            # except:
            coach_bio_tables = None
        if coach_bio_tables != None:
            with open('temp coach file.csv', encoding = 'utf-8') as file2read:
                my_csv = list(csv.reader(file2read))
            data = requests.get(link)
            soup = BeautifulSoup(data.content, 'html.parser')
            #find all hyperlinks on the disambiguation website
            tables = soup.find_all("table", class_ = 'infobox vcard')
            # try:
            if tables != []:
                try:
                    coach_name = tables[0].find('caption', class_ = 'infobox-title fn').text# Find all elements with the tag "a"
                except:
                    coach_name = None
                if coach_name != None:
                    try:
                        data = college_format(my_csv)
                        # print('college')
                        list_to_dict(schools_dict, data[0], data[1], data[2], coach_name)
                    except:
                        try:
                            data = pro_format(my_csv, link)
                            # print('pro')
                            list_to_dict(schools_dict, data[0], data[1], data[2], coach_name)
                        except:
                            fail_list.append(link)
                            # print(link, 'failed')
                            coach_bio_tables = pd.read_html(link, flavor = 'bs4')
                            if 'citations' in coach_bio_tables[0].to_string():
                                coach_bio = coach_bio_tables[1].to_csv('temp coach file.csv')
                            else:
                                coach_bio = coach_bio_tables[0].to_csv('temp coach file.csv')
                            with open('temp coach file.csv', encoding = 'utf-8') as file2read:
                                my_csv = list(csv.reader(file2read))
                                new_list = ''
                                for i in my_csv:
                                    for j in i:
                                        new_list += j
                                my_csv = new_list
                                # print(my_csv)
                                if 'coaching career' in my_csv.lower() or 'as a coach' in my_csv.lower():
                                    dummy = True
                                else:
                                    fail_list_missing.append(link)
                else:
                    fail_list.append(link)
        else:
              fail_list.append(link)   
    return schools_dict, fail_list, fail_list_missing

with open('school_names.txt', 'r') as file2read:
    school_list = []
    for i in file2read.readlines():
        school_list.append('https://en.wikipedia.org/wiki/Category:'+i.strip()+'_football_coaches')
# print(school_list)
url_list = school_list
total_fail_list = []
url_list[105] = 'https://en.wikipedia.org/wiki/Category:Louisiana%E2%80%93Monroe_Warhawks_football_coaches'

# # Open the JSON file
# with open('../coach_history.json') as f:
#     schools_dict = json.load(f)

# counter = 165
# for i in url_list[164:]:
#     print('School ', counter)
#     print(i)
#     schools_dict, fail_list, fail_list_missing = coaches_to_dict(schools_dict, i)
#     with open('../coach_history.json', 'w') as json_file:
#         json.dump(schools_dict, json_file, indent=4)
#     for j in fail_list:
#         total_fail_list.append(j)
#     counter += 1

# with open('../coach_history.json') as f:
#     schools_dict = json.load(f)

# data = pro_format(my_csv, link)
# schools_dict, fail_list, fail_list_missing = coaches_to_dict(schools_dict, i)
    # with open('../coach_history.json', 'w') as json_file:
    #     json.dump(schools_dict, json_file, indent=4)


# # Open the JSON file
# with open('../coach_history.json') as f:
#     schools_dict = json.load(f)

schools_dict = {}
fail_list = ['https://en.wikipedia.org/wiki/Chip_Vaughn']
for i in fail_list:
    link = i
    print(link)
    # coach_bio_tables = pd.read_html(link, flavor = 'bs4')
    # if 'citations' in coach_bio_tables[0].to_string():
    #     print('detected')
    #     coach_bio = coach_bio_tables[1].to_csv('temp coach file.csv')
    # else:
    #     coach_bio = coach_bio_tables[0].to_csv('temp coach file.csv')
    
    data = requests.get(link)
    soup = BeautifulSoup(data.content, 'html.parser')
    #find all hyperlinks on the disambiguation website
    tables = soup.find_all("table", class_ = 'infobox vcard')
    table = tables[0] # Find all elements with the tag "a"
    coach_bio = pd.read_html(str(table), flavor = 'bs4')
    table = coach_bio[0].to_csv('temp coach file.csv')
    
    with open('temp coach file.csv', encoding = 'utf-8') as file2read:
        my_csv = list(csv.reader(file2read))
    data = requests.get(link)
    soup = BeautifulSoup(data.content, 'html.parser')
    #find all hyperlinks on the disambiguation website
    tables = soup.find_all("table", class_ = 'infobox vcard')
    coach_name = tables[0].find('caption', class_ = 'infobox-title fn').text# Find all elements with the tag "a"
    # try:
    # data = college_format(my_csv)
    # print('college')
    # list_to_dict(schools_dict, data[0], data[1], data[2], coach_name)
    # # except:
    data = pro_format(my_csv, link)
    # print('pro')
    list_to_dict(schools_dict, data[0], data[1], data[2], coach_name)

# with open('../coach_history.json', 'w') as json_file:
#     json.dump(schools_dict, json_file, indent=4)


'''need to fix pro format for when capitalization rules arent followed'''
