'''this file generates a json file that contains links for tables of school scores'''

link = 'https://pro-football-results.com/afl.htm'

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

schools_dict = {}
data = requests.get(link, headers=headers)
soup = BeautifulSoup(data.content, 'html.parser')
table = soup.find('table')
links = table.find_all('a')
for link in links:
    if link.text.strip() != '':
        try:
            print(link)
            link_list = link.get('href').strip().split('/f/')
            new_link = link_list[0]+'/f/z/'+link_list[1][:-4]+'_re.htm'
            schools_dict[link.text.strip()] = new_link
        except:
            print(link)
            break




with open('arenafl_links.json', 'w') as file2read:
    json.dump(schools_dict, file2read)

# seasons = {}
# for name in list(schools.keys()):
#     link = schools[name]
#     data = requests.get(link, headers=headers)
#     soup = BeautifulSoup(data.content, 'html.parser')
#     tables = pd.read_html(str(soup))