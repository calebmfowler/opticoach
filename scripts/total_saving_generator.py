'''this file generates a dictionary capable of converting from no mascots to mascot-including names'''

import json as json
from bs4 import BeautifulSoup
import requests as requests

with open('total_no_mascots_inv.json', 'r') as file2read:
    no_mascots = json.load(file2read)
    
    
with open('ufl_links.json', 'r') as file2read:
    links = json.load(file2read)
    
    
mascots = list(links.keys())

def get_nomascot(link):
    '''this function takes in a link for a team and finds its simplest name'''
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Referer": "https://google.com",  # Spoofing as if coming from Google
        "Accept-Language": "en-US,en;q=0.9",
    }
    og_link = link
    data = requests.get(link, headers=headers)
    soup = BeautifulSoup(data.content, 'html.parser')
    link = soup.find('a')
    link = link.get('href')
    link_list = link.split('/f/')
    link = link_list[0]+'/f/z/' + link_list[1][:-4]+'_re.htm'
    data = requests.get(link, headers=headers)
    soup = BeautifulSoup(data.content, 'html.parser')
    links = soup.find_all('a')
    for link in links:
        link_val = link
        link_list = link.get('href').split('/f/')
        link = link_list[0]+'/f/z/' + link_list[1][:-4]+'_re.htm'
        if link == og_link:
            link = link_val
            break
    print(og_link, link_val.text.strip())
    return link.text.strip()


# with open('total_no_mascots.json', 'r') as file2read:
#     no_mascots = json.load(file2read)


for key in mascots:
    mascot = key
    link = links[mascot]
    no_mascot = get_nomascot(link)
    no_mascots[no_mascot] = mascot




with open('total_no_mascots_inv.json', 'w') as file2write:
    json.dump(no_mascots, file2write)
    