from copy import deepcopy

class Aggregator:
    '''
    ### class Aggregator
    This `class` aggregates data via web-scraping, OCR of sports statistics books, etc.

    ### dict aggregatedFiles
    This `dict` stores `string` keys of all aggregated files and `string` values of file names

    ### void __init__(self)
    This `void` function is called as the Aggregator constructor. If provided an Aggregator,
    it operates as a copy constructor.

    ### void aggregate()
    This `void` function aggregates all data, updating the files referenced by aggregatedFiles
    '''

    def __init__(self, arg=None):
        if arg == None:
            self.aggregatedFiles = {}
        elif type(arg) == Aggregator:
            self.aggregatedFiles = deepcopy(arg.aggregatedFiles)
        else:
            raise Exception("Incorrect arguments for Aggregator.__init__(self, arg=None)")
        return

    def aggregate(self):
        self.aggregatedFiles['rosters'] = 'files/rosters.json' #json of college rosters including draft information
        self.aggregatedFiles['draft'] = 'files/draft_dict.json'
        self.aggregatedFiles['records'] = 'files/records.json' #json of pro and college games
        self.aggregatedFiles['polls'] = 'files/polls.json' #json of the AP Poll by year/week
        self.aggregatedFiles['nfl_links'] = 'files/nfl_links.json' #json of associated score table links for NFL teams
        self.aggregatedFiles['cfl_links'] = 'files/cfl_links.json' #json of associated score table links for CFL teams
        self.aggregatedFiles['ufl_links'] = 'files/ufl_links.json' #json of associated score table links for UFL teams
        self.aggregatedFiles['arenafl_links'] = 'files/arenafl_links.json' #json of associated score table links for Arena FL teams
        self.aggregatedFiles['map'] = 'files/mapping_schools.json' #json mapping school name variations to a single school name
        self.aggregatedFiles['coaches'] = 'files/trimmed_coach_dictionary.json' #json of coach history
        return