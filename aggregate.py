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
        self.aggregatedFiles['testCoachHistory'] = 'files/testCoachHistory.json'
        self.aggregatedFiles['testGameHistory'] = 'files/testGameHistory.json'
        return