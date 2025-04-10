class Aggregator:
    '''
    ### class Aggregator
    This `class` aggregates data via web-scraping, OCR of sports statistics books, etc.

    ### dict aggregatedFiles
    This `dict` stores `string` keys of all aggregated files and `string` values of file names

    ### void aggregate()
    This `void` function aggregates all data, updating the files referenced by aggregatedFiles
    '''

    def aggregate(self):
        self.aggregatedFiles = {}
        return