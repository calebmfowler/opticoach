class Postprocessor:
    '''
    # `class Preprocessor`
    This `class` postprocesses the predicted data into a format suitable for reporting results.

    ## `dict preprocessedFiles`
    This `dict` stores `string` keys of all postprocessed files and `string` values of file names

    ## `void preprocess()`
    This `void` function postprocesses all predicted data, updating the files referenced by postprocessedFiles
    '''

    postprocessedFiles = {}

    def postprocess(predicedFiles):
        return