from collections import defaultdict
from pandas import DataFrame, Series
import pickle as pkl
import json

def save_pkl(data, filename='MISSING_FILENAME.pkl'):
    """
    Save data to a pickle (.pkl) file.

    Parameters:
    -----------
    data : any
        The Python object to be serialized and saved.
    filename : str, optional
        The name of the file to save the data to. Defaults to 'MISSING_FILENAME.pkl'.

    Returns:
    --------
    None
        Prints a confirmation message when saving is complete.
    """

    file = open(filename, 'wb')
    pkl.dump(data, file)
    file.close()
    print(f"\nData saved to {filename}\n")

def load_pkl(filename='MISSING_FILENAME.pkl'):
    """
    Load and deserialize a Python object from a pickle (.pkl) file.

    Parameters:
    -----------
    filename : str, optional
        The name of the pickle file to load. Defaults to 'MISSING_FILENAME.pkl'.

    Returns:
    --------
    data : any
        The Python object that was deserialized from the file.

    Side Effects:
    -------------
    Prints a confirmation message upon successful loading.
    """

    file = open(filename, 'rb')
    data = pkl.load(file)
    file.close()
    print(f"\nData loaded from {filename}\n")
    return data

def save_json(data, filename='MISSING_FILENAME.json'):
    """
    Save data to a JSON file.

    Parameters:
    -----------
    data : dict or list
        The data to be serialized into JSON format.
    filename : str, optional
        The name of the file to save the data to. Defaults to 'MISSING_FILENAME.json'.

    Returns:
    --------
    None
        Prints a confirmation message upon successful save.
    """

    file = open(filename, 'wb')
    json.dump(data, file)
    file.close()
    print(f"\nData saved to {filename}\n")

def load_json(filename='MISSING_FILENAME.json'):
    """
    Load and deserialize data from a JSON file.

    Parameters:
    -----------
    filename : str, optional
        The name of the JSON file to load. Defaults to 'MISSING_FILENAME.json'.

    Returns:
    --------
    data : dict or list
        The data deserialized from the JSON file.

    Side Effects:
    -------------
    Prints a confirmation message upon successful loading.
    """

    file = open(filename, 'rb')
    data = json.load(file)
    file.close()
    print(f"\nData loaded from {filename}\n")
    return data

def traverse_dict(data, path=[]):
    """
    Recursively traverse a nested dictionary and yield paths to each terminal value.

    Parameters:
    -----------
    data : dict
        The dictionary to traverse. Can include nested dictionaries and lists.
    path : list, optional
        The current traversal path (used internally during recursion). Defaults to an empty list.

    Yields:
    -------
    list
        A list representing the full path from the root to each terminal value or values in lists.

    Notes:
    ------
    - If a value is a nested dictionary, the function recurses deeper.
    - If a value is a list, each list item is appended to the path.
    - If a value is a scalar, it is returned as the last item in the path.
    """

    for key, value in dict(data).items():
        if isinstance(value, dict):
            yield from traverse_dict(value, path + [key])
        elif isinstance(value, list):
            yield path + [key] + value
        else:
            yield path + [key, value]

def tabulate_dict(data, columnDepth, indexDepth, valueDepth):
    """
    Converts a nested dictionary into a tabular format using specified depths for columns, 
    indices, and values.

    Args:
        data (dict): The input dictionary to be tabulated.
        columnDepth (int or list): The depth(s) in the dictionary to be used as columns.
        indexDepth (int or list): The depth(s) in the dictionary to be used as indices.
        valueDepth (int or list): The depth(s) in the dictionary to be used as values.

    Returns:
        pandas.DataFrame: A DataFrame representing the tabulated data.

    Example:
        >>> data = {
        ...     "A": {"X": {"value": 1}, "Y": {"value": 2}},
        ...     "B": {"X": {"value": 3}, "Y": {"value": 4}}
        ... }
        >>> tabulated = tabulate_dictionary(data, columnDepth=1, indexDepth=0, valueDepth=2)
        >>> print(tabulated)
    """
    
    collectedData = []
    for element in traverse_dict(data):
        if isinstance(indexDepth, int):
            index = element[indexDepth]
        elif isinstance(indexDepth, list):
            index = ', '.join([element[i] for i in indexDepth])
        else:
            raise Exception("Improper indexDepth for tabulate_dict")
        
        if isinstance(valueDepth, int):
            value = element[valueDepth]
        elif isinstance(valueDepth, tuple):
            a, b = valueDepth
            value = element[a:b]
        elif isinstance(valueDepth, list):
            value = [element[i] for i in valueDepth]
        else:
            raise Exception("Improper valueDepth for tabulate_dict")
        
        if isinstance(columnDepth, int):
            columnList = [element[columnDepth]]
        elif isinstance(columnDepth, tuple):
            a, b = columnDepth
            columnList = element[a:b]
        elif isinstance(columnDepth, list):
            columnList = [element[i] for i in columnDepth]
        else:
            raise Exception("Improper columnDepth for tabulate_dict")
        
        for column in columnList:
            collectedData.append((index, column, value))
    
    collectedData = DataFrame(collectedData, columns=['index', 'column', 'value'])
    tabulatedData = collectedData.pivot_table(index='index', columns='column', values='value', aggfunc='first')
    return tabulatedData

def serialize_dict(data, indexDepth, valueDepth):
    """
    Traverse and flatten a nested dictionary into a structured pandas DataFrame.

    Parameters:
    -----------
    data : dict
        The nested dictionary to be serialized. Can include nested dicts and lists.
    
    indexDepth : int or list of ints
        Specifies the position(s) in each traversal path to use as the row index in the resulting DataFrame.
        - If an int, selects a single element from the path.
        - If a list of ints, combines multiple elements into a single string index.
    
    valueDepth : int, list of ints, or tuple
        Specifies the position(s) in each traversal path to extract as the value for each row.
        - If an int, selects a single value.
        - If a list of ints, joins multiple values into a comma-separated string.
        - If a tuple (start, end), slices the path to return a sublist. If end is None, slices to the end.
    
    Returns:
    --------
    serializedData : pandas.DataFrame
        A DataFrame with index based on `indexDepth` and values extracted using `valueDepth`.

    Notes:
    ------
    - Uses `traverse_dictionary()` to yield all paths from the input dictionary.
    - Aggregates values based on index/value positions and reshapes them into a DataFrame.
    - The resulting DataFrame has one row per unique index and one or more columns of extracted values.
    """

    collectedIndices, collectedValues = [], []
    for element in traverse_dict(data):
        if isinstance(indexDepth, int):
            index = element[indexDepth]
        elif isinstance(indexDepth, list):
            index = ', '.join([element[i] for i in indexDepth])
        else:
            raise Exception("Improper indexDepth for serialize_dict")
        
        if isinstance(valueDepth, int):
            value = element[valueDepth]
        elif isinstance(valueDepth, tuple):
            a, b = valueDepth
            value = element[a:b]
        elif isinstance(valueDepth, list):
            value = [element[i] for i in valueDepth]
        else:
            raise Exception("Improper valueDepth for tabulate_dict")
        
        collectedIndices.append(index)
        collectedValues.append(value)
    
    serializedData = Series(collectedValues, index=collectedIndices)
    return serializedData

def recolumnate_df(data, columnation):
    '''
    # The below is chat's vectorization of this function
    
    recolumnatedData = columnation.copy()
    for index in recolumnatedData.index:
        for column in recolumnatedData.columns:
            oldColumn = columnation.loc[index, column]
            if not isinstance(oldColumn, str) and not isnan(oldColumn):
                value = data.loc[index, oldColumn]
                recolumnatedData.loc[index, column] = value
    return recolumnatedData
    '''
    data, columnation = DataFrame(data), DataFrame(columnation)
    long = columnation.stack().rename("old_column").reset_index()
    long.columns = ["index", "column", "old_column"]
    long = long[long["old_column"].apply(lambda x: isinstance(x, str))]

    data_long = data.stack().rename("value").reset_index()
    data_long.columns = ["index", "old_column", "value"]
    
    merged = long.merge(data_long, on=["index", "old_column"], how="left")

    recolumnated = merged.pivot(index="index", columns="column", values="value")
    
    recolumnated.index.name = columnation.index.name
    recolumnated.columns.name = None

    return recolumnated

def bound_data(data, start, end):
    '''
    Restrict a DataFrame to a specified range of integer index values.
    Also, sort by string casted columns and integer casted indices.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame whose index will be filtered and sorted.
    
    start : int
        The starting index value (inclusive) of the desired range.
    
    end : int
        The ending index value (inclusive) of the desired range.

    Returns:
    --------
    pandas.DataFrame
        A new DataFrame containing only the rows within the specified index range, sorted by index.

    Notes:
    ------
    - Converts the index to integers before filtering.
    - Sorts the DataFrame by its index prior to slicing.
    '''
    if isinstance(data, DataFrame):
        data.index = data.index.astype(int)
        data.columns = data.columns.astype(str)
        data = data.sort_index().loc[start:end]
        data = data[sorted(data.columns)]
        return data
    elif isinstance(data, Series):
        data.index = data.index.astype(int)
        data = data.sort_index().loc[start:end]
        return data
    else:
        raise Exception("improper data provided to bound_data")

