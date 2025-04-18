from collections import defaultdict
from collections.abc import Iterable
from pandas import DataFrame, Series
import pickle as pkl
import json

def save_pkl(data, filename='MISSING_FILENAME.pkl'):
    file = open(filename, 'wb')
    pkl.dump(data, file)
    file.close()
    print(f"Data saved to {filename}")

def load_pkl(filename='MISSING_FILENAME.pkl'):
    file = open(filename, 'rb')
    data = pkl.load(file)
    file.close()
    print(f"Data loaded from {filename}")
    return data

def save_json(data, filename='MISSING_FILENAME.json'):
    file = open(filename, 'wb')
    json.dump(data, file)
    file.close()
    print(f"Data saved to {filename}")

def load_json(filename='MISSING_FILENAME.json'):
    file = open(filename, 'rb')
    data = json.load(file)
    file.close()
    print(f"Data loaded from {filename}")
    return data

def traverse_dictionary(data, path=[]):
    for key, value in dict(data).items():
        if isinstance(value, dict):
            yield from traverse_dictionary(value, path + [key])
        elif isinstance(value, list):
            yield path + [key] + value
        else:
            yield path + [key, value]


def tabulate_dictionary(data, columnDepth, indexDepth, valueDepth):
    collectedData = []
    for element in traverse_dictionary(data):
        if isinstance(indexDepth, list):
            index = ', '.join([element[i] for i in indexDepth])
        else:
            index = element[indexDepth]
        if isinstance(valueDepth, list):
            value = ', '.join([element[i] for i in valueDepth])
        elif isinstance(valueDepth, tuple):
            a, b = valueDepth
            if b == None:
                value = element[a:]
            else:
                value = element[a:b]
        else:
            value = element[valueDepth]
        if isinstance(columnDepth, list):
            column = ', '.join([element[i] for i in columnDepth])
        else:
            column = element[columnDepth]
        collectedData.append((index, column, value))
    collectedDictionary = defaultdict(dict)
    for index, column, value in collectedData:
        collectedDictionary[index][column] = value
    tabulatedData = DataFrame.from_dict(collectedDictionary, orient='index')
    return tabulatedData

def serialize_dictionary(data, indexDepth, valueDepth):
    collectedData = []
    for element in traverse_dictionary(data):
        if isinstance(indexDepth, list):
            index = ', '.join([element[i] for i in indexDepth])
        else:
            index = element[indexDepth]
        if isinstance(valueDepth, list):
            value = ', '.join([element[i] for i in valueDepth])
        elif isinstance(valueDepth, tuple):
            a, b = valueDepth
            if b == None:
                value = element[a:]
            else:
                value = element[a:b]
        else:
            value = element[valueDepth]
        collectedData.append((index, value))
    collectedDictionary = defaultdict(dict)
    for index, value in collectedData:
        collectedDictionary[index] = value
    serializedData = DataFrame.from_dict(collectedDictionary, orient='index')
    return serializedData

def bound(df, start, end):
    df.index = df.index.astype(int)
    df = df.sort_index().loc[start:end]
    return df