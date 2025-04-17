from pandas import DataFrame
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
        else:
            yield path + [key, value]

def tabulate_dictionary(data, columnDepth, indexDepth, valueDepth):
    
    retabulatedData = DataFrame()

    for element in traverse_dictionary(data):
        index = element[indexDepth]
        value = element[valueDepth]
        column = element[columnDepth]
        retabulatedData.loc[index, column] = value
    
    return retabulatedData