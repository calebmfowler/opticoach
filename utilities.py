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