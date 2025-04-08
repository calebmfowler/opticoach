import pickle as pkl

def save_pkl(data, filename='MISSING_FILENAME.pkl'):
    file = open(filename, 'wb')
    pkl.dump(data, file)
    file.close()
    print(f"\nData saved to {filename}")

def load_pkl(filename='MISSING_FILENAME.pkl'):
    file = open(filename, 'rb')
    data = pkl.load(file)
    file.close()
    print(f"\nData loaded from {filename}")
    return data

