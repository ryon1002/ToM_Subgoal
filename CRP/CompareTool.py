import pickle

def checkStoreResult(data, prev_data_file, check, dump):
    if check:
        prev_data = pickle.load(open(prev_data_file, "r"))
        if data != prev_data:
            print False
    if dump:
        pickle.dump(data, open(prev_data_file, "w"))
    return data