import pickle


def pickle_save(data, filename):
    with open('{}.pickle'.format(filename), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(filename):
    with open('{}.pickle'.format(filename), 'rb') as handle:
        return pickle.load(handle)