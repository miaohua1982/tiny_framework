
def save_model(model, filename):
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_model(filename):
    import pickle
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

def all(tensor):
    return tensor.flatten().sum().item() == tensor.el_num()