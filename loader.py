from datasets import load_dataset

def load_data(dir=None, dataset=None, data_lang=None, is_local=False):
    if is_local:
        train = load_dataset("audiofolder", data_dir=dir, split="train")
        test = load_dataset("audiofolder", data_dir=dir, split="test")
        validation = None
    else:
        train = load_dataset(dataset, data_lang, split="train")
        validation = load_dataset(dataset, data_lang, split="validation")
        test = load_dataset(dataset, data_lang, split="test")
    return train, validation, test