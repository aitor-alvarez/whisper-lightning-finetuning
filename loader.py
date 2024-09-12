from datasets import load_dataset

def load_data(dir):
    speech_train = load_dataset("audiofolder", data_dir=dir, split="train")
    speech_test = load_dataset("audiofolder", data_dir=dir, split="test")
