from datasets import load_dataset
import json, os


def extract_characters(batch):
    txt = " ".join(batch["sentence"])
    vocab = list(set(txt))
    dict_out = {'vocab': [vocab], 'txt': [txt]}
    return dict_out

def extract_vocab(train, test):
    vocab_train = train.map(extract_characters, batched=True, batch_size=-1,
                                   remove_columns=train.column_names)
    vocab_test = test.map(extract_characters, batched=True, batch_size=-1,
                                 remove_columns=test.column_names)

    #vocab = list(set(vocab_train["vocab"]) | set(vocab_test["vocab"]))
    vocab_dict = {v: k for k, v in enumerate(sorted(vocab_train))}
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    with open('vocab.json', 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)

def load_data(dir=None, dataset=None, data_lang=None, is_local=False):
    if is_local:
        train = load_dataset("audiofolder", data_dir=dir, split="train")
        test = load_dataset("audiofolder", data_dir=dir, split="test")
    else:
        train = load_dataset(dataset, data_lang, split="train", streaming=True)
        test = load_dataset(dataset, data_lang, split="test", streaming=True)

    if not os.path.exists('vocab.json'):
        extract_vocab(train, test)

    return train, test