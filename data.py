from torch import Tensor
import lightning as L
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from loader import load_data
from torch.utils.data import DataLoader
from transformers import WhisperProcessor
from datasets import Audio
from typing import Optional
from utils import DataCollatorSpeechSeq2SeqWithPadding

class SpeechDataModule(L.LightningDataModule):
    def __init__(self, model_name: str, batch_size: int, dir: Optional[str]=None, dataset: Optional[str]=None,
                 data_lang: Optional[str] =None, is_local: bool=True):
        super().__init__()
        self.train = None
        self.test = None
        self.batch_size = batch_size
        self.dir = dir
        self.dataset = dataset
        self.lang = data_lang
        self.is_local = is_local
        self.processor = WhisperProcessor.from_pretrained(model_name, language=data_lang, task="transcribe")
        self.data_collator = DataCollatorSpeechSeq2SeqWithPadding


    def prepare_dataset(self, batch):
        audio = batch["audio"]
        batch["input_values"] = self.processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["input_length"] = len(batch["input_values"])
        batch["labels"] = self.processor(batch["sentence"]).input_ids
        return batch

    def setup(self, stage: str =None):
        if self.is_local:
            self.train, self.test = load_data(dir=self.dir)
        elif self.is_local is False:
            self.train, self.test = load_data(dataset=self.dataset, data_lang=self.lang)
        self.train = self.train.cast_column("audio", Audio(sampling_rate=16_000))
        self.test = self.test.cast_column("audio", Audio(sampling_rate=16_000))
        self.train = self.train.map(self.prepare_dataset, remove_columns=self.train.column_names)
        self.test = self.test.map(self.prepare_dataset, remove_columns=self.test.column_names)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train, batch_size=self.batch_size, collate_fn=self.data_collator, num_workers=2, persistent_workers=True)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test, batch_size=self.batch_size, collate_fn=self.data_collator,  num_workers=2, persistent_workers=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.validation, batch_size=self.batch_size, collate_fn=self.data_collator, num_workers=2, persistent_workers=True)