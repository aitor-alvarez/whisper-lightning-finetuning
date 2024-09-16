import torch
import lightning as L
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from loader import load_data
from torch.utils.data import DataLoader
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration
from datasets import Audio
from typing import Optional, List, Union, Dict

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
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.tokenizer = WhisperTokenizer.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)


    def prepare_dataset(self, batch):
        audio = batch["audio"]
        batch["input_features"] = self.processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["labels"] = self.tokenizer(batch["sentence"]).input_ids
        return batch


    def data_collator(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"]
        decoder_input_ids = labels[:, :-1]
        labels = labels[:, 1:]
        labels_mask = labels_batch.attention_mask[:, 1:]

        # replace padding with -100 to ignore correctly when computing the loss
        labels = labels.masked_fill(labels_mask.ne(1), -100)

        # replace initial prompt tokens with -100 to ignore correctly when computing the loss
        bos_index = torch.argmax((labels == self.model.config.decoder_start_token_id).long(), dim=1)
        prompt_mask = torch.arange(labels.shape[1]) < bos_index[:, None]
        labels = torch.where(prompt_mask, -100, labels)

        batch["labels"] = labels
        batch["decoder_input_ids"] = decoder_input_ids

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