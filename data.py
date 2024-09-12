from torch import Tensor
import lightning as L
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from loader import load_data
from torch.utils.data import DataLoader
from transformers import WhisperProcessor


class SpeechDataModule(L.LightningDataModule):
    def __init__(self, model_name: str, batch_size: int, dir: str=None, dataset: str=None,
                 data_lang:str =None, is_local: bool=True):
        super().__init__()
        self.batch_size = batch_size
        self.dir = dir
        self.dataset = dataset
        self.lang = data_lang
        self.is_local = is_local
        self.processor = WhisperProcessor.from_pretrained(model_name)

    def collate_pad(self, batch):
        (x, sr, y) = zip(*batch)
        x = self.processor.feature_extractor(
            x, sampling_rate=sr[0], return_tensors="pt"
        ).input_features
        y = Tensor(list(y))
        return x, sr[0], y

    def setup(self, stage: str =None):
        if self.is_local:
            self.train, self.validation, self.test = load_data(dir=self.dir)
        elif self.is_local is False:
            self.train, self.validation, self.test = load_data(dataset=self.dataset, data_lang=self.lang)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train, batch_size=self.batch_size, collate_fn=self.collate_pad, num_workers=4)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test, batch_size=self.batch_size, collate_fn=self.collate_pad, num_workers=4)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.validation, batch_size=self.batch_size, collate_fn=self.collate_pad, num_workers=4)