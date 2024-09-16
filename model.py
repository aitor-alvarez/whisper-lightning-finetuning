import torch
import lightning as L
from torch import nn
from transformers import WhisperForConditionalGeneration, get_linear_schedule_with_warmup, WhisperTokenizer
from evaluate import load
from data import SpeechDataModule

class WhisperLightning(L.LightningModule):
    def __init__(self, model_name: str):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.warmup_steps = 400
        self.weight_decay = 0.00
        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name)
        self.model.config.apply_spec_augment = True
        self.model.config.mask_feature_prob = 0.05
        self.model.config.forced_decoder_ids = None
        self.model.config.suppress_tokens = []
        self.tokenizer = WhisperTokenizer.from_pretrained(model_name, language='es', task="transcribe")
        self.wer = load("wer")

    def metrics(self, predicted, labels):
        labels[labels == -100] = self.tokenizer.pad_token_id
        preds_str = self.tokenizer.batch_decode(predicted, skip_special_tokens=True)
        label_str = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        wer = self.wer.compute(predictions=preds_str, references=label_str)

        return wer

    def step(self, batch):
        x = batch['input_features']
        y = batch['labels']
        decoder_input_ids = batch['decoder_input_ids']
        output = self.model(x, decoder_input_ids=decoder_input_ids, labels=y)
        self.log(f"loss", output.loss, prog_bar=True, sync_dist=True)
        return output.loss


    def training_step(self, batch):
        error = self.step(batch)
        scheduler = self.lr_schedulers()
        self.log("lr", scheduler.get_last_lr()[0], prog_bar=True)
        return error

    def test_step(self, batch):
        return self.step(batch)

    def optimizer_step(self, epoch, batch, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure=optimizer_closure)
        # update learning rate
        self.lr_schedulers().step()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        return [optimizer], [scheduler]