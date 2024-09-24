import torch
import lightning as L
from transformers import (WhisperForConditionalGeneration, get_linear_schedule_with_warmup,
                          WhisperTokenizer, WhisperProcessor)
from torchmetrics.text import WordErrorRate


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
        self.model.freeze_feature_encoder = True
        self.tokenizer = WhisperTokenizer.from_pretrained(model_name, language='es', task="transcribe")
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.wer = WordErrorRate()


    def compute_metrics(self, predicted, labels):
        pred_str = self.processor.batch_decode(predicted, skip_special_tokens=True)
        labels[labels == -100] = self.tokenizer.pad_token_id
        label_str = self.processor.batch_decode(labels, skip_special_tokens=True)
        wer = 100*self.wer(pred_str, label_str)
        return wer



    def training_step(self, batch, name='train'):
        x = batch['input_features']
        y = batch['labels']
        decoder_input_ids = batch['decoder_input_ids']
        output = self.model(x, decoder_input_ids=decoder_input_ids, labels=y)
        self.log(f"{name} loss", output.loss, prog_bar=True, sync_dist=True)
        scheduler = self.lr_schedulers()
        self.log("lr", scheduler.get_last_lr()[0], prog_bar=True)
        return output.loss

    def test_step(self, batch, name='test'):
        x = batch['input_features']
        y = batch['labels']
        decoder_input_ids = batch['decoder_input_ids']
        output = self.model(x, decoder_input_ids=decoder_input_ids, labels=y)
        pred_ids = torch.argmax(output.logits, dim=1)
        wer = self.compute_metrics(pred_ids, y)
        print(wer.item())
        return output.loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        return [optimizer], [scheduler]