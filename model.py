import torch
import lightning as L
from torch import nn
from transformers import WhisperForConditionalGeneration, get_linear_schedule_with_warmup

class WhisperLightning(L.LightningModule):
    def __init__(self, model_name: str, batch_size: int, epochs: int):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.batch_size = batch_size
        self.epochs = epochs
        self.warmup_steps = 400
        self.weight_decay = 0.00
        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name)

    def step(self, batch):
        x, yi, yo = batch
        logits = self.model.generate(x, output_logits=True)
        loss = nn.functional.cross_entropy(logits.transpose(1, 2), yo)
        self.log(f"loss", loss, prog_bar=True, sync_dist=True)


    def training_step(self, batch):
        loss = self.step(batch)
        scheduler = self.lr_schedulers()
        self.log("lr", scheduler.get_last_lr()[0], prog_bar=True)
        return loss

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
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
        return [optimizer] [scheduler]