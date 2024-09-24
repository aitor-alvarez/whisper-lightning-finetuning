from argparse import ArgumentParser
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from model import WhisperLightning
from data import SpeechDataModule
from torch import cuda, autograd


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--data_folder', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--lang', type=str)
    parser.add_argument('--n_gpus', type=int)
    parser.add_argument('--n_nodes', type=int)
    parser.add_argument('--strategy', type=str)
    args = parser.parse_args()

    #For possible errors in the backward pass
    autograd.set_detect_anomaly(True)

    model = WhisperLightning(args.model_name)

    logger = WandbLogger(
        project="finetuning_whisper",
        log_model=True,
        save_dir="./wandb",
    )

    if args.n_gpus and args.n_nodes:
        trainer = Trainer(max_epochs=args.num_epochs, logger = logger, accelerator='cuda', accumulate_grad_batches=2,
                      strategy=args.strategy, devices=args.n_gpus, num_nodes=args.n_nodes, precision=16)
    else:
        trainer = Trainer(max_epochs=args.num_epochs, logger=logger, accumulate_grad_batches=2,
                          accelerator='cpu', devices="auto")
    if args.dataset:
        data = SpeechDataModule(model_name=args.model_name, batch_size=args.batch_size,
                                dataset=args.dataset, data_lang=args.lang, is_local=False)
    elif args.data_folder:
        data = SpeechDataModule(model_name=args.model_name, batch_size=args.batch_size,
                                dir=args.data_folder)
    else:
        data = None
        print("Check the parameters needed to train the model. There is no dataset path or name provided.")

    if data is not None:
        data.setup()
        trainer.fit(model, datamodule=data)
        trainer.print(cuda.memory_summary())
        trainer.save_checkpoint('./checkpoints')
        trainer.test(model, datamodule=data)

        print("training is completed")