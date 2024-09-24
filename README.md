## Whisper Finetuning

This is a Pytorch Lightning repo to fine-tune the Whisper model.

The data module is designed to process common voice dataset or a locally stored dataset.

To train the model just execute the following command (for 4 GPUs on a single node):

```
python main.py --model_name 'openai/whisper-large-v3' --batch_size 4 --num_epochs 100 --dataset 'mozilla-foundation/common_voice_16_0'
               --lang 'es' --n_gpus 4 --n_nodes 1 --strategy="ddp"
```

For locally hosted dataset use the following command:

```
python main.py --model_name 'openai/whisper-large-v3' --batch_size 4 --num_epochs 100 --data_folder 'path_to_dataset'
               --lang 'es' --n_gpus 4 --n_nodes 1 --strategy="ddp"
```
