## Whisper Finetuning

This is a Pytorch Lightning repo to fine-tune the Whisper model.

The data module is designed to process common voice dataset or a locally stored dataset.

To train the model just execute the following command:

```
python main.py --model_name 'openai/whisper-large-v3' --batch_size 16 --num_epochs 100 --dataset 'mozilla-foundation/common_voice_16_0'
               --lang 'es' --n_gpus 4 --n_nodes 1
```
