# IN22VoiceCraft

IN22VoiceCraft aims at supporting the fine-tuning of the English VoiceCraft checkpoint on the 22 official languages of India starting from the original [VoiceCraft](https://github.com/jasonppy/VoiceCraft) repo.


## Installation

Follow the installation instructions in the original [VoiceCraft repository](https://github.com/jasonppy/VoiceCraft). 
Ensure all dependencies are properly installed and the environment is set up for training.
Download the [pre-trained English checkpoint](https://huggingface.co/pyp1/VoiceCraft/resolve/main/830M_TTSEnhanced.pth?download=true) and the [EnCodec checkpoint](https://huggingface.co/pyp1/VoiceCraft/resolve/main/encodec_4cb2048_giga.th?download=true).


## Steps for setting up training

##### 1. Make a JSONL manifest. 
Each line in the JSONL file should represent a data record in JSON format. Below is an example:
```json
{"audio_filepath": "/path/to/audio/hindi_sample_001.wav", "text": "यह एक उदाहरण वाक्य है", "duration": 5.123456, "dataset": "dummy_dataset", "segment_id": "hindi_sample_001", "begin_time": 0.0, "end_time": 5.123456}
```

**Note:** Check durations of each audio and filter out null duration audios from the manifest
You may wish to split the manifest into appropriate train-valid-test sets. You should have the following manifests - 
```
datasets/ivr/metadata_train.json
datasets/ivr/metadata_valid.json
datasets/ivr/metadata_test.json
```

##### 2. Extract Graphemes and Codes

Run the script to generate phonemes and codes from the manifest:

```bash
CUDA_VISIBLE_DEVICES=0 python ./data/make_phonemes_and_codes_from_manifest.py \
    --save_dir datasets/ivr \
    --encodec_model_path checkpoints/encodec_4cb2048_giga.th \
    --mega_batch_size 128 \
    --batch_size 128 \
    --max_len 30000 \
    --n_workers 64 \
    --n_splits 8 \
    --split_idx 0
```

Parameter Explanation:

`--n_splits:` Specifies the total number of splits for processing, typically used when leveraging multiple GPUs. For example, if you have 8 GPUs, set this to 8.

`--split_idx:` Specifies the index of the current split to process. Each GPU should handle one split, with split_idx ranging from 0 to n_splits-1.


##### 3. Generate Training manifests
Create the required manifests for training or fine-tuning:
```
python datasets/ivr/make_manifest.py
```

##### 4. Start the Fine-tuning Script
Run the fine-tuning script to adapt the model to your dataset:
```bash
./our_scripts/finetune.sh
```

For inference please refer to ![inference.py](inference.py)

### Citation
If you used this repository, please cite the following works. Thank you :)

```bibtex
@article{ai4bharat2024indicvoicesr,
  title={IndicVoices-R: Unlocking a Massive Multilingual Multi-speaker Speech Corpus for Scaling Indian TTS},
  author={Sankar, Ashwin and Anand, Srija and Varadhan, Praveen Srinivasa and Thomas, Sherry and Singal, Mehak and Kumar, Shridhar and Mehendale, Deovrat and Krishana, Aditi and Raju, Giri and Khapra, Mitesh},
  journal={NeurIPS 2024 Datasets and Benchmarks},
  year={2024}
}

@article{peng2024voicecraft,
  author    = {Peng, Puyuan and Huang, Po-Yao and Mohamed, Abdelrahman and Harwath, David},
  title     = {VoiceCraft: Zero-Shot Speech Editing and Text-to-Speech in the Wild},
  journal   = {arXiv},
  year      = {2024},
}
```

### License
This codebase is under CC BY-NC-SA 4.0 (LICENSE-CODE), following the [license here](https://github.com/jasonppy/VoiceCraft/blob/master/LICENSE-CODE).

### Acknowledgements
We thank the authors of [VoiceCraft](https://github.com/jasonppy/VoiceCraft) for open-sourcing their work. 
