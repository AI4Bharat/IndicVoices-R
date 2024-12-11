# IndicVoices-R: Unlocking a Massive Multilingual Multi-speaker Speech Corpus for Scaling Indian TTS
> 🎉 Accepted at NeurIPS 2024  (Datasets and Benchmark Track)

[![DOI](https://zenodo.org/badge/813636000.svg)](https://zenodo.org/doi/10.5281/zenodo.11636050)  [![Paper](https://img.shields.io/badge/arXiv-2409.05356-brightgreen.svg?style=flat-square)](https://arxiv.org/pdf/2409.05356)

We present IndicVoices-R, an ASR enhanced TTS dataset for the 22 official Indian languages, with over 1700 hours of high-quality speech in the voice of more than 10k speakers. Follow the instructions given below to download and access the dataset.

We train the Voicecraft model, as implemented [here](https://github.com/jasonppy/voicecraft/) to demonstrate the prowess of our dataset for Text-to-Speech.

### Abstract
Recent advancements in text-to-speech (TTS) synthesis show that large-scale models trained with extensive web data produce highly natural-sounding output. However, such data is scarce for Indian languages due to the lack of high-quality, manually subtitled data on platforms like LibriVox or YouTube. To address this gap, we enhance existing large-scale ASR datasets containing natural conversations collected in low-quality environments to generate high-quality TTS training data. Our pipeline leverages the cross-lingual generalization of denoising and speech enhancement models trained on English and applied to Indian languages. This results in IndicVoices-R (IV-R), the largest multilingual Indian TTS dataset derived from an ASR dataset, with 1,704 hours of high-quality speech from 10,496 speakers across 22 Indian languages. IV-R matches the quality of gold-standard TTS datasets like LJSpeech, LibriTTS, and IndicTTS. We also introduce the IV-R Benchmark, the first to assess zero-shot, few-shot, and many-shot speaker generalization capabilities of TTS models on Indian voices, ensuring diversity in age, gender, and style. We demonstrate that fine-tuning an English pre-trained model on a combined dataset of high-quality IndicTTS and our IV-R dataset results in better zero-shot speaker generalization compared to fine-tuning on the IndicTTS dataset alone. Further, our evaluation reveals limited zero-shot generalization for Indian voices in TTS models trained on prior datasets, which we improve by fine-tuning the model on our data containing diverse set of speakers across language families. We open-source all data and code necessary to replicate the first TTS model for all 22 official Indian languages.

### Manifest for Transcript and Metadata

We include the fields listed below in the manifest file for each language, accompanying the audios.

| Field                | Description                                                     |
|----------------------|-----------------------------------------------------------------|
| `filename`           | Points to the wav file                                          |
| `text`               | Transcript for audio (normalized version)                       |
| `duration`           | Audio duration in seconds                                       |
| `lang`               | ISO code for language (given in metadata)                       |
| `samples`            | Number of samples                                               |
| `verbatim`           | Verbatim version of the transcript                              |
| `normalized`         | Normalized version of the transcript (same as text)             |
| `speaker_id`         | Unique speaker ID                                               |
| `scenario`           | Type of data                                                    |
| `task_name`          | Task name                                                       |
| `gender`             | Gender of the speaker                                           |
| `age_group`          | Age group of the speaker                                        |
| `job_type`           | Job type of the speaker                                         |
| `qualification`      | Qualification of the speaker                                    |
| `area`               | Area from which the speaker belongs                             |
| `district`           | District from which the speaker belongs                         |
| `state`              | State from which the speaker belongs                            |
| `occupation`         | Speaker's occupation                                            |
| `verification_report`| Verification markers as given by the QA team                    |
| `chunk_name`         | Audio chunk name                                                |
| `snr`                | Signal-to-noise ratio                                           |
| `c50`                | Clarity index (C50)                                             |
| `utterance_pitch_mean`| Mean pitch of the utterance                                    |
| `utterance_pitch_std` | Standard deviation of the utterance pitch                      |
| `cer`                | Character error rate                                            |


### Setup

#### Download the data
IndicVoices-R is available as tar files [here](https://ai4bharat.iitm.ac.in/datasets/IndicVoices-R)
To download and untar the dataset for a single language, use wget as follows:

```bash
wget -O <save as filename> <url to tar file> | tar -xz
```

Example
```bash
wget -O Assamese.tar https://indic-tts-public.objectstore.e2enetworks.net/data/ivr/Assamese.tar.gz | tar -xz
```

To download data for multiple languages, refer the `data_links.txt` file and run the following bash script

```bash
echo "Starting download and extraction process..."
while IFS= read -r url; do
    wget "$url" -O - | tar -xf -
done < data_links.txt
echo "Process completed."
```
or, download simultaneously

```bash
cat data_links.txt | xargs -n 1 -P 4 -I {} sh -c 'echo "Downloading and extracting: {}"; wget "{}" -O - | tar -xf -; echo "Completed: {}"'
```

Here `-n 1` processes one url at a time and `-P 4` runs 4 downloads in parallel - adjust according to your CPU resources.

#### Fine-tuning VoiceCraft
Please refer to ![VoiceCraft/README.md](VoiceCraft/README.md) for more details.

### Citation
If you used this repository or the dataset, please cite our work. Thank you :)

```bibtex
@article{ai4bharat2024indicvoicesr,
  title={IndicVoices-R: Unlocking a Massive Multilingual Multi-speaker Speech Corpus for Scaling Indian TTS},
  author={Sankar, Ashwin and Anand, Srija and Varadhan, Praveen Srinivasa and Thomas, Sherry and Singal, Mehak and Kumar, Shridhar and Mehendale, Deovrat and Krishana, Aditi and Raju, Giri and Khapra, Mitesh},
  journal={NeurIPS 2024 Datasets and Benchmarks},
  year={2024}
}
