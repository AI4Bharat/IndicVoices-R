# IndicVoices-R: Unlocking a Massive Multilingual Multi-speaker Speech Corpus for Scaling Indian TTS

### Abstract
Recent advancements in text-to-speech (TTS) synthesis show that large-scale models trained with extensive web data produce highly natural-sounding output. However, such data is scarce for Indian languages due to the lack of high-quality, manually subtitled data on platforms like LibriVox or YouTube. To address this gap, we enhance existing large-scale ASR datasets containing natural conversations collected in low-quality environments to generate high-quality TTS training data. Our pipeline leverages the cross-lingual generalization of denoising and speech enhancement models trained on English and applied to Indian languages. This results in IndicVoices-R (IV-R), the largest multilingual Indian TTS dataset derived from an ASR dataset, with 1,704 hours of high-quality speech from 10,496 speakers across 22 Indian languages. IV-R matches the quality of gold-standard TTS datasets like LJSpeech, LibriTTS, and IndicTTS. We also introduce the IV-R Benchmark, the first to assess zero-shot, few-shot, and many-shot speaker generalization capabilities of TTS models on Indian voices, ensuring diversity in age, gender, and style. We demonstrate that fine-tuning an English pre-trained model on a combined dataset of high-quality IndicTTS and our IV-R dataset results in better zero-shot speaker generalization compared to fine-tuning on the IndicTTS dataset alone. Further, our evaluation reveals limited zero-shot generalization for Indian voices in TTS models trained on prior datasets, which we improve by fine-tuning the model on our data containing diverse set of speakers across language families. We open-source all data and code, releasing the first TTS model for all 22 official Indian languages.


### Resources

Download the data [here](https://ai4bharat.iitm.ac.in/indicvoices_r/)

### Manifest Format

````
    "filename": "<AUDIOS/audios>/2533274790514854_chunk_4.wav",                          # Points to the wav file
    "text": "<TRANSCRIPT>",                   # Transcript for audio, we use Normalized version of the transcript
    "duration": <DURATION>,                                                          #  Audio duration in seconds
    "lang": "<LANG_CODE(ISO)>",                                      # ISO code for language (given in meta data)
    "samples": <NUMBER_OF_SAMPLES>,                                                           # Number of samples
    "verbatim": "<VERBATIM VERSION OF TRANSCRIPT>",                          # Verbatim version of the transcript
    "normalized": "<NORMALIZE>",                                           # Normalized version of the transcript
    "speaker_id": "S4258780200341914",                                                        # Unique speaker ID
    "scenario": "Extempore",                                                                       # Type of data
    "task_name": "KYP - Traveling",                                                                   # Task name
    "gender": "Male",                                                                     # Gender of the speaker
    "age_group": "18-30",                                                              # Age group of the speaker
    "job_type": "Student",                                                              # Job type of the speaker
    "qualification": "Undergrad and Grad.",                                        # Qualification of the speaker
    "area": "Rural",                                                        # Area from which the speaker belongs
    "district": "Barpeta",                                              # District from which the speaker belongs
    "state": "Assam",                                                      # State from which the speaker belongs
    "occupation": "Private tutor",                                                         # Speaker's occupation
    "verification_report": "{}",                                   # Verification markers as given by the QA team
    "chunk_name": "2533274790514854_chunk_4.wav",                                              # Audio chunk name
    "snr": xx.xx,
    "c50": xx.xx,
    "utterance_pitch_mean": xx.xx,
    "utterance_pitch_std": xx.xx,
    "cer": 0.xx,
````

### LICENSE

[CC-BY-4.0](/LICENSE.md)
