import os
import torch
import torchaudio
import numpy as np
import random
from argparse import Namespace
import json
from data.tokenizer import AudioTokenizer, TextTokenizer
from huggingface_hub import hf_hub_download
from inference_tts_scale import inference_one_sample_graphemes
from shutil import copy2
import gradio as gr

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def process_item(model, config, phn2num, audio_tokenizer, prompt_audio, prompt_text, input_text, codec_audio_sr, top_k, top_p, temperature, stop_repetition, kvcache, silence_tokens, sample_batch_size, seed):
    seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    orig_audio = prompt_audio

    orig_transcript = prompt_text
    sentence = input_text

    words = sentence.split(" ")
    chunks = []
    N_WORDS = 20 
    for ix in range(0, len(words), N_WORDS):
        chunks.append(" ".join(words[ix: ix+N_WORDS]))
    chunks = list(filter(lambda x: x != "", chunks))

    info = torchaudio.info(orig_audio)
    audio_dur = info.num_frames / info.sample_rate
    cut_off_sec = audio_dur - 0.01

    assert cut_off_sec < audio_dur, f"cut_off_sec {cut_off_sec} is larger than the audio duration {audio_dur}"
    prompt_end_frame = int(cut_off_sec * info.sample_rate)

    decode_config = {
        'top_k': top_k, 'top_p': top_p, 'temperature': temperature, 
        'stop_repetition': stop_repetition, 'kvcache': kvcache, "codec_audio_sr": codec_audio_sr, 
        "codec_sr": 50, "silence_tokens": silence_tokens, "sample_batch_size": sample_batch_size
    }
    prompt = orig_transcript
    audio_chunks = []
    for ix, chunk in enumerate(chunks):
        chunk_i = prompt + ' ' + chunk
        concated_audio, gen_audio = inference_one_sample_graphemes(
            model, Namespace(**config), phn2num, audio_tokenizer, orig_audio, 
            chunk_i, device, decode_config, prompt_end_frame
        )
        audio_chunks.append(gen_audio.squeeze(0).cpu())

    gen_audio = torch.cat(audio_chunks, dim=1)

    output_file = "output.flac"
    torchaudio.save(output_file, gen_audio, codec_audio_sr, format="flac")
    return output_file

def load_model(model_path, encodec_fn):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path '{model_path}' does not exist. Please check the path and try again.")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    voicecraft_name = "830M_TTSEnhanced.pth"
    from models import voicecraft

    ckpt = torch.load(model_path, map_location='cpu')
    model = voicecraft.VoiceCraft(ckpt['config'])
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()

    config = vars(model.args)
    phn2num = ckpt["phn2num"]

    if not os.path.exists(encodec_fn):
        os.system(f"wget https://huggingface.co/pyp1/VoiceCraft/resolve/main/encodec_4cb2048_giga.th")
        os.system(f"mv encodec_4cb2048_giga.th {encodec_fn}")
    audio_tokenizer = AudioTokenizer(signature=encodec_fn, device=device)
    
    return model, config, phn2num, audio_tokenizer

model_path = '/home/model/ttsteam/repos/voicecraft/h100_checkpoints/tts-asr/best_bundle.pth'
encodec_fn = "./pretrained_models/encodec_4cb2048_giga.th"
model, config, phn2num, audio_tokenizer = load_model(model_path, encodec_fn)

interface = gr.Interface(
    fn=lambda prompt_audio, prompt_text, input_text: process_item(
        model, config, phn2num, audio_tokenizer, prompt_audio, prompt_text, input_text,
        codec_audio_sr=16000, top_k=0, top_p=0.9, temperature=1.0, stop_repetition=5,
        kvcache=1, silence_tokens=[1388, 1898, 131], sample_batch_size=3, seed=1
    ),
    inputs=[
        gr.Audio(source="upload", type="filepath", label="Prompt Audio"),
        gr.Textbox(lines=2, placeholder="Enter the prompt text here...", label="Prompt Text"),
        gr.Textbox(lines=2, placeholder="Enter the text for which speech should be generated...", label="Input Text")
    ],
    outputs=gr.Audio(label="Generated Speech"),
    title="VoiceCraft Model",
    description="Provide a prompt audio and text, and input text to generate speech."
)

if __name__ == "__main__":
    interface.launch()
