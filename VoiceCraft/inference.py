import os
import torch
import torchaudio
import numpy as np
import random
from argparse import Namespace, ArgumentParser
from tqdm import tqdm
import json
import concurrent
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from data.tokenizer import AudioTokenizer, TextTokenizer
from huggingface_hub import hf_hub_download
from inference_tts_scale import inference_one_sample_graphemes
from shutil import copy2

def read_jsonl(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

indoaryan = [
    'Assamese', 'Gujarati', 'Hindi', 
    'Kashmiri', 'Konkani', 'Maithili', 'Marathi', 
    'Nepali', 'Odia', 'Punjabi', 'Sanskrit', 'Santali', 'Sindhi', "Urdu" 
]
sino_tibetan = ['Bodo', 'Manipuri']
dravidian = ['Kannada', 'Malayalam', 'Tamil', 'Telugu']
fam2langs = {
    'dravidian': dravidian,
    'indoaryan': indoaryan, 
    'sino_tibetan': sino_tibetan,
    "all": dravidian + indoaryan
}

def get_spk2item(items):
    spk2item = {}
    for item in tqdm(items, desc='making spk2item'):
        spk_id = item['speaker_id']
        if spk_id not in spk2item.keys():
            spk2item[spk_id] = [item]
        else:
            spk2item[spk_id].append(item)
    return spk2item

def contains_english_characters(sentence):
    # Loop through each character in the sentence
    for char in sentence:
        # Check if the character is an English letter
        if char.isalpha() and char.isascii():
            return True
    return False

def sample_speaker_prompt(spk2item, spk_id, item):
    prompt_item = random.sample(spk2item[spk_id], 1)[0]
    if prompt_item['text'] == item['text']: # make sure test set prompt not used
        prompt_item = random.sample(spk2item[spk_id], 1)[0]
    while prompt_item["duration"] < 1:
        prompt_item = random.sample(spk2item[spk_id], 1)[0]
    return prompt_item

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def process_item(args, model, config, phn2num, audio_tokenizer, item, output_dir, i):
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    orig_audio = item['prompt_audio']

    item['output_filename'] = f'{i}.wav'

    # copy prompt to folder
    prompt_dir = os.path.join(output_dir, 'prompts')
    os.makedirs(prompt_dir, exist_ok=True)
    dest_path = os.path.join(prompt_dir, item['output_filename'])
    copy2(orig_audio, dest_path)

    orig_transcript = item['prompt_text']
    # test sentence

    sentence = item['text']
    item['text'] = sentence

    filepath = f"{output_dir}/{os.path.basename(orig_audio)[:-4]}.wav"
    # cut_off_sec = item['prompt_duration'] - 0.01
    # target_transcript = orig_transcript + item['verbatim']
    words = item['text'].split(" ")
    chunks = []
    N_WORDS = 20 
    for ix in range(0, len(words), N_WORDS):
        
        chunks.append(" ".join(words[ix: ix+N_WORDS]))
    chunks = list(filter(lambda x: x != "", chunks))
    # print('info', orig_audio)
    info = torchaudio.info(orig_audio)
    audio_dur = info.num_frames / info.sample_rate
    cut_off_sec = audio_dur - 0.01
    # print(audio_dur)
    # print('done info')

    assert cut_off_sec < audio_dur, f"cut_off_sec {cut_off_sec} is larger than the audio duration {audio_dur}"
    prompt_end_frame = int(cut_off_sec * info.sample_rate)

    decode_config = {
        'top_k': args.top_k, 'top_p': args.top_p, 'temperature': args.temperature, 
        'stop_repetition': args.stop_repetition, 'kvcache': args.kvcache, "codec_audio_sr": args.codec_audio_sr, 
        "codec_sr": args.codec_sr, "silence_tokens": args.silence_tokens, "sample_batch_size": args.sample_batch_size
    }
    prompt = orig_transcript
    audio_chunks = []
    for ix, chunk in enumerate(chunks):
        chunk_i = prompt + ' ' + chunk
        print("LN 109", chunk_i)
        concated_audio, gen_audio = inference_one_sample_graphemes(
            model, Namespace(**config), phn2num, audio_tokenizer, orig_audio, 
            chunk_i, device, decode_config, prompt_end_frame
        )
        audio_chunks.append(gen_audio.squeeze(0).cpu())
            # print(gen_audio.shape)
            # prompt = chunk
        # except Exception as e:
        #     # print('skipped', e, item)
        #     print('skipped', e)
        #     return

    gen_audio = torch.cat(audio_chunks, dim=1)
    # concated_audio, gen_audio = concated_audio[0].cpu(), gen_audio[0].cpu()

    # filename = f'{i+1}_' + os.path.basename(orig_audio)
    filename = item['output_filename']
    samples_dir = os.path.join(output_dir, 'samples_enhprompts')
    os.makedirs(samples_dir, exist_ok=True)
    filepath = f"{samples_dir}/{filename}"
    torchaudio.save(filepath, gen_audio, args.codec_audio_sr)
    print('Saved to ', filepath)

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from models import voicecraft
    model_path = args.model_path

    ckpt = torch.load(model_path, map_location='cpu')
    model = voicecraft.VoiceCraft(ckpt['config'])
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()

    config = vars(model.args)
    phn2num = ckpt["phn2num"]

    encodec_fn = "./pretrained_models/encodec_4cb2048_giga.th"
    if not os.path.exists(encodec_fn):
        os.system(f"wget https://huggingface.co/pyp1/VoiceCraft/resolve/main/encodec_4cb2048_giga.th")
        os.system(f"mv encodec_4cb2048_giga.th ./pretrained_models/encodec_4cb2048_giga.th")
    audio_tokenizer = AudioTokenizer(signature=encodec_fn, device=device)

    filepath = args.manifest_path # '/nlsasfs/home/ai4bharat/praveens/ttsteam/repos/voicecraft/demo/srvm/demo_sys/demo.json'
    test = read_jsonl(filepath)

    output_dir = os.path.join(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
        # with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        #     futures = [
        #         executor.submit(process_item, args, model, config, phn2num, audio_tokenizer, spk2item, item, language, output_dir)
        #         for item in test
        #     ]
        #     for future in tqdm(concurrent.futures.as_completed(futures), desc=f"Processing {language} ({split})"):
        #         future.result()
    for idx, item in enumerate(tqdm(test)):
        # try:
        process_item(args, model, config, phn2num, audio_tokenizer, item, output_dir, idx)
        # except Exception as e:
        #     print(e)
        #     pass


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--manifest_path", default='/nlsasfs/home/ai4bharat/praveens/ttsteam/repos/voicecraft/demo/srvm/demo_gupshup/manifest.jsonl', type=str, required=False, help="Base directory for manifests")
    parser.add_argument("--model_path", default='/home/tts/ttsteam/repos/VoiceCraft/logs/ivr_ia/gigaspeech/e830M_ft/best_bundle.pth', type=str, required=True, help="Path to the model file")
    parser.add_argument("--output_dir", default='/home/tts/ttsteam/repos/VoiceCraft/demo/ivr_ia', type=str, required=True, help="Directory to save output files")
    parser.add_argument("--language_family", default='indoaryan', type=str, required=False, help="language_family")
    parser.add_argument("--language", default='', type=str, required=False, help="language")
    parser.add_argument("--split", default='', type=str, required=False, help="split")
    parser.add_argument("--replace_path", action='store_true', help="Whether to replace the audio and transcript paths")
    parser.add_argument("--num_workers", default=1, type=int, help="Number of workers for parallel processing")
    parser.add_argument("--codec_audio_sr", default=16000, type=int, help="Sampling rate for codec audio")
    parser.add_argument("--codec_sr", default=50, type=int, help="Sampling rate for codec")
    parser.add_argument("--top_k", default=0, type=int, help="Top-K sampling for inference")
    parser.add_argument("--top_p", default=0.9, type=float, help="Top-P (nucleus) sampling for inference")
    parser.add_argument("--temperature", default=1.0, type=float, help="Temperature for sampling")
    parser.add_argument("--silence_tokens", default=[1388, 1898, 131], type=int, nargs='+', help="Tokens representing silence")
    parser.add_argument("--kvcache", default=1, type=int, help="Cache key and value tensors to save memory during inference")
    parser.add_argument("--stop_repetition", default=5, type=int, help="Stop repetition to avoid long silences")
    parser.add_argument("--sample_batch_size", default=3, type=int, help="Batch size for sampling during inference")
    parser.add_argument("--seed", default=1, type=int, help="Random seed for reproducibility")
    args = parser.parse_args()
    main(args)
