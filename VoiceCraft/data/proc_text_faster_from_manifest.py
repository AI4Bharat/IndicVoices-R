import os
import json
import math
import argparse
import torchaudio
from tqdm import tqdm

import numpy as np
import time
from datasets import load_dataset, DownloadConfig, Dataset, DatasetDict, Audio, load_from_disk


from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
from rich.progress import Progress

def read_jsonl(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def clone_fp(item):
    if ('audio_filepath' not in item):
        item["audio_filepath"] = item["filepath"]
    else:
        item['filepath'] = item['audio_filepath']
    item["segment_id"] = os.path.basename(item["filepath"])[:-4]
    item["begin_time"] = 0.0
    # item["end_time"] = float(ffmpeg.probe(item["audio_filepath"])['format']['duration'])
    # item["end_time"] = item['duration']
    info = torchaudio.info(item["audio_filepath"])
    item["end_time"]  = info.num_frames / info.sample_rate
    return item

def parse_args():
    parser = argparse.ArgumentParser(description="encode the librilight dataset using encodec model")
    parser.add_argument("--dataset_size", type=str, default='xs', help='sizes of gigaspeech, xs, s, m, l, xl. we use xl for VoiceCraft training, xs is good for debugging')
    parser.add_argument('--save_dir', type=str, default="/data/scratch/pyp/datasets/gigaspeech_phn_enc_manifest_debug", help="path to the manifest, phonemes, and encodec codes dirs")
    parser.add_argument('--n_workers', type=int, default=4, help="Number of parallel worker processes")
    return parser.parse_args()
    
if __name__ == "__main__":
    import logging
    formatter = (
        "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d || %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)
    args = parse_args()


    # from tokenizer import TextTokenizer, tokenize_text
    # get the path
    phn_save_root = os.path.join(args.save_dir, args.dataset_size, "phonemes")
    vocab_fn = os.path.join(args.save_dir, args.dataset_size, "vocab_all.txt")
    os.makedirs(phn_save_root, exist_ok=True)

    # https://github.com/SpeechColab/GigaSpeech
    # there are only four different punctuations
    # need to check whether there are other < started strings
    punc2sym = {" <COMMA>": ",", " <PERIOD>": ".", " <QUESTIONMARK>": "?", " <EXCLAMATIONPOINT>": "!"} # note the space in front of each punc name
    gar2sym = {"<SIL>": "#%#", "<MUSIC>": "##%", "<NOISE>": "%%#", "<OTHER>":"%#%"} # so that they are savely keep as the original sym when using tokenize_text
    punc2sym.update(gar2sym)

    word2sym = { "h æ ʃ h ɐ ʃ p ɚ s ɛ n t": "<MUSIC>", "h æ ʃ p ɚ s ɛ n t h æ ʃ": "<SIL>", "p ɚ s ɛ n t h ɐ ʃ p ɚ s ɛ n t": "<OTHER>", "p ɚ s ɛ n t p ɚ s ɛ n t h æ ʃ": "<NOISE>"}
    forbidden_words = set(['#%#', '##%', '%%#', '%#%'])

    # dc = DownloadConfig(cache_dir=args.download_to)
    stime = time.time()
    logging.info("loading the dataset...")
    # gs = load_dataset("speechcolab/gigaspeech", args.dataset_size, use_auth_token=True, cache_dir = args.download_to, download_config=dc)

    train = read_jsonl('/projects/data/ttsteam/datasets/manifests/voicecraft/in5k/metadata_train.json')
    test = read_jsonl('/projects/data/ttsteam/datasets/manifests/voicecraft/in5k/metadata_test.json')

    train_ds = Dataset.from_list(train)
    test_ds = Dataset.from_list(test)

    gs = DatasetDict({
        "train": train_ds,
        "validation": test_ds,
        "test": test_ds,
    })
    logging.info(f"time spend on loading the dataset: {time.time() - stime:.2f} seconds")

    splits = ['validation', 'test', 'train']
    
    logging.info(f"gigaspeech dataset {args.dataset_size} info: {gs}")
    logging.info(f"phonemizing...")
    phn_vocab = set()
    all_lens = []


    def process_item(item, phn_save_root, punc2sym, word2sym, forbidden_words):
        save_fn = os.path.join(phn_save_root, item['segment_id']+".txt")
        text = item['text']
        if sum(word in forbidden_words for word in text.split(" ")):
            logging.info(f"skip {item['segment_id']}, because it contains forbiden words. It's transcript: {text}")
            skip += 1
            return None, None
        for k, v in punc2sym.items():
            text = text.replace(k, v)
        # phn = tokenize_text(text_tokenizer, text)
        phn = text.split(" ")
        phn = list(map(lambda x: " ".join([y for y in x if y.isprintable()]), phn))
        phn = "_".join(phn)
        phn_seq = " ".join(phn)
        for k, v in word2sym.items():
            phn_seq = phn_seq.replace(k, v)
        return phn_seq.split(" "), len(phn_seq.split(" "))
        # with open(save_fn, "w") as f:
            # f.write(phn_seq)

    def process_split(split, gs, phn_save_root, punc2sym, word2sym, forbidden_words, num_workers):
        
        with Pool(processes=num_workers) as pool:
            with Progress() as progress:
                task = progress.add_task(f"Processing {split}...", total=len(gs[split]))
                results = []
                for item in gs[split]:
                    result = pool.apply_async(process_item, (item, phn_save_root, punc2sym, word2sym, forbidden_words))
                    results.append(result)

                for result in tqdm(results):
                    phn_seq, length = result.get()
                    if phn_seq:
                        phn_vocab.update(phn_seq)
                        all_lens.append(length)
                    progress.update(task, advance=1)
    
    for split in splits:
        logging.info(f"Now processing split {split}...")
        process_split(split, gs, phn_save_root, punc2sym, word2sym, forbidden_words, args.n_workers)
        logging.info(f"Split {split} has {len(gs[split])} samples in total.")
    
    print(f"phn vocab size: {len(list(phn_vocab))}")
    print("phn sequence stats: ")
    print(f"longest: {max(all_lens)}")
    print(f"shortest: {min(all_lens)}")
    print(f"median: {np.quantile(all_lens, 0.5)}")
    print(f"95 percentile longest: {np.quantile(all_lens, 0.95)}")
    print("write vocabulary to ", vocab_fn)

    phn_vocab = sorted(phn_vocab)
    with open(vocab_fn, "w") as f:
        for i, phn in enumerate(list(phn_vocab)):
            if i < len(list(phn_vocab)) - 1:
                f.write(f"{str(i)} {phn}\n")
            else:
                f.write(f"{str(i)} {phn}")

