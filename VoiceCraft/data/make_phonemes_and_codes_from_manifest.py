import os
import json
import math
import argparse
import torchaudio
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed

def read_jsonl(filepath):
    """
    Reads a JSON Lines file and returns a list of JSON objects.

    Args:
        filepath: Path to the JSON Lines file.

    Returns:
        A list of dictionaries, where each dictionary represents a parsed JSON object.
    """
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
    info = torchaudio.info(item["audio_filepath"])
    item["end_time"]  = info.num_frames / info.sample_rate
    return item

def parse_args():
    parser = argparse.ArgumentParser(description="encode the librilight dataset using encodec model")
    parser.add_argument("--dataset_size", type=str, default='xs', help='sizes of gigaspeech, xs, s, m, l, xl. we use xl for VoiceCraft training, xs is good for debugging')
    parser.add_argument('--download_to', type=str, default="/data/scratch/pyp/datasets/gigaspeech_debug", help="dir where you want the huggingface gigaspeech dataset to be downloaded to")
    parser.add_argument('--save_dir', type=str, default="/data/scratch/pyp/datasets/gigaspeech_phn_enc_manifest_debug", help="path to the manifest, phonemes, and encodec codes dirs")
    parser.add_argument('--encodec_model_path', type=str, default="/nlsasfs/home/ai4bharat/praveens/ttsteam/repos/invoicecraft/pretrained_models/encodec_4cb2048_giga.th")
    parser.add_argument('--n_workers', type=int, default=4, help="Number of parallel worker processes")
    parser.add_argument('--mega_batch_size', type=int, default=100, help="Number of samples in each mega batch for multiprocess dataloading")
    parser.add_argument('--batch_size', type=int, default=4, help="batch size for encodec encoding, decrease it if OOM. This is the sum of batch size *over each gpu*, so increase it if you are using more gpus")
    parser.add_argument('--model_sr', type=int, default=16000, help='encodec input audio sample rate')
    parser.add_argument('--downsample_rate', type=int, default=320, help='encodec downsample rate')
    parser.add_argument('--model_code_sr', type=int, default=50, help='encodec model code sample rate')
    parser.add_argument('--len_cap', type=float, default=35.0, help='will drop audios that are longer than this number')
    parser.add_argument('--max_len', type=int, default=30000, help='max length of audio in samples, if exceed, will cut a batch into half to process, decrease this number if OOM on your machine')
    parser.add_argument('--split_idx', type=int, required=True, help='Index of the current split (0-based)')
    parser.add_argument('--n_splits', type=int, required=True, help='Total number of splits')
    return parser.parse_args()
    
if __name__ == "__main__":
    import logging
    formatter = (
        "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d || %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)
    args = parse_args()

    import os
    import numpy as np
    import torch
    import tqdm
    import time
    from datasets import load_dataset, DownloadConfig, Dataset, DatasetDict, Audio, load_from_disk
    
    # get the path
    phn_save_root = os.path.join(args.save_dir, args.dataset_size, "phonemes")
    codes_save_root = os.path.join(args.save_dir, args.dataset_size, "encodec_16khz_4codebooks")
    vocab_fn = os.path.join(args.save_dir, args.dataset_size, "vocab.txt")
    os.makedirs(phn_save_root, exist_ok=True)
    os.makedirs(codes_save_root, exist_ok=True)


    def sort_by_audio_len(lens):
        inds = np.argsort(lens).tolist()
        print(len(inds))
        logging.info(f"longest: {lens[inds[-1]]*args.model_code_sr} encodec codes, {lens[inds[-1]]:.2f} sec.")
        logging.info(f"shortest: {lens[inds[0]]*args.model_code_sr} encodec codes, {lens[inds[0]]:.2f} sec.")
        logging.info(f"median: {lens[inds[len(inds)//2]]*args.model_code_sr} encodec codes, {lens[inds[len(inds)//2]]:.2f} sec.")
        logging.info(f"95 percentile longest: {lens[inds[int(len(inds)*0.95)]]*args.model_code_sr} encodec codes, {lens[inds[int(len(inds)*0.95)]]:.2f} sec.")
        return inds[::-1]
    
    def write_array_to_txt_file(array, filename):
        with open(filename, 'w') as f:
            for a in array[:-1]:
                f.write(' '.join(map(str, a))+'\n')
            f.write(' '.join(map(str, array[-1])))
    

    # Function to split data into chunks
    def get_chunk(data, split_idx, n_splits):
        chunk_size = math.ceil(len(data) / n_splits)
        start_idx = split_idx * chunk_size
        end_idx = min((split_idx + 1) * chunk_size, len(data))
        return data[start_idx:end_idx]

    ### phonemization
    # load tokenizer
    # load the encodec model
    from audiocraft.solvers import CompressionSolver
    model = CompressionSolver.model_from_checkpoint(args.encodec_model_path)
    model = model.cuda()
    model = model.eval()


    # https://github.com/SpeechColab/GigaSpeech
    # there are only four different punctuations
    # need to check whether there are other < started strings
    punc2sym = {" <COMMA>": ",", " <PERIOD>": ".", " <QUESTIONMARK>": "?", " <EXCLAMATIONPOINT>": "!"} # note the space in front of each punc name
    gar2sym = {"<SIL>": "#%#", "<MUSIC>": "##%", "<NOISE>": "%%#", "<OTHER>":"%#%"} # so that they are savely keep as the original sym when using tokenize_text
    punc2sym.update(gar2sym)

    word2sym = { "h æ ʃ h ɐ ʃ p ɚ s ɛ n t": "<MUSIC>", "h æ ʃ p ɚ s ɛ n t h æ ʃ": "<SIL>", "p ɚ s ɛ n t h ɐ ʃ p ɚ s ɛ n t": "<OTHER>", "p ɚ s ɛ n t p ɚ s ɛ n t h æ ʃ": "<NOISE>"}
    forbidden_words = set(['#%#', '##%', '%%#', '%#%'])

    stime = time.time()
    logging.info("loading the dataset...")

    train = read_jsonl('datasets/ivr/manifests/metadata_train.json')
    test = read_jsonl('datasets/ivr/manifests/metadata_test.json')

    def modify(t):
        if 'audio_filepath' not in t:
            t['audio_filepath'] = t['filepath']
        return t
    

    print('Total', len(train), len(test))
    train = get_chunk(train, args.split_idx, args.n_splits)
    test = get_chunk(test, args.split_idx, args.n_splits)
    print('Chunk', len(train), len(test))

    # Assuming clone_fp and train are already defined
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(clone_fp, item) for item in train]
        for _ in tqdm.tqdm(as_completed(futures), total=len(train), desc='train'):
            pass  # Progress bar will update as each task completes

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(clone_fp, item) for item in test]
        for _ in tqdm.tqdm(as_completed(futures), total=len(test), desc='test'):
            pass  # Progress bar will update as each task completes



    train = list(map(clone_fp, train))
    test = list(map(clone_fp, test))
    print("LN 161 Downsampling audios", )
    train_ds = Dataset.from_list(train).cast_column("audio_filepath", Audio(sampling_rate=16_000)).rename_columns({"audio_filepath": "audio"})
    test_ds = Dataset.from_list(test).cast_column("audio_filepath", Audio(sampling_rate=16_000)).rename_columns({"audio_filepath": "audio"})

    gs = DatasetDict({
        "train": train_ds,
        "validation": test_ds,
        "test": test_ds,
    })
    gs.save_to_disk(f"hf_cache/{args.split_idx}__{args.n_splits}")

    # gs = load_from_disk(f"hf_cache/{args.split_idx}__{args.n_splits}")
    logging.info(f"time spend on loading the dataset: {time.time() - stime:.2f} seconds")

    splits = ['validation', 'test', 'train']
    
    logging.info(f"gigaspeech dataset {args.dataset_size} info: {gs}")
    logging.info(f"processing texts...")
    phn_vocab = set()
    all_lens = []
    
    # you will see a ton of [WARNING] words_mismatch.py:88......, it's not a issue
    for split in tqdm.tqdm(splits):
        skip = 0
        logging.info(f"now processing split {split}...")
        for item in tqdm.tqdm(gs[split]):
            save_fn = os.path.join(phn_save_root, item['segment_id']+".txt")
            text = item['text']
            if sum(word in forbidden_words for word in text.split(" ")):
                logging.info(f"skip {item['segment_id']}, because it contains forbiden words. It's transcript: {text}")
                skip += 1
                continue
            for k, v in punc2sym.items():
                text = text.replace(k, v)
            # phn = tokenize_text(text_tokenizer, text)
            phn = text.split(" ")
            phn = list(map(lambda x: " ".join([y for y in x if y.isprintable()]), phn))
            phn = "_".join(phn)
            phn_seq = " ".join(phn)
            for k, v in word2sym.items():
                phn_seq = phn_seq.replace(k, v)
            phn_vocab.update(phn_seq.split(" "))
            all_lens.append(len(phn_seq.split(" ")))
            with open(save_fn, "w") as f:
                f.write(phn_seq)
        logging.info(f"split {split} has {len(gs[split])} samples in total, skipped {skip} due to forbiden words")

    print(f"phn vocab size: {len(list(phn_vocab))}")
    print("phn sequence stats: ")
    print(f"longest: {max(all_lens)}")
    print(f"shortest: {min(all_lens)}")
    print(f"median: {np.quantile(all_lens, 0.5)}")
    print(f"95 percentile longest: {np.quantile(all_lens, 0.95)}")
    print("write vocabulary to ", vocab_fn)
    with open(vocab_fn, "w") as f:
        for i, phn in enumerate(list(phn_vocab)):
            if i < len(list(phn_vocab)) - 1:
                f.write(f"{str(i)} {phn}\n")
            else:
                f.write(f"{str(i)} {phn}")

    class mydataset(torch.utils.data.Dataset):
        def __init__(self, split):
            super().__init__()
            self.data = gs[split]
        def __len__(self):
            return len(self.data)
        def __getitem__(self, ind):
            try:
                segment_id, audio, sr, text, begin_time, end_time = self.data[ind]['segment_id'], torch.from_numpy(self.data[ind]['audio']['array']).float(), self.data[ind]['audio']['sampling_rate'], self.data[ind]['text'], self.data[ind]['begin_time'], self.data[ind]['end_time']
            except:
                return self.__getitem__(random.randint(0, len(self.data)))
            
            return segment_id, audio, sr, text, begin_time, end_time
        
        def collate(self, batch):
            res = {'segment_id': [], "audio": [], "sr": [], "text": [], "begin_time": [], "end_time": []}
            for item in batch:
                if item[0] != None and item[4] != None and item[5] != None:
                    res['segment_id'].append(item[0])
                    res['audio'].append(item[1])
                    res['sr'].append(item[2])
                    res['text'].append(item[3])
                    # res["audio_len"].append(item[4])
                    res['begin_time'].append(item[4])
                    res['end_time'].append(item[5])
            return res


    ## encodec codes extraction
    logging.info("encodec encoding...")
    train_dataset = mydataset('train')
    train_loader = torch.torch.utils.data.DataLoader(train_dataset, batch_size=args.mega_batch_size, shuffle=False, drop_last=False, num_workers=args.n_workers, collate_fn=train_dataset.collate)
    validation_dataset = mydataset('validation')
    validation_loader = torch.torch.utils.data.DataLoader(validation_dataset, batch_size=args.mega_batch_size, shuffle=False, drop_last=False, num_workers=args.n_workers, collate_fn=validation_dataset.collate)
    test_dataset = mydataset('test')
    test_loader = torch.torch.utils.data.DataLoader(test_dataset, batch_size=args.mega_batch_size, shuffle=False, drop_last=False, num_workers=args.n_workers, collate_fn=test_dataset.collate)
    splits = ['validation', 'test', 'train']
    loaders = [validation_loader, test_loader, train_loader]
    for split, loader in zip(splits, loaders):
        skip = 0
        logging.info(f"now processing split {split}...")
        mega_n_steps = int(np.ceil(len(gs[split]) / args.mega_batch_size))
        logging.info(f"partition the split {split} into {mega_n_steps} parts, each has {args.mega_batch_size} samples")
        for m, mega_batch in enumerate(loader):
            # print(mega_batch)
            logging.info(f"====================================")
            logging.info(f"====================================")
            logging.info(f"now processing mega step {m+1}/{mega_n_steps}")
            lengths = np.array(mega_batch['end_time']) - np.array(mega_batch['begin_time'])
            # lengths = mega_batch["audio_len"]
            # print(lengths)
            sorted_inds = sort_by_audio_len(lengths)
            for j in range(len(sorted_inds))[::-1]:
                if lengths[sorted_inds[j]] < 0.2 or lengths[sorted_inds[j]] > args.len_cap: # skip samples that are too short (shorter than 0.2s), or too big (bigger than 80s)
                    skip += 1
                    del sorted_inds[j]
            
            n_steps = int(np.ceil(len(sorted_inds) / args.batch_size))
            for n in tqdm.tqdm(range(n_steps), disable=True):
                # print("LN 247")
                inds_used = sorted_inds[n*args.batch_size:(n+1)*args.batch_size]
                audio_batch = [mega_batch['audio'][id] for id in inds_used]
                sr_batch = [mega_batch['sr'][id] for id in inds_used]
                segment_id_batch = [mega_batch['segment_id'][id] for id in inds_used]
                text_batch = [mega_batch['text'][id] for id in inds_used]
                padded_wav = torch.nn.utils.rnn.pad_sequence(audio_batch, batch_first=True).unsqueeze(1) # [B, T] -> [B, 1, T]
                all_lens = [lengths[id] for id in inds_used]
                with torch.no_grad():
                    if max(all_lens) > args.max_len and len(all_lens) > 1: # NOTE decrease args.max_len if OOM, or chunk it into more than 2 forward passes
                        codes = []
                        inwav = padded_wav.cuda()
                        codes.append(model.encode(inwav[:len(inwav)//2])[0].cpu())
                        codes.append(model.encode(inwav[len(inwav)//2:])[0].cpu())
                        codes = torch.cat(codes, dim=0)
                    else:
                        encoded_frames = model.encode(padded_wav.cuda())
                        # logging.info(f"encoded_frames: {encoded_frames[0].shape}")
                        codes = encoded_frames[0].cpu()
                # print(codes)
                for i, length in enumerate(all_lens):
                    save_fn = os.path.join(codes_save_root, segment_id_batch[i]+".txt")
                    # print("LN 269 ", save_fn)
                    actual_len = round(length * args.model_code_sr) # 320 is downsample rate for this model
                    cur_code = codes[i].tolist() if type(codes) == list else codes[i, :, :actual_len].tolist()
                    write_array_to_txt_file(cur_code, save_fn)
