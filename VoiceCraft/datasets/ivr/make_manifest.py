import os
import os.path as osp
import json
import pandas as pd
from multiprocessing import Pool
from rich.progress import Progress

def read_jsonl(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def get_codec_length(filepath):
    with open(filepath, 'r') as fp:
        str_codes = [l.strip().split() for l in fp.readlines()]
        return len(str_codes[0])

def get_codec_len(filepath):
    with open(filepath, 'r') as fp:
        quantizer_one_codes = fp.readline().strip().split()
        return len(quantizer_one_codes)

def process_file(file):
    segment_id = file.replace('.txt', '').strip()
    segment_len = get_codec_len(osp.join('encodec_16khz_4codebooks', file))
    return (0, segment_id, segment_len)

def calculate_lengths_parallel(valid_files, num_workers=64):
    records = []
    
    # Use Rich progress bar to track progress
    with Progress() as progress:
        task = progress.add_task("[green]Processing files...", total=len(valid_files))

        # Use multiprocessing to parallelize the loop
        with Pool(num_workers) as pool:
            for result in pool.imap_unordered(process_file, valid_files):
                records.append(result)
                progress.advance(task)

    # Create DataFrame from results
    df = pd.DataFrame.from_records(records)
    print('Total:', len(df))
    return df


if __name__ == '__main__':
    os.makedirs('manifest', exist_ok=True)

    # Find matching codec-text pairs
    code_files = set(os.listdir('encodec_16khz_4codebooks'))
    phoneme_files = set(os.listdir('phonemes'))
    valid_files = list(code_files & phoneme_files) # set intersection
    
    print('Code files:', len(code_files))
    print('Phoneme files:', len(phoneme_files))
    print('Valid files:', len(valid_files))

    # Calculate codec lengths
    df = calculate_lengths_parallel(valid_files)

    # Get train-test splits from original manifests and write to new manifests
    train = read_jsonl('datasets/ivr/manifests/metadata_train.json')
    train = [osp.splitext(osp.basename(d['audio_filepath']))[0] for d in train]
    train_df = df[df[1].isin(train)]
    train_df.to_csv('manifest/train.txt', sep='\t', index=False, header=False)

    test = read_jsonl('datasets/ivr/manifests/metadata_valid.json')
    test = [osp.splitext(osp.basename(d['audio_filepath']))[0] for d in test]
    test_df = df[df[1].isin(test)]
    test_df.to_csv('manifest/validation.txt', sep='\t', index=False, header=False)

    test = read_jsonl('datasets/ivr/manifests/metadata_valid.json')
    test = [osp.splitext(osp.basename(d['audio_filepath']))[0] for d in test]
    test_df = df[df[1].isin(test)]
    test_df.to_csv('manifest/validation.txt', sep='\t', index=False, header=False)

    print('# Speech Tokens: ', df[2].sum())