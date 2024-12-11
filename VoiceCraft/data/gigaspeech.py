import os
import torch
import random
import copy
import logging
import shutil

class dataset(torch.utils.data.Dataset):
    def __init__(self, args, split):
        super().__init__()
        self.args = args
        self.split = split
        assert self.split in ['train', 'validation', 'test']
        manifest_fn = os.path.join(self.args.dataset_dir, self.args.manifest_name, self.split+".txt")

        with open(manifest_fn, "r") as rf:
            data = [l.strip().split("\t") for l in rf.readlines()]
        lengths_list = [int(item[2]) for item in data]
        self.data = []
        self.lengths_list = []
        for d, l in zip(data, lengths_list):
            if l >= self.args.encodec_sr*self.args.audio_min_length:
                if self.args.drop_long and l > self.args.encodec_sr*self.args.audio_max_length:
                    continue
                self.data.append(d)
                self.lengths_list.append(l)
        
        if self.args.use_prompt:
            spk2idx = {}
            for idx, item in enumerate(self.data):
                spk = item[3]
                if spk in spk2idx:
                    spk2idx[spk].append(idx)
                else:
                    spk2idx[spk] = [idx]
            self.spk2idx = spk2idx

        logging.info(f"number of data points for {self.split} split: {len(self.lengths_list)}")

        # phoneme vocabulary
        vocab_fn = os.path.join(self.args.dataset_dir,"vocab.txt")
        shutil.copy(vocab_fn, os.path.join(self.args.exp_dir, "vocab.txt"))
        with open(vocab_fn, "r") as f:
            temp = [l.strip().split(" ") for l in f.readlines() if len(l.strip().split(" ")) >= 2]
            print("VOCAB SIZE:", len(temp))
            self.phn2num = {item[1]:int(item[0]) for item in temp}
        
        self.symbol_set = set(["<SIL>", "<MUSIC>", "<NOISE>", "<OTHER>"])
    
    def __len__(self):
        return len(self.lengths_list)
    
    def _load_phn_enc(self, index):
        item = self.data[index]
        pf = os.path.join(self.args.dataset_dir, self.args.phn_folder_name, item[1]+".txt")
        ef = os.path.join(self.args.dataset_dir, self.args.encodec_folder_name, item[1]+".txt")
        try:
            with open(pf, "r") as p, open(ef, "r") as e:
                phns = [l.strip() for l in p.readlines()]
                assert len(phns) == 1, phns
                # x = [self.phn2num[item] for item in phns[0].split(" ") if item not in self.symbol_set] # drop ["<SIL>", "<MUSIC>", "<NOISE>", "<OTHER>"], as they are not in training set annotation
                x = [self.phn2num[item] for item in phns[0].split(" ") if item not in self.symbol_set and item!='' and item in self.phn2num]
                encos = [l.strip().split() for k, l in enumerate(e.readlines()) if k < self.args.n_codebooks]
                
                assert len(encos) == self.args.n_codebooks, ef
                if self.args.special_first:
                    y = [[int(n)+self.args.n_special for n in l] for l in encos]
                else:
                    y = [[int(n) for n in l] for l in encos]
        except Exception as e:
            logging.info(f"loading failed for {pf} and {ef}, maybe files don't exist or are corrupted")
            logging.info(f"error message: {e}")
            # logging.info(f"Len of encos: {len(encos)}")
            # logging.info(f"Len of phns: {len(phns)}")
            return [], [[]]

        return x, y

    def sample_prompt(self, index):
        speaker_style_id = self.data[index][3]
        candidates = self.spk2idx[speaker_style_id]
        candidate = random.choice(candidates)
        if candidate == index:
            candidate = random.choice(candidates)
        x2, y2 = self._load_phn_enc(candidate)
        x2_len, y2_len = len(x2), len(y2[0])
        if x2_len == 0 or y2_len == 0:
            return None, None
        if y2_len < self.args.encodec_sr*self.args.audio_min_length or x2_len > self.args.text_max_length or y2_len > self.args.encodec_sr*self.args.audio_max_length:
            return None, None
        return x2, y2

    def __getitem__(self, index):
        x, y = self._load_phn_enc(index)
        x1, y1 = x, y
        x_len, y_len = len(x), len(y[0])
        y1_len = y_len

        if x_len == 0 or y_len == 0:
            return {
            "x": None, 
            "x_len": None, 
            "y": None, 
            "y_len": None, 
            "y1_len": None,
            "y_mask_interval": None, # index y_mask_interval[1] is the position of start_of_continue token
            "extra_mask_start": None # this is only used in VE1
            }
        while y_len < self.args.encodec_sr*self.args.audio_min_length:
            assert not self.args.dynamic_batching
            index = random.choice(range(len(self))) # regenerate an index
            x, y = self._load_phn_enc(index)
            x_len, y_len = len(x), len(y[0])
            y1_len = y_len
        if self.args.drop_long:
            while x_len > self.args.text_max_length or y_len > self.args.encodec_sr*self.args.audio_max_length:
                index = random.choice(range(len(self))) # regenerate an index
                x, y = self._load_phn_enc(index)
                x_len, y_len = len(x), len(y[0])
                y1_len = y_len

        ### padding and cropping below ###
        ### padding and cropping below ###
        # adjust the length of encodec codes, pad to max_len or randomly crop
        # orig_y_len = copy.copy(y_len)
        # max_len = int(self.args.audio_max_length * self.args.encodec_sr)
        # if y_len > max_len:
        #     audio_start = random.choice(range(0, y_len-max_len))
        #     for i in range(len(y)):
        #         y[i] = y[i][audio_start:(audio_start+max_len)]
        #     y_len = max_len
        # else:
        #     audio_start = 0
        #     if not self.args.dynamic_batching:
        #         pad = [0] * (max_len - y_len) if self.args.sep_special_token else [self.args.audio_pad_token] * (max_len - y_len)
        #         for i in range(len(y)):
        #             y[i] = y[i] + pad

        # print(len(y[0]), len(y))

        # print(f"x_len: {x_len}, y_len: {y_len}, y1_len: {y1_len}")
        if self.args.use_prompt:
            sample_prompt = random.uniform(0, 1) > self.args.use_prompt_prob
            if sample_prompt:
                x2, y2 = self.sample_prompt(index)
                # print(len(x2), len(y2[0]))
                if x2 is not None and y2 is not None and x != x2 and y != y2:
                    x = x + x2
                    x_len = len(x)
                    y = [y1_i + y2_i for y1_i, y2_i in zip(y, y2)]
                    y_len = len(y[0])
                    # print(f"x2_len: {len(x2)}, y2_len: {len(y2[0])}")
                    # print(f"x1: {len(x1)}, x2: {len(x2)}, y1: {len(y1[0])}, y2: {len(y2[0])}, x: {len(x)}, y: {len(y)}")

        # print(f"AFTER PROMPT: x_len: {x_len}, y_len: {y_len}, y1_len: {y1_len}")
        # print(f"x_len_final: {len(x)}, y_len_final: {len(y[0])}")
        assert y_len >= y1_len, f"y_len {y_len} is smaller then y1_len ({y1_len}) {self.data[index]}"

        return {
            "x": torch.LongTensor(x), 
            "x_len": x_len, 
            "y": torch.LongTensor(y), 
            "y_len": y_len,
            "y1_len": y1_len,
            }
            

    def collate(self, batch):
        out = {key:[] for key in batch[0]}
        for item in batch:
            if item['x'] == None: # deal with load failure
                continue
            for key, val in item.items():
                out[key].append(val)
        res = {}
        if self.args.pad_x:
            res["x"] = torch.stack(out["x"], dim=0)
        else:
            res["x"] = torch.nn.utils.rnn.pad_sequence(out["x"], batch_first=True, padding_value=self.args.text_pad_token)
        res["x_lens"] = torch.LongTensor(out["x_len"])
        if self.args.dynamic_batching:
            if out['y'][0].ndim==2:
                res['y'] = torch.nn.utils.rnn.pad_sequence([item.transpose(1,0) for item in out['y']],padding_value=self.args.audio_pad_token)
                res['y'] = res['y'].permute(1,2,0) # T B K -> B K T
            else:
                assert out['y'][0].ndim==1, out['y'][0].shape
                res['y'] = torch.nn.utils.rnn.pad_sequence(out['y'], batch_first=True, padding_value=self.args.audio_pad_token)
        else:
            res['y'] = torch.stack(out['y'], dim=0)
        res["y_lens"] = torch.LongTensor(out["y_len"])
        res["y1_lens"] = torch.LongTensor(out["y1_len"])
        res["text_padding_mask"] = torch.arange(res['x'][0].shape[-1]).unsqueeze(0) >= res['x_lens'].unsqueeze(1)
        res["audio_padding_mask"] = torch.arange(res['y'][0].shape[-1]).unsqueeze(0) >= res['y_lens'].unsqueeze(1)
        return res