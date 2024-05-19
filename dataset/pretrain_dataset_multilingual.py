import base64
import json
import copy
import math
import random
import sys
import re
import io
import traceback
from base64 import b64decode

from random import randint, shuffle
from random import random as rand

import torch
from torchvision.transforms.functional import hflip, resize
from torchvision.transforms import InterpolationMode

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset import build_tokenizer
from dataset.utils import pre_caption
from dataset.dist_dataset import DistLineReadingDataset


class TextMaskingGenerator:
    def __init__(self, tokenizer, mask_prob, mask_max, skipgram_prb=0.2, skipgram_size=3, mask_whole_word=True,
                 use_roberta=False):
        self.id2token = {i: w for w, i in tokenizer.get_vocab().items()}
        print("len(tokenizer.id2token), ", len(self.id2token), flush=True)

        self.use_roberta = use_roberta

        for i in range(len(self.id2token)):
            assert i in self.id2token.keys()  # check

        self.cls_token = tokenizer.cls_token
        self.mask_token = tokenizer.mask_token

        self.mask_max = mask_max
        self.mask_prob = mask_prob

        self.skipgram_prb = skipgram_prb
        self.skipgram_size = skipgram_size
        self.mask_whole_word = mask_whole_word

    def get_random_word(self):
        i = randint(0, len(self.id2token) - 1)
        return self.id2token[i]

    def __call__(self, tokens: list):  # tokens: [CLS] + ...
        n_pred = min(self.mask_max, max(
            1, int(round(len(tokens) * self.mask_prob))))

        # candidate positions of masked tokens
        assert tokens[0] == self.cls_token
        special_pos = set([0])  # will not be masked
        cand_pos = list(range(1, len(tokens)))

        shuffle(cand_pos)
        masked_pos = set()
        max_cand_pos = max(cand_pos)
        for pos in cand_pos:
            if len(masked_pos) >= n_pred:
                break
            if pos in masked_pos:
                continue

            def _expand_whole_word(st, end):
                new_st, new_end = st, end

                if self.use_roberta:
                    while (new_st > 1) and (tokens[new_st][0] != 'Ġ'):
                        new_st -= 1
                    while (new_end < len(tokens)) and (tokens[new_end][0] != 'Ġ'):
                        new_end += 1
                else:
                    # bert, WordPiece
                    while (new_st >= 0) and tokens[new_st].startswith('##'):
                        new_st -= 1
                    while (new_end < len(tokens)) and tokens[new_end].startswith('##'):
                        new_end += 1

                return new_st, new_end

            if (self.skipgram_prb > 0) and (self.skipgram_size >= 2) and (rand() < self.skipgram_prb):
                # ngram
                cur_skipgram_size = randint(2, self.skipgram_size)
                if self.mask_whole_word:
                    st_pos, end_pos = _expand_whole_word(
                        pos, pos + cur_skipgram_size)
                else:
                    st_pos, end_pos = pos, pos + cur_skipgram_size
            else:
                if self.mask_whole_word:
                    st_pos, end_pos = _expand_whole_word(pos, pos + 1)
                else:
                    st_pos, end_pos = pos, pos + 1

            for mp in range(st_pos, end_pos):
                if (0 < mp <= max_cand_pos) and (mp not in special_pos):
                    masked_pos.add(mp)
                else:
                    break

        masked_pos = list(masked_pos)
        n_real_pred = len(masked_pos)
        if n_real_pred > n_pred:
            shuffle(masked_pos)
            masked_pos = masked_pos[:n_pred]

        for pos in masked_pos:
            if rand() < 0.8:  # 80%
                tokens[pos] = self.mask_token
            elif rand() < 0.5:  # 10%
                tokens[pos] = self.get_random_word()

        return tokens, masked_pos


class ImageMultiTextDataset(DistLineReadingDataset):
    def __init__(self, config, data_path, rank=0, world_size=1, shuffle=True, repeat=True, transform=None):
        super().__init__(data_path, rank, world_size, shuffle, repeat)

        if 'images' in config.keys():
            self.image_key = config['images']['image_key']
            self.is_image_rpath = config['images']['is_image_rpath']
            self.caption_key = config['images']['caption_key']
            self.batch_size = config['images']['batch_size']
            self.tokenized = config['images']['tokenized']
            if 'language_chosen' in config['images'].keys():
                assert isinstance(config['images']['language_chosen'], list)
                self.language_chosen = config['images']['language_chosen']
                print("### language_chosen, ", self.language_chosen, flush=True)
            else:
                self.language_chosen = []
            self.lan_num = len(self.language_chosen)
            self.lanid_ls = list(range(self.lan_num))
        assert 'xlm-roberta' in config['text_encoder']
        self.tokenizer = build_tokenizer(config['text_encoder'])

        self.add_eos = True  # always add eos

        self.cls_token = self.tokenizer.cls_token
        self.eos_token = self.tokenizer.sep_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.mask_token_id = self.tokenizer.mask_token_id

        self.mask_generator = TextMaskingGenerator(self.tokenizer, config['mask_prob'],
                                                   config['max_masks'], config['skipgram_prb'],
                                                   config['skipgram_size'], mask_whole_word=False)

        self.PAD_mask = -100  # loss will ignore this
        self.max_words = config['max_words']
        self.max_tokens = config['max_tokens']
        self.max_masks = config['max_masks']

        self.transform = transform
        self.image_res = config['image_res']
        self.patch_size = config['patch_size']
        assert self.image_res % self.patch_size == 0
        self.num_patch = int(self.image_res / self.patch_size)

        self.sample_2_captions = config['sample_2_captions']
        self.sample_n_captions = config['sample_n_captions']
        self.caption_num = config['caption_num']
        print("### sample_2_captions: ", self.sample_2_captions)

    def get_caption(self, captions, language='', return_keys=False):
        if isinstance(captions, list):
            captions = random.choice(captions)

        assert isinstance(captions, dict)

        if language:
            return captions[language]

        if len(self.language_chosen):
            to_be_chosen = set(captions.keys()) & self.language_chosen
            assert len(to_be_chosen) >= len(self.language_chosen)
        else:
            to_be_chosen = set(captions.keys())

        if len(to_be_chosen) >= 1:
            k = random.choice(list(to_be_chosen))

            if return_keys:
                return captions[k], k

            return captions[k]

        else:
            raise ValueError("len(to_be_chosen) < 1")

    def get_2_captions(self, captions, language='', return_keys=False):
        if isinstance(captions, list):
            captions = random.choice(captions)

        assert isinstance(captions, dict)

        if language:
            return captions[language]

        if len(self.language_chosen):
            to_be_chosen = set(captions.keys()) & self.language_chosen
            assert len(to_be_chosen) >= len(self.language_chosen)
        else:
            to_be_chosen = set(captions.keys())

        if len(to_be_chosen) >= 2:
            k, k_2 = random.sample(to_be_chosen, 2)

            if return_keys:
                return captions[k], captions[k_2], k, k_2

            return captions[k], captions[k_2]

        else:
            raise ValueError("len(to_be_chosen) < 2")

    def __iter__(self):
        for example in self.generate():
            try:
                ann = json.loads(example)
                assert isinstance(ann, dict), "ann is not dict"
                if self.is_image_rpath:  # read path or base64 encoding
                    image = ann[self.image_key].encode('utf-8')
                    image = base64.b64decode(image)
                    image = io.BytesIO(image)
                    image = Image.open(image).convert('RGB')
                else:
                    path = ann['image_path']
                    # if reading from HDFS, use this:
                    # image = Image.open(io.BytesIO(b64decode(ann[self.image_key]))).convert("RGB")
                    image = Image.open(path).convert('RGB')
                image = self.transform(image)

                if self.sample_2_captions:
                    caption, caption_2 = self.get_2_captions(ann[self.caption_key])

                    text_ids, text_atts, text_ids_masked, masked_pos, masked_ids = self.preprocess(caption)
                    text_ids_2, text_atts_2, text_ids_masked_2, masked_pos_2, masked_ids_2 = self.preprocess(caption_2)
                    yield image, text_ids, text_atts, text_ids_masked, masked_pos, masked_ids, \
                          text_ids_2, text_atts_2, text_ids_masked_2, masked_pos_2, masked_ids_2
                elif self.sample_n_captions:
                    # 在这里获得多个文本的数据
                    captions = ann[self.caption_key]
                    text_ids, text_atts, text_ids_masked, masked_pos, masked_ids = [], [], [], [], []
                    if not self.language_chosen:
                        self.language_chosen = list(captions.keys())
                        # sort language original list by default dict order
                        self.language_chosen = sorted(self.language_chosen)
                        self.lan_num = len(self.language_chosen)
                        self.lanid_ls = list(range(self.lan_num))
                        print('language_chosen:', self.language_chosen)
                    lanid_chosen = self.lanid_ls
                    if self.caption_num > 0:
                        # caption_num > 0 ==> random choose caption_num language text
                        lanid_chosen = random.sample(lanid_chosen, self.caption_num)
                        lanid_chosen.sort()
                    language_ls = []
                    # get language list as same as the original sequence
                    for lanid in lanid_chosen:
                        language_ls.append(self.language_chosen[lanid])
                    for lan in language_ls:
                        caption = captions[lan]
                        text_id, text_att, text_id_masked, masked_po, masked_id = self.preprocess(caption)
                        if not isinstance(text_id, torch.Tensor):
                            text_id = torch.tensor(text_id, dtype=torch.long)
                        if not isinstance(text_att, torch.Tensor):
                            text_att = torch.tensor(text_att, dtype=torch.long)
                        if not isinstance(text_id_masked, torch.Tensor):
                            text_id_masked = torch.tensor(text_id_masked, dtype=torch.long)
                        if not isinstance(masked_po, torch.Tensor):
                            masked_po = torch.tensor(masked_po, dtype=torch.long)
                        if not isinstance(masked_id, torch.Tensor):
                            masked_id = torch.tensor(masked_id, dtype=torch.long)
                        text_ids.append(text_id)
                        text_atts.append(text_att)
                        text_ids_masked.append(text_id_masked)
                        masked_pos.append(masked_po)
                        masked_ids.append(masked_id)
                    yield image, text_ids, text_atts, text_ids_masked, masked_pos, masked_ids
                else:
                    caption = self.get_caption(ann[self.caption_key])
                    text_ids, text_atts, text_ids_masked, masked_pos, masked_ids = self.preprocess(caption)
                    text_ids_2, text_atts_2, text_ids_masked_2, masked_pos_2, masked_ids_2 = [None] * 5

                    yield image, text_ids, text_atts, text_ids_masked, masked_pos, masked_ids, \
                          text_ids_2, text_atts_2, text_ids_masked_2, masked_pos_2, masked_ids_2

            except Exception as e:
                print(traceback.format_exc())
                print('encounter broken data: %s' % e)
                print('-' * 20)
                sys.stdout.flush()

    def preprocess(self, text):
        if self.tokenized:
            tokens = text.strip().split(' ')
        else:
            text = pre_caption(text, self.max_words)  # be careful, if text is '', it will cause error
            tokens = self.tokenizer.tokenize(text)

        tokens = [self.cls_token] + tokens[:self.max_tokens - 1]

        if self.add_eos:
            tokens = tokens[:self.max_tokens - 1]
            tokens += [self.eos_token]

        n_tokens = len(tokens)
        assert n_tokens >= 2, "len(word tokens) < 2"

        text_ids = self.tokenizer.convert_tokens_to_ids(tokens)  # list of int

        tokens_masked, masked_pos = self.mask_generator(copy.deepcopy(tokens))
        text_ids_masked = self.tokenizer.convert_tokens_to_ids(tokens_masked)  # list of int
        masked_ids = [text_ids[p] for p in masked_pos]

        # pad
        n_pad = self.max_tokens - n_tokens
        text_ids = text_ids + [self.pad_token_id] * n_pad
        text_atts = [1] * n_tokens + [0] * n_pad

        text_ids_masked = text_ids_masked + [self.pad_token_id] * n_pad
        n_pad = self.max_masks - len(masked_ids)
        masked_pos = masked_pos + [0] * n_pad
        masked_ids = masked_ids + [self.PAD_mask] * n_pad

        return text_ids, text_atts, text_ids_masked, masked_pos, masked_ids

    def collate_fn(self, batch):
        batch_tensors = []
        if self.sample_n_captions:
            for x in zip(*batch):  # x in ([image,...,image],[text_ids,...,text_ids],...)
                if isinstance(x[0], torch.Tensor):
                    batch_tensors.append(torch.stack(x))
                else:
                    data_ls = []
                    for y in x:
                        data_ls.append(torch.stack(y))
                    batch_tensors.append(torch.concat(data_ls))
        else:
            for x in zip(*batch):  # x in ([image,...,image],[text_ids,...,text_ids],...)
                if x[0] is None:
                    batch_tensors.append(None)
                elif isinstance(x[0], torch.Tensor):
                    batch_tensors.append(torch.stack(x))
                else:
                    batch_tensors.append(torch.tensor(x, dtype=torch.long))
        return batch_tensors
