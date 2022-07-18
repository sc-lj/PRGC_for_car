# /usr/bin/env python
# coding=utf-8
"""Dataloader"""

import os
import json

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from transformers import BertTokenizer

from dataloader_utils import read_examples, convert_examples_to_features


class FeatureDataset(Dataset):
    """Pytorch Dataset for InputFeatures
    """

    def __init__(self, features):
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]


class CustomDataLoader(object):
    def __init__(self, params):
        self.params = params

        self.train_batch_size = params.train_batch_size
        self.val_batch_size = params.val_batch_size
        self.test_batch_size = params.test_batch_size

        self.data_dir = params.data_dir
        self.max_seq_length = params.max_seq_length
        self.tokenizer = BertTokenizer(vocab_file=os.path.join(params.bert_model_dir, 'vocab.txt'),
                                       do_lower_case=False)
        self.data_cache = params.data_cache
        self.rel_num = params.rel_num

    # @staticmethod
    def collate_fn_train(self, features):
        """将InputFeatures转换为Tensor,for CarFaultRelation data。
        Args:
            features (List[InputFeatures])
        Returns:
            tensors (List[Tensors])
        """
        max_len = max([f.text_len for f in features])
        batch_size = len(features)
        batch_input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        batch_attention_masks = torch.zeros(
            batch_size, max_len, dtype=torch.long)
        batch_seq_tags = torch.zeros(batch_size, 2, max_len, dtype=torch.long)
        batch_poten_relations = torch.zeros(batch_size, dtype=torch.long)
        batch_corres_tags = torch.zeros(
            batch_size, max_len, max_len, dtype=torch.long)
        batch_rel_tags = torch.zeros(
            batch_size, self.rel_num, dtype=torch.long)

        for i, f in enumerate(features):
            text_len = f.text_len
            batch_input_ids[i, :text_len] = torch.tensor(f.input_ids)
            batch_attention_masks[i, :text_len] = torch.tensor(
                f.attention_mask)
            batch_seq_tags[i, :, :text_len] = torch.tensor(f.seq_tag)
            batch_poten_relations[i, ] = torch.tensor(f.relation)
            batch_corres_tags[i, :text_len,
                              :text_len] = torch.tensor(f.corres_tag)
            batch_rel_tags[i] = torch.tensor(f.rel_tag)

        tensors = [batch_input_ids, batch_attention_masks, batch_seq_tags,
                   batch_poten_relations, batch_corres_tags, batch_rel_tags]
        return tensors

    def collate_fn_train_v1(self, features):
        """将InputFeatures转换为Tensor
        Args:
            features (List[InputFeatures])
        Returns:
            tensors (List[Tensors])
        """
        input_ids = torch.tensor(
            [f.input_ids for f in features], dtype=torch.long)
        attention_mask = torch.tensor(
            [f.attention_mask for f in features], dtype=torch.long)
        seq_tags = torch.tensor(
            [f.seq_tag for f in features], dtype=torch.long)
        poten_relations = torch.tensor(
            [f.relation for f in features], dtype=torch.long)
        corres_tags = torch.tensor(
            [f.corres_tag for f in features], dtype=torch.long)
        rel_tags = torch.tensor(
            [f.rel_tag for f in features], dtype=torch.long)

        tensors = [input_ids, attention_mask, seq_tags,
                   poten_relations, corres_tags, rel_tags]
        return tensors

    @staticmethod
    def collate_fn_test_v1(features):
        """将InputFeatures转换为Tensor
        Args:
            features (List[InputFeatures])
        Returns:
            tensors (List[Tensors])
        """
        input_ids = torch.tensor(
            [f.input_ids for f in features], dtype=torch.long)
        attention_mask = torch.tensor(
            [f.attention_mask for f in features], dtype=torch.long)
        triples = [f.triples for f in features]
        input_tokens = [f.input_tokens for f in features]
        tensors = [input_ids, attention_mask, triples, input_tokens]
        return tensors

    def collate_fn_test(self, features):
        """将InputFeatures转换为Tensor
        Args:
            features (List[InputFeatures])
        Returns:
            tensors (List[Tensors])
        """
        max_len = max([f.text_len for f in features])
        batch_size = len(features)
        batch_input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        batch_attention_masks = torch.zeros(
            batch_size, max_len, dtype=torch.long)

        for i, f in enumerate(features):
            text_len = f.text_len
            batch_input_ids[i, :text_len] = features.input_ids
            batch_attention_masks[i, :max_len] = f.attention_mask

        triples = [f.triples for f in features]
        input_tokens = [f.input_tokens for f in features]
        tensors = [batch_input_ids,
                   batch_attention_masks, triples, input_tokens]
        return tensors

    def get_features(self, data_sign, ex_params):
        """convert InputExamples to InputFeatures
        :param data_sign: 'train', 'val' or 'test'
        """
        print("=*=" * 10)
        print("Loading {} data...".format(data_sign))
        # get features
        cache_path = os.path.join(self.data_dir, "{}.cache.{}".format(
            data_sign, str(self.max_seq_length)))
        if os.path.exists(cache_path) and self.data_cache:
            features = torch.load(cache_path)
        else:
            # get relation to idx
            with open(self.data_dir / f'rel2id.json', 'r', encoding='utf-8') as f_re:
                rel2idx = json.load(f_re)[-1]
            # get examples
            if data_sign in ("train", "val", "test", "pseudo", 'EPO', 'SEO', 'SOO', 'Normal', '1', '2', '3', '4', '5'):
                examples = read_examples(
                    self.data_dir, data_sign=data_sign, rel2idx=rel2idx)
            else:
                raise ValueError(
                    "please notice that the data can only be train/val/test!!")
            features = convert_examples_to_features(self.params, examples, self.tokenizer, rel2idx, data_sign,
                                                    ex_params)
            # save data
            if self.data_cache:
                torch.save(features, cache_path)
        return features

    def get_dataloader(self, data_sign="train", ex_params=None):
        """construct dataloader
        :param data_sign: 'train', 'val' or 'test'
        """
        # InputExamples to InputFeatures
        features = self.get_features(data_sign=data_sign, ex_params=ex_params)
        dataset = FeatureDataset(features)
        print(f"{len(features)} {data_sign} data loaded!")
        print("=*=" * 10)
        # construct dataloader
        # RandomSampler(dataset) or SequentialSampler(dataset)
        if data_sign == "train":
            datasampler = RandomSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.train_batch_size,
                                    collate_fn=self.collate_fn_train)
        elif data_sign == "val":
            datasampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.val_batch_size,
                                    collate_fn=self.collate_fn_test)
        elif data_sign in ("test", "pseudo", 'EPO', 'SEO', 'SOO', 'Normal', '1', '2', '3', '4', '5'):
            datasampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.test_batch_size,
                                    collate_fn=self.collate_fn_test)
        else:
            raise ValueError(
                "please notice that the data can only be train/val/test !!")
        return dataloader


if __name__ == '__main__':
    from utils import Params

    params = Params(corpus_type='WebNLG')
    ex_params = {
        'ensure_relpre': True
    }
    dataloader = CustomDataLoader(params)
    feats = dataloader.get_features(ex_params=ex_params, data_sign='test')
    print(feats[7].input_tokens)
