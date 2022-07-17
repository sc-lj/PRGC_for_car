# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import random
from multiprocessing import Pool
import functools
import re
import numpy as np
from collections import defaultdict
from itertools import chain

from utils import Label2IdxSub, Label2IdxObj


class InputExample(object):
    """a single set of samples of data
    """

    def __init__(self, text, en_pair_list, re_list, rel2ens, pos_list=None, rel2ens2pos=None):
        self.text = text
        self.en_pair_list = en_pair_list
        self.re_list = re_list
        self.rel2ens = rel2ens
        self.pos_list = pos_list
        self.rel2ens2pos = rel2ens2pos


class InputFeatures(object):
    """
    Desc:
        a single set of features of data
    """

    def __init__(self,
                 input_tokens,
                 input_ids,
                 attention_mask,
                 seq_tag=None,
                 corres_tag=None,
                 relation=None,
                 triples=None,
                 rel_tag=None,
                 text_len=None
                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.seq_tag = seq_tag
        self.corres_tag = corres_tag
        self.relation = relation
        self.triples = triples
        self.rel_tag = rel_tag
        self.text_len = text_len


def read_examples_v1(data_dir, data_sign, rel2idx):
    """load data to InputExamples
    """
    examples = []

    # read src data
    with open(data_dir / f'{data_sign}_triples.json', "r", encoding='utf-8') as f:
        data = json.load(f)
        for sample in data:
            text = sample['text']
            rel2ens = defaultdict(list)
            en_pair_list = []
            re_list = []

            for triple in sample['triple_list']:
                en_pair_list.append([triple[0], triple[-1]])
                re_list.append(rel2idx[triple[1]])
                rel2ens[rel2idx[triple[1]]].append((triple[0], triple[-1]))
            example = InputExample(
                text=text, en_pair_list=en_pair_list, re_list=re_list, rel2ens=rel2ens)
            examples.append(example)
    print("InputExamples:", len(examples))
    return examples


def read_examples(data_dir, data_sign, rel2idx):
    """load data to InputExamples
    """
    examples = []

    # read src data
    with open(data_dir / f'{data_sign}.json', "r", encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            sample = json.loads(line)
            text = sample['text']
            rel2ens = defaultdict(list)
            rel2ens2pos = defaultdict(list)
            en_pair_list = []
            re_list = []
            pos_list = []
            subjects = sample['h']
            sub_text = subjects['name']
            sub_pos = subjects['pos']

            objects = sample['t']
            obj_text = objects['name']
            obj_pos = objects['pos']

            relation = sample['relation']
            en_pair_list.append([sub_text, obj_text])
            re_list.append(rel2idx[relation])
            pos_list.append([sub_pos, obj_pos])
            rel2ens[rel2idx[relation]].append((sub_text, obj_text))
            rel2ens2pos[rel2idx[relation]].append((sub_pos, obj_pos))
            example = InputExample(
                text=list(text), en_pair_list=en_pair_list, re_list=re_list, rel2ens=rel2ens, pos_list=pos_list, rel2ens2pos=rel2ens2pos)
            examples.append(example)
    print("InputExamples:", len(examples))
    return examples


def find_head_idx(source, target):
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1


def _get_so_head_v1(en_pair, tokenizer, text_tokens, positions):
    sub_pos = positions[0]
    obj_pos = positions[1]
    # sub = tokenizer.tokenize(en_pair[0])
    # obj = tokenizer.tokenize(en_pair[1])
    sub = list(en_pair[0])
    obj = list(en_pair[1])
    if text_tokens[sub_pos[0]:sub_pos[1]] != sub:
        sub_head = find_head_idx(source=text_tokens, target=sub)
    else:
        sub_head = sub_pos[0]

    if sub == obj:
        obj_head = find_head_idx(
            source=text_tokens[sub_head + len(sub):], target=obj)
        if obj_head != -1:
            obj_head += sub_head + len(sub)
        else:
            obj_head = obj_pos[0]
    else:
        if text_tokens[obj_pos[0]:obj_pos[1]] != obj:
            obj_head = find_head_idx(source=text_tokens, target=obj)
        else:
            obj_head = obj_pos[0]

    return sub_head, obj_head, sub, obj


def _get_so_head(en_pair, tokenizer, text_tokens):
    sub = tokenizer.tokenize(en_pair[0])
    obj = tokenizer.tokenize(en_pair[1])
    sub_head = find_head_idx(source=text_tokens, target=sub)
    if sub == obj:
        obj_head = find_head_idx(
            source=text_tokens[sub_head + len(sub):], target=obj)
        if obj_head != -1:
            obj_head += sub_head + len(sub)
        else:
            obj_head = sub_head
    else:
        obj_head = find_head_idx(source=text_tokens, target=obj)
    return sub_head, obj_head, sub, obj


def convert_v1(example: InputExample, max_text_len: int, tokenizer, rel2idx, data_sign, ex_params):
    """转换函数
    Args:
        example (_type_): 一个样本示例
        max_text_len (_type_): 样本的最大长度
        tokenizer (_type_): _description_
        rel2idx (dict): 关系的索引
        data_sign (_type_): 数据集
        ex_params (_type_): 额外的参数
    Returns:
        _type_: _description_
    """
    text_tokens = tokenizer.tokenize(example.text)
    # cut off
    if len(text_tokens) > max_text_len:
        text_tokens = text_tokens[:max_text_len]

    # token to id
    input_ids = tokenizer.convert_tokens_to_ids(text_tokens)
    attention_mask = [1] * len(input_ids)
    # zero-padding up to the sequence length
    if len(input_ids) < max_text_len:
        pad_len = max_text_len - len(input_ids)
        # token_pad_id=0
        input_ids += [0] * pad_len
        attention_mask += [0] * pad_len

    # train data
    if data_sign == 'train':
        # construct tags of correspondence and relation
        # subject和object相关性 target
        corres_tag = np.zeros((max_text_len, max_text_len))
        rel_tag = len(rel2idx) * [0]
        for en_pair, rel in zip(example.en_pair_list, example.re_list):
            # get sub and obj head
            sub_head, obj_head, _, _ = _get_so_head(
                en_pair, tokenizer, text_tokens)
            # construct relation tag
            rel_tag[rel] = 1
            # 只将head 的index标记为1
            if sub_head != -1 and obj_head != -1:
                corres_tag[sub_head][obj_head] = 1

        sub_feats = []
        # positive samples，标记subject和object的序列
        for rel, en_ll in example.rel2ens.items():
            # init
            tags_sub = max_text_len * [Label2IdxSub['O']]
            tags_obj = max_text_len * [Label2IdxSub['O']]
            for en in en_ll:
                # get sub and obj head
                sub_head, obj_head, sub, obj = _get_so_head(
                    en, tokenizer, text_tokens)
                if sub_head != -1 and obj_head != -1:
                    if sub_head + len(sub) <= max_text_len:
                        tags_sub[sub_head] = Label2IdxSub['B-H']
                        tags_sub[sub_head + 1:sub_head +
                                 len(sub)] = (len(sub) - 1) * [Label2IdxSub['I-H']]
                    if obj_head + len(obj) <= max_text_len:
                        tags_obj[obj_head] = Label2IdxObj['B-T']
                        tags_obj[obj_head + 1:obj_head +
                                 len(obj)] = (len(obj) - 1) * [Label2IdxObj['I-T']]
            seq_tag = [tags_sub, tags_obj]

            # sanity check
            assert len(input_ids) == len(tags_sub) == len(tags_obj) == len(
                attention_mask) == max_text_len, f'length is not equal!!'
            sub_feats.append(InputFeatures(
                input_tokens=text_tokens,
                input_ids=input_ids,
                attention_mask=attention_mask,
                corres_tag=corres_tag,
                seq_tag=seq_tag,
                relation=rel,
                rel_tag=rel_tag
            ))
        # relation judgement ablation
        if not ex_params['ensure_rel']:
            # negative samples, 采样一些负样本的关系数据集
            neg_rels = set(rel2idx.values()).difference(set(example.re_list))
            neg_rels = random.sample(neg_rels, k=ex_params['num_negs'])
            for neg_rel in neg_rels:
                # init，针对关系的负样本，只对subject和object的序列全部置为O，其他的沿用正样本的数据
                seq_tag = max_text_len * [Label2IdxSub['O']]
                # sanity check
                assert len(input_ids) == len(seq_tag) == len(
                    attention_mask) == max_text_len, f'length is not equal!!'
                seq_tag = [seq_tag, seq_tag]
                sub_feats.append(InputFeatures(
                    input_tokens=text_tokens,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    corres_tag=corres_tag,
                    seq_tag=seq_tag,
                    relation=neg_rel,
                    rel_tag=rel_tag
                ))
    # val and test data
    else:
        triples = []
        for rel, en in zip(example.re_list, example.en_pair_list):
            # get sub and obj head
            sub_head, obj_head, sub, obj = _get_so_head(
                en, tokenizer, text_tokens)
            if sub_head != -1 and obj_head != -1:
                h_chunk = ('H', sub_head, sub_head + len(sub))
                t_chunk = ('T', obj_head, obj_head + len(obj))
                triples.append((h_chunk, t_chunk, rel))
        sub_feats = [
            InputFeatures(
                input_tokens=text_tokens,
                input_ids=input_ids,
                attention_mask=attention_mask,
                triples=triples
            )
        ]

    # get sub-feats
    return sub_feats


def convert(example: InputExample, max_text_len: int, tokenizer, rel2idx, data_sign, ex_params):
    """转换函数
    Args:
        example (_type_): 一个样本示例
        max_text_len (_type_): 样本的最大长度
        tokenizer (_type_): _description_
        rel2idx (dict): 关系的索引
        data_sign (_type_): 数据集
        ex_params (_type_): 额外的参数
    Returns:
        _type_: _description_
    """
    text_tokens = example.text
    offset_len = 0
    # cut off
    if len(text_tokens) > max_text_len-2:
        min_index = min(sum(sum(example.pos_list, []), []))
        max_index = max(sum(sum(example.pos_list, []), []))
        min_offset = len(text_tokens)-(max_text_len-2)
        if min_offset < min_index-10:
            offset_len = random.choice(
                list(range(min_offset, max(min_offset, min_index-10))))
            text_tokens = text_tokens[offset_len:]
        else:
            offset_len = random.choice(
                list(range(max_index+10, len(text_tokens)-min_offset)))
            text_tokens = text_tokens[:offset_len]
            offset_len = 0
    text_tokens = ["[CLS]"] + text_tokens + ['[SEP]']
    text_len = len(text_tokens)
    # 使用文本长度作为最大长度
    max_text_len = text_len
    # token to id
    input_ids = tokenizer.convert_tokens_to_ids(text_tokens)
    attention_mask = [1] * len(input_ids)

    # train data
    if data_sign == 'train':
        # construct tags of correspondence and relation
        # subject和object相关性 target
        corres_tag = np.zeros((max_text_len, max_text_len))
        rel_tag = len(rel2idx) * [0]
        for en_pair, rel, pos in zip(example.en_pair_list, example.re_list, example.pos_list):
            # get sub and obj head
            # [CLS]
            sub_head, obj_head = pos[0][0]+1-offset_len, pos[1][0]+1-offset_len
            assert "".join(text_tokens[sub_head:sub_head + len(en_pair[0])]) == en_pair[0], "".join(
                text_tokens[obj_head:obj_head + len(en_pair[1])]) == en_pair[1]
            # construct relation tag
            rel_tag[rel] = 1
            # 只将head 的index标记为1
            if sub_head != -1 and obj_head != -1:
                corres_tag[sub_head][obj_head] = 1

        sub_feats = []
        rel2ens2pos = example.rel2ens2pos
        # positive samples，标记subject和object的序列
        for rel, en_ll in example.rel2ens.items():
            # init
            tags_sub = max_text_len * [Label2IdxSub['O']]
            tags_obj = max_text_len * [Label2IdxSub['O']]
            pos_ll = rel2ens2pos[rel]
            for en, pos in zip(*(en_ll, pos_ll)):
                # get sub and obj head
                sub_head, obj_head = pos[0][0]+1 - \
                    offset_len, pos[1][0]+1-offset_len
                assert "".join(text_tokens[sub_head:sub_head + len(en_pair[0])]) == en_pair[0], "".join(
                    text_tokens[obj_head:obj_head + len(en_pair[1])]) == en_pair[1]
                sub = list(en[0])
                obj = list(en[1])
                if sub_head != -1 and obj_head != -1:
                    if sub_head + len(sub) <= max_text_len:
                        tags_sub[sub_head] = Label2IdxSub['B-H']
                        tags_sub[sub_head + 1:sub_head +
                                 len(sub)] = (len(sub) - 1) * [Label2IdxSub['I-H']]
                    if obj_head + len(obj) <= max_text_len:
                        tags_obj[obj_head] = Label2IdxObj['B-T']
                        tags_obj[obj_head + 1:obj_head +
                                 len(obj)] = (len(obj) - 1) * [Label2IdxObj['I-T']]
            seq_tag = [tags_sub, tags_obj]

            # sanity check
            assert len(input_ids) == len(tags_sub) == len(tags_obj) == len(
                attention_mask) == max_text_len, f'length is not equal!!'
            sub_feats.append(InputFeatures(
                input_tokens=text_tokens,
                input_ids=input_ids,
                attention_mask=attention_mask,
                corres_tag=corres_tag,
                seq_tag=seq_tag,
                relation=rel,
                rel_tag=rel_tag,
                text_len=text_len
            ))
        # relation judgement ablation
        if not ex_params['ensure_rel']:
            # negative samples, 采样一些负样本的关系数据集
            neg_rels = set(rel2idx.values()).difference(set(example.re_list))
            neg_rels = random.sample(neg_rels, k=ex_params['num_negs'])
            for neg_rel in neg_rels:
                # init，针对关系的负样本，只对subject和object的序列全部置为O，其他的沿用正样本的数据
                seq_tag = max_text_len * [Label2IdxSub['O']]
                # sanity check
                assert len(input_ids) == len(seq_tag) == len(
                    attention_mask) == max_text_len, f'length is not equal!!'
                seq_tag = [seq_tag, seq_tag]
                sub_feats.append(InputFeatures(
                    input_tokens=text_tokens,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    corres_tag=corres_tag,
                    seq_tag=seq_tag,
                    relation=neg_rel,
                    rel_tag=rel_tag,
                    text_len=text_len
                ))
    # val and test data
    else:
        triples = []
        for rel, en, pos in zip(example.re_list, example.en_pair_list, example.pos_list):
            # get sub and obj head
            sub_head, obj_head = pos[0][0]+1-offset_len, pos[1][0]+1-offset_len
            assert "".join(text_tokens[sub_head:sub_head + len(en[0])]) == en[0], "".join(
                text_tokens[obj_head:obj_head + len(en[1])]) == en[1]
            sub = list(en[0])
            obj = list(en[1])
            if sub_head != -1 and obj_head != -1:
                h_chunk = ('H', sub_head, sub_head + len(sub))
                t_chunk = ('T', obj_head, obj_head + len(obj))
                triples.append((h_chunk, t_chunk, rel))
        sub_feats = [
            InputFeatures(
                input_tokens=text_tokens,
                input_ids=input_ids,
                attention_mask=attention_mask,
                triples=triples,
                text_len=text_len
            )
        ]

    # get sub-feats
    return sub_feats


def convert_examples_to_features(params, examples, tokenizer, rel2idx, data_sign, ex_params):
    """convert examples to features.
    :param examples (List[InputExamples])
    """
    max_text_len = params.max_seq_length
    # multi-process
    # with Pool(10) as p:
    #     convert_func = functools.partial(convert, max_text_len=max_text_len, tokenizer=tokenizer, rel2idx=rel2idx,
    #                                      data_sign=data_sign, ex_params=ex_params)
    #     features = p.map(func=convert_func, iterable=examples)
    # return list(chain(*features))
    features = []
    for example in examples:
        feature = convert(example, max_text_len=max_text_len, tokenizer=tokenizer, rel2idx=rel2idx,
                          data_sign=data_sign, ex_params=ex_params)
        features.extend(feature)
    return features
