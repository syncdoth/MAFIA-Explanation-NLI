import numpy as np
from application_utils.common_utils import get_efficient_mask_indices
import pickle
import copy


class TextXformer:
    # note: this xformer is not the transformer from Vaswani et al., 2017

    def __init__(self, inputs, baseline_ids):
        """
        inputs: dict of huggingface tokenizer output. assume each array have
            shape [T,].
        baseline_ids: a np array of shape [T,]
        """
        self.input = inputs
        self.input_ids = inputs['input_ids']
        self.baseline_ids = baseline_ids
        self.num_features = len(self.input_ids)

    def simple_xform(self, inst):
        mask_indices = np.argwhere(inst == True).flatten()
        id_list = list(self.baseline_ids)
        for i in mask_indices:
            id_list[i] = self.input_ids[i]
        return id_list

    def efficient_xform(self, inst):
        mask_indices, base, change = get_efficient_mask_indices(
            inst, self.baseline_ids, self.input_ids)
        for i in mask_indices:
            base[i] = change[i]
        return base

    def __call__(self, inst):
        id_list = self.efficient_xform(inst)
        return id_list

    def process_batch_ids(self, batch_ids):
        """
        batch_ids: list of numpy arrays, each is a input_ids
        """
        batch = {}
        for k, v in self.input.items():
            if k == 'input_ids':
                batch[k] = np.array(batch_ids)
            else:
                batch[k] = np.repeat(v[np.newaxis, :], len(batch_ids), axis=0)
        return batch


def process_stop_words(explanation, tokens, strip_first_last=True):
    explanation = copy.deepcopy(explanation)
    tokens = copy.deepcopy(tokens)
    stop_words = set([
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "he",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "that",
        "the",
        "to",
        "was",
        "were",
        "will",
        "with",
        "s",
        "ll",
    ])
    for i, token in enumerate(tokens):
        if token in stop_words:
            if (i,) in explanation:
                explanation[(i,)] = 0.0

    if strip_first_last:
        explanation.pop((0,))
        explanation.pop((len(tokens) - 1,))
        tokens = tokens[1:-1]
    return explanation, tokens


def get_input_baseline_ids(text, baseline_token, tokenizer, text_pair=None):
    """
    return: dict of numpy arrays
    """
    inputs = prepare_huggingface_data(text, tokenizer, text_pair=text_pair)
    inputs = {k: v[0] for k, v in inputs.items()}
    baseline_id = prepare_huggingface_data(baseline_token, tokenizer)['input_ids'][0][1]

    # make baseline inputs
    input_ids = inputs['input_ids']
    tokenizer.sep_token_id
    baseline_ids = np.where(
        np.isin(input_ids, [tokenizer.sep_token_id, tokenizer.cls_token_id]), input_ids,
        baseline_id)

    return inputs, baseline_ids


def get_token_list(sentence, tokenizer):
    if isinstance(sentence, str):
        X = prepare_huggingface_data(sentence, tokenizer)
        batch_ids = X["input_ids"][0]
    else:
        batch_ids = sentence
    tokens = tokenizer.convert_ids_to_tokens(batch_ids)
    return tokens


def get_sst_sentences(split="test", path="../../downloads/sst_data/sst_trees.pickle"):
    with open(path, "rb") as handle:
        sst_trees = pickle.load(handle)

    data = []
    for s in range(len(sst_trees[split])):
        sst_item = sst_trees[split][s]

        sst_item = sst_trees[split][s]
        sentence = sst_item[0]
        data.append(sentence)
    return data


def prepare_huggingface_data(sentences, tokenizer, text_pair=None):
    encoded_sentence = tokenizer(sentences,
                                 text_pair=text_pair,
                                 padding='longest',
                                 return_tensors='np',
                                 return_token_type_ids=True,
                                 return_attention_mask=True)
    return encoded_sentence
