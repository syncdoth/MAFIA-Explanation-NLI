import sys

sys.path.insert(0, '/data/schoiaj/repos/nli_explain')

import argparse
import json
from itertools import chain
import string

import torch
from explainers.archipelago.application_utils.text_utils import *
from explainers.archipelago.application_utils.text_utils_torch import BertWrapperTorch
from explainers.archipelago.explainer import Archipelago, CrossArchipelago
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils.data_utils import load_df
from utils.utils import load_pretrained_config


def process_token(tokens, idx):
    """
    merge subwords.
    """
    # TODO: either do this or use tokenizer: not both.
    tok = tokens[idx]
    # backwards
    if tok.startswith('##'):
        while tok.startswith('##'):
            tok = tokens[idx - 1] + tok[2:]
            idx -= 1
        return tok
    if idx + 1 == len(tokens):
        return tok
    # forwards
    while tokens[idx + 1].startswith('##'):
        tok = tok + tokens[idx + 1][2:]
        idx += 1
        if idx + 1 == len(tokens):
            return tok
    return tok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_name',
                        type=str,
                        default='bert-base',
                        choices=['bert-base', 'roberta-large'])
    parser.add_argument('--data_root', type=str, default='data/e-SNLI')
    parser.add_argument('--mode', type=str, default='test', choices=['dev', 'test'])
    parser.add_argument('--explainer',
                        type=str,
                        default='arch',
                        choices=['arch', 'cross_arch'])
    parser.add_argument('--baseline_token', type=str, default='[MASK]')
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--format',
                        type=str,
                        default='token',
                        choices=['token', 'interaction'])
    args = parser.parse_args()

    data = load_df(args.data_root, mode=args.mode)
    sent_data, gt_rationale, labels = data
    del data, gt_rationale, labels  # unused

    config = load_pretrained_config(args.model_name)
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(config['model_card'])
    model = AutoModelForSequenceClassification.from_pretrained(config['model_card'])
    model_wrapper = BertWrapperTorch(model, device)

    all_explanations = run(sent_data, tokenizer, model_wrapper, config['label_map'], args)
    with open(
            f'explanations/{args.model_name}_{args.explainer}_{args.topk}_{args.mode}_{args.format}_BT={args.baseline_token}.json',
            'w') as f:
        json.dump(all_explanations, f, indent=4)


def run(data, tokenizer, model_wrapper, label_map, args):
    inv_label_map = {idx: label for label, idx in label_map.items()}

    all_explanations = []
    pbar = tqdm(zip(*data), total=len(data[0]))
    for premise, hypothesis in pbar:
        text_inputs, baseline_ids = get_input_baseline_ids(premise,
                                                           args.baseline_token,
                                                           tokenizer,
                                                           text_pair=hypothesis)
        _text_inputs = {k: v[np.newaxis, :] for k, v in text_inputs.items()}
        pred = np.argmax(model_wrapper(**_text_inputs)[0])
        xf = TextXformer(text_inputs, baseline_ids, sep_token_id=tokenizer.sep_token_id)

        if args.explainer == 'arch':
            apgo = Archipelago(model_wrapper,
                               data_xformer=xf,
                               output_indices=pred,
                               batch_size=args.batch_size)
        elif args.explainer == 'cross_arch':
            apgo = CrossArchipelago(model_wrapper,
                                    data_xformer=xf,
                                    output_indices=pred,
                                    batch_size=args.batch_size)
        else:
            raise NotImplementedError

        explanation = apgo.explain(top_k=args.topk, use_embedding=True)
        tokens = get_token_list(text_inputs['input_ids'], tokenizer)
        explanation, tokens = process_stop_words(explanation, tokens)

        topk_exp = [
            k for k, _ in sorted(explanation.items(), key=lambda x: x[1], reverse=True)
        ][:args.topk]  # TODO: here, topk means sth different
        sep_position = tokens.index(tokenizer.sep_token)

        if args.format == 'token':
            token_idx = set(chain.from_iterable(topk_exp))
            pre_rat = []
            hyp_rat = []
            for idx in token_idx:
                token = process_token(tokens, idx - 1)
                if token in string.punctuation:
                    continue
                if (idx - 1) < sep_position:
                    pre_rat.append(token)
                if (idx - 1) > sep_position and 'bert' in args.model_name:
                    hyp_rat.append(token)
                elif (idx - 1) > sep_position + 1 and 'roberta' in args.model_name:
                    # TODO: why +1 ?
                    hyp_rat.append(token)

            all_explanations.append({
                'pred_label': inv_label_map[pred],
                'premise_rationales': pre_rat,
                'hypothesis_rationales': hyp_rat,
            })
        elif args.format == 'interaction':
            rationales = []
            for interaction in topk_exp:
                pre_group = []
                hyp_group = []
                for idx in interaction:
                    token = process_token(tokens, idx - 1)
                    if (idx - 1) < sep_position:
                        pre_group.append(token)
                    if (idx - 1) > sep_position and 'bert' in args.model_name:
                        hyp_group.append(token)
                    elif (idx - 1) > sep_position + 1 and 'roberta' in args.model_name:
                        hyp_group.append(token)
                rationales.append([tuple(pre_group), tuple(hyp_group)])
            all_explanations.append({
                'pred_label': inv_label_map[pred],
                'pred_rationales': rationales,
            })

        else:
            raise NotImplementedError

    return all_explanations


if __name__ == "__main__":
    main()
