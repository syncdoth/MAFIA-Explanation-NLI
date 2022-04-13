from itertools import chain
import os
import string
import sys

sys.path.insert(0, '/data/schoiaj/repos/nli_explain')

import argparse
import json

import torch
from explainers.archipelago.get_explainer import ArchExplainerInterface
from explainers.integrated_hessians.IH_explainer import IHBertExplainer
from explainers.naive_explain.naive_explainer import NaiveExplainer
from explainers.mask_explain.mask_explainer import MaskExplainer
from tqdm import tqdm
from utils.data_utils import load_df


def get_base_idx(tokens, idx):
    if not tokens[idx].startswith('##'):
        return idx

    while tokens[idx].startswith('##'):
        # no need to check against idx=-1, since subword are always matched.
        idx -= 1
    return idx


def process_token(tokens, idx):
    """
    merge subwords.
    """
    tok = tokens[idx]
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
                        choices=[
                            'arch', 'cross_arch', 'arch_pair', 'cross_arch_pair',
                            'naive_occlusion', 'naive_interaction_occlusion', 'IH',
                            'mask_explain'
                        ])
    parser.add_argument('--baseline_token', type=str, default='[MASK]')
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--arch_int_topk', type=int, default=5)
    parser.add_argument('--format',
                        type=str,
                        default='token',
                        choices=['token', 'interaction'])
    parser.add_argument('--do_cross_merge', action='store_true')
    # mask explainer args
    parser.add_argument('--interaction_order',
                        type=str,
                        help='comma separated values',
                        default='1')
    parser.add_argument('--mask_p', type=float, default=0.5)
    parser.add_argument('--mask_n', type=int, default=10000)
    parser.add_argument('--inverse_mask', action='store_true')
    parser.add_argument('--no_correction', action='store_true')
    args = parser.parse_args()

    data = load_df(args.data_root, mode=args.mode)
    sent_data, gt_rationale, labels = data
    del data, gt_rationale, labels  # unused

    device = torch.device("cuda")
    if 'arch' in args.explainer:
        explainer = ArchExplainerInterface(args.model_name,
                                           device=device,
                                           baseline_token=args.baseline_token,
                                           explainer_class=args.explainer)
        explain_kwargs = dict(batch_size=args.batch_size,
                              topk=args.arch_int_topk,
                              do_cross_merge=args.do_cross_merge)
        args.explainer = f'{args.explainer}-{args.arch_int_topk}'
    elif 'naive' in args.explainer:
        explainer = NaiveExplainer(args.model_name,
                                   device=device,
                                   baseline_token=args.baseline_token,
                                   interaction_occlusion='interaction' in args.explainer)
        explain_kwargs = dict(return_cache=False, do_cross_merge=args.do_cross_merge)
    elif args.explainer == 'IH':
        # NOTE: currently, IH only support Bert
        args.model_name = 'bert-base'
        explainer = IHBertExplainer(args.model_name,
                                    device=device,
                                    baseline_token=args.baseline_token)
        explain_kwargs = dict(batch_size=16,
                              num_samples=128,
                              use_expectation=False,
                              do_cross_merge=args.do_cross_merge,
                              get_cross_effects=True)
    elif args.explainer == 'mask_explain':
        interaction_order = tuple(map(int, args.interaction_order.split(',')))
        # NOTE: mask explainer works only with attention perturbations
        if 'attention' not in args.baseline_token:
            args.baseline_token = f'attention+{args.baseline_token}'
        explainer = MaskExplainer(args.model_name,
                                  device=device,
                                  baseline_token=args.baseline_token)
        explain_kwargs = dict(batch_size=args.batch_size,
                              interaction_order=interaction_order,
                              mask_p=args.mask_p,
                              mask_n=args.mask_n,
                              inverse_mask=args.inverse_mask,
                              no_correction=args.no_correction)
        args.explainer = f'{args.explainer}-{args.interaction_order}-p{args.mask_p}-n{args.mask_n}-inv{int(args.inverse_mask)}'
        args.explainer += 'noCorr' if args.no_correction else ''
    else:
        raise NotImplementedError

    explanation_fname = f'explanations/{args.model_name}_{args.explainer}_{args.topk}_{args.mode}_BT={args.baseline_token}_{args.format}'

    if args.do_cross_merge:
        explanation_fname += '_X'
    if os.path.isfile(f'{explanation_fname}.json'):
        os.remove(f'{explanation_fname}.json')
    with open(f'{explanation_fname}.json', 'a') as write_file:
        write_file.write('[\n')
        run(sent_data, explainer, write_file, args, **explain_kwargs)
        write_file.write(']')


def run(data, explainer, out_file, args, **explain_kwargs):
    inv_label_map = explainer.get_label_map(inv=True)

    pbar = tqdm(enumerate(zip(*data)), total=len(data[0]))
    for i, (premise, hypothesis) in pbar:
        explanation, tokens, pred = explainer.explain(premise, hypothesis,
                                                      **explain_kwargs)

        topk_exp = [
            k for k, v in sorted(explanation.items(), key=lambda x: x[1], reverse=True)
            if v > 0
        ][:args.topk]
        sep_position = tokens.index(explainer.tokenizer.sep_token)

        if args.format == 'token':
            token_idx = set(chain.from_iterable(topk_exp))
            token_idx = sorted(set([get_base_idx(tokens, idx) for idx in token_idx]))
            pre_rat = []
            hyp_rat = []
            for idx in token_idx:
                token = process_token(tokens, idx)
                if token in string.punctuation:
                    continue
                if idx < sep_position:
                    pre_rat.append(token)
                if idx > sep_position and 'bert' in args.model_name:
                    hyp_rat.append(token)
                elif idx > sep_position + 1 and 'roberta' in args.model_name:
                    # TODO: why +1 ?
                    hyp_rat.append(token)

            current_explanation = {
                'pred_label': inv_label_map[pred],
                'premise_rationales': pre_rat,
                'hypothesis_rationales': hyp_rat,
            }
        elif args.format == 'interaction':
            rationales = []
            proc_interaction = set([
                sorted(set([get_base_idx(tokens, idx)
                            for idx in interaction]))
                for interaction in topk_exp
            ])
            for interaction in proc_interaction:
                pre_group = []
                hyp_group = []
                for idx in interaction:
                    token = process_token(tokens, idx)
                    if idx < sep_position:
                        pre_group.append(token)
                    if idx > sep_position and 'bert' in args.model_name:
                        hyp_group.append(token)
                    elif idx > sep_position + 1 and 'roberta' in args.model_name:
                        hyp_group.append(token)
                    rationales.append((pre_group, hyp_group))
            current_explanation = {
                'pred_label': inv_label_map[pred],
                'pred_rationales': rationales,
            }

        else:
            raise NotImplementedError

        json.dump(current_explanation, out_file, indent=4)
        if i == len(data[0]) - 1:
            out_file.write('\n')
        else:
            out_file.write(',\n')


if __name__ == "__main__":
    main()
