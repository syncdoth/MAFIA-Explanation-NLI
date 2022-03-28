from itertools import chain
import string
import sys

sys.path.insert(0, '/data/schoiaj/repos/nli_explain')

import argparse
import json

import torch
from explainers.archipelago.get_explainer import ArchExplainerInterface
from explainers.integrated_hessians.IH_explainer import IHBertExplainer
from explainers.naive_explainer import NaiveExplainer
from tqdm import tqdm
from utils.data_utils import load_df


def process_token(tokens, idx):
    """
    merge subwords.
    """
    # TODO: either do this or use tokenizer: not both.
    # TODO: duplicate problem
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
                        choices=[
                            'arch', 'cross_arch', 'arch_pair', 'cross_arch_pair',
                            'naive_occlusion', 'naive_interaction_occlusion', 'IH'
                        ])
    parser.add_argument('--baseline_token', type=str, default='[MASK]')
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--format',
                        type=str,
                        default='token',
                        choices=['token', 'interaction'])
    parser.add_argument('--do_cross_merge', action='store_true')
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
                              topk=args.topk,
                              do_cross_merge=args.do_cross_merge)
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
        explain_kwargs = dict(batch_size=24,
                              num_samples=128,
                              use_expectation=False,
                              do_cross_merge=args.do_cross_merge)
    else:
        raise NotImplementedError

    all_explanations = run(sent_data, explainer, args, **explain_kwargs)
    with open(
            f'explanations/{args.model_name}_{args.explainer}_{args.topk}_{args.mode}_BT={args.baseline_token}_{args.format}.json',
            'w') as f:
        json.dump(all_explanations, f, indent=4)


def run(data, explainer, args, **explain_kwargs):
    inv_label_map = explainer.get_label_map(inv=True)

    all_explanations = []
    pbar = tqdm(zip(*data), total=len(data[0]))
    for premise, hypothesis in pbar:
        explanation, tokens, pred = explainer.explain(premise, hypothesis,
                                                      **explain_kwargs)

        topk_exp = [
            k for k, _ in sorted(explanation.items(), key=lambda x: x[1], reverse=True)
        ][:args.topk]
        sep_position = tokens.index(explainer.tokenizer.sep_token)

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
