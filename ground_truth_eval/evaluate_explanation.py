import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse
import json

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from utils.data_utils import load_df
from utils.utils import load_pretrained_config


def get_numbered_list(token_list):
    """
    If there's a duplicate, append number to the second occurence.

    [man, walks, man, man] -> [man, walks, man1, man2]
    """
    token_set = {}
    new_token_list = []
    for x in token_list:
        if x in token_set:
            new_token_list.append(x.lower() + str(token_set[x]))
            token_set[x] += 1
        else:
            new_token_list.append(x.lower())
            token_set[x] = 1
    return new_token_list


def find_common_tokens(pred, gt):
    """
    The main purpose of this function is to find common tokens in two lists.
    However, it should care about duplicates. e.g.
    [man, walks, man], [the, man, walks]  ->  common: [man, walks]
    [man, man], [the, man, walks, man]    ->  common: [man, walks, man]

    Therefore, normal set intersection logic will not work.

    TODO: 2 out of 3 problem.
        sentence: man man man
        pred: *man* man *man* -> [man, man]
        gt: *man* *man* man -> [man, man]

        the common token should be [man], only the first occurence. But now,
        we cannot account for that.

        NOTE: This is very very rare, and not sure if this happens in the dataset.
    """
    numbered_pred = get_numbered_list(pred)
    numbered_gt = get_numbered_list(gt)

    return set(numbered_pred) & set(numbered_gt)


def compute_f1(pred_rationale, gt_rationale):
    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_rationale) == 0 or len(gt_rationale) == 0:
        return [int(pred_rationale == gt_rationale)] * 3

    common_tokens = find_common_tokens(pred_rationale, gt_rationale)

    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0, 0, 0

    prec = len(common_tokens) / len(pred_rationale)
    rec = len(common_tokens) / len(gt_rationale)
    f1 = 2 * (prec * rec) / (prec + rec)

    return prec, rec, f1


def compute_score(gt_rationale1, gt_rationale2, pred_rationale1, pred_rationale2):
    p1, r1, f1_1 = compute_f1(pred_rationale1, gt_rationale1)
    p2, r2, f1_2 = compute_f1(pred_rationale2, gt_rationale2)

    return [p1, r1, f1_1, p2, r2, f1_2]


def jaccard_sim(first, second):
    numbered_first = set(get_numbered_list(first))
    numbered_second = set(get_numbered_list(second))

    if len(numbered_first | numbered_second) == 0:
        return 0
    return len(numbered_first & numbered_second) / len(numbered_first | numbered_second)


def interaction_f1(gt_rationales,
                   pred_rationales,
                   skip_intra_rationale=False,
                   max_only=False,
                   topk=None):
    """
    gt_rationales: list of list of 2 tuple of words:
        [
            [(sent1 group), (sent2 group)],   # 1st interaction rationale
            [(sent1 group), (sent2 group)],   # 2nd interaction rationale
            ...
        ]
    pred_rationales: same format as gt_rationales.
    """
    precision = []
    for p in pred_rationales[:topk]:
        if skip_intra_rationale and (len(p[0]) == 0 or len(p[1]) == 0):
            # if the explanation is not cross-sentence, continue
            continue
        similarities = []
        for g in gt_rationales:
            similarities.append(compute_f1(p[0], g[0])[-1] * compute_f1(p[1], g[1])[-1])
        precision.append(max(similarities))

    if len(precision) == 0:
        return 0
    if max_only:
        return max(precision)
    return sum(precision) / len(precision)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_name', type=str, default='bert-base')
    parser.add_argument('--data_root', type=str, default='data/e-SNLI')
    parser.add_argument('--mode', type=str, default='test', choices=['dev', 'test'])
    parser.add_argument('--how', type=str, default='union', choices=['vote', 'union'])
    parser.add_argument('--explainer', type=str, default='arch')
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--baseline_token', type=str, default='[MASK]')
    parser.add_argument('--metric',
                        type=str,
                        default='token_f1',
                        choices=['token_f1', 'interaction_f1', 'interaction_f1-max'])
    parser.add_argument('--do_cross_merge', action='store_true')
    parser.add_argument('--skip_neutral', action='store_true')
    parser.add_argument('--only_correct', action='store_true')
    parser.add_argument('--test_annotator', type=int, default=-1)
    args = parser.parse_args()

    config = load_pretrained_config(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config['model_card'])
    # NOTE: tokenizing the rationales, since the explanations are also on subwords.
    # NOTE: not needed, since we merge token in process_tokens.
    data = load_df(args.data_root,
                   args.how,
                   mode=args.mode,
                   rationale_format=args.metric.split('_')[0])
    sent, gt_rationale, label = data
    if isinstance(gt_rationale, tuple) and len(gt_rationale) == 2:
        gt_rationale = list(zip(*gt_rationale))
    data = (gt_rationale, label)

    explanation_fname = (f'explanations/{args.model_name}_{args.explainer}'
                         f'_{args.mode}_BT={args.baseline_token}')
    if args.test_annotator > 0:
        args.only_correct = False
        explanation_fname = f'explanations/annotator{args.test_annotator}'
    if args.metric == 'token_f1':
        # load from explanations that have token rationales:
        # format:
        # {
        #     "pred_label": "contradiction",
        #     "premise_rationales": [
        #         "choir",
        #         "songs"
        #     ],
        #     "hypothesis_rationales": [
        #         "has",
        #         "cracks",
        #         "the",
        #         "ceiling"
        #     ]
        # }
        explanation_fname += '_token'

    elif 'interaction_f1' in args.metric:
        # load from explanation that have interaction rationales:
        # format:
        # {
        #      "pred_label": "contradiction",
        #      "pred_rationales": [
        #           [("choir", "song"), ("ceiling")],
        #           [("choir", "song"), ("cracks")]
        #      ]
        # }
        explanation_fname += '_interaction'
    if args.do_cross_merge:
        explanation_fname += '_X'

    print('loading from:', explanation_fname + '.json')
    with open(explanation_fname + '.json', 'r') as f:
        explanations = json.load(f)

    run(data, explanations, args)


def run(data, explanations, args, verbose=False):
    scores = []
    pbar = zip(*data, explanations)
    if verbose:
        pbar = tqdm(pbar, total=len(explanations))
    for gt_rationale, label, exp in pbar:
        if args.skip_neutral:
            if label == 'neutral':
                continue
        if args.only_correct:
            if label != exp['pred_label']:
                continue
        if args.metric == 'token_f1':
            scores.append(
                compute_score(gt_rationale[0], gt_rationale[1], exp['premise_rationales'],
                              exp['hypothesis_rationales']))
        elif 'interaction_f1' in args.metric:
            if len(gt_rationale) == 0:  # this might happen with vote
                continue
            scores.append([
                interaction_f1(gt_rationale,
                               exp['pred_rationales'],
                               skip_intra_rationale=False,
                               max_only='max' in args.metric,
                               topk=k) for k in range(1, args.topk + 1)
            ])
        else:
            raise NotImplementedError

    scores = np.array(scores)

    if args.metric == 'token_f1':

        premise_precision = scores[:, 0].mean() * 100
        premise_recall = scores[:, 1].mean() * 100
        premise_f1 = scores[:, 2].mean() * 100
        hypothesis_precision = scores[:, 3].mean() * 100
        hypothesis_recall = scores[:, 4].mean() * 100
        hypothesis_f1 = scores[:, 5].mean() * 100

        print('premise:')
        print('\tprecision\trecall\tf1')
        print(f'\t{premise_precision:.2f}\t{premise_recall:.2f}\t{premise_f1:.2f}')
        print()
        print('hypothesis:')
        print('\tprecision\trecall\tf1')
        print(
            f'\t{hypothesis_precision:.2f}\t{hypothesis_recall:.2f}\t{hypothesis_f1:.2f}')

        results = pd.DataFrame({
            'precision': {
                'premise': premise_precision,
                'hypothesis': hypothesis_precision,
            },
            'recall': {
                'premise': premise_recall,
                'hypothesis': hypothesis_recall,
            },
            'f1': {
                'premise': premise_f1,
                'hypothesis': hypothesis_f1,
            },
        })
    elif 'interaction_f1' in args.metric:
        results = pd.DataFrame(scores.mean(0),
                               columns=[f'interaction_f1'],
                               index=range(1, args.topk + 1))

    save_name = f'eval_results/{args.model_name}_{args.explainer}_{args.metric}_{args.mode}_{args.how}_{args.topk}_BT={args.baseline_token}'
    print(save_name)
    print(results)
    if args.test_annotator > 0:
        save_name = f'eval_results/annotator{args.test_annotator}_{args.metric}_{args.mode}_{args.how}'

    if args.do_cross_merge:
        save_name += '_X'
    if args.skip_neutral:
        save_name += '_skip-neutral'
    if args.only_correct:
        save_name += '_only-correct'
    results.to_csv(f'{save_name}.csv', index=False)


if __name__ == "__main__":
    main()