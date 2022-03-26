import itertools

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils.data_utils import perturb_text


class NaiveExplainer:

    def __init__(self, model_name, device='cpu'):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
            device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device

    def explain(self, premise, hypothesis, k, baseline_token='[MASK]'):
        """
        explain with naive occlusion: pairwise interaction
        """
        full_inp = self.tokenizer(premise, text_pair=hypothesis, return_tensors='pt')
        logits = torch.softmax(self.model(**full_inp.to(self.device)).logits[0], dim=-1)
        orig_confidence, target_class = logits.max(-1)
        target_class = target_class.item()
        orig_confidence = orig_confidence.item()

        # premise first
        perturbed_premise = perturb_text(premise, baseline_token=baseline_token)
        pre_confidences = []
        for sent, _ in perturbed_premise:
            inp = self.tokenizer(sent, text_pair=hypothesis, return_tensors='pt')
            conf = torch.softmax(self.model(**inp.to(self.device)).logits[0],
                                 dim=-1)[target_class].item()
            conf = orig_confidence - conf
            pre_confidences.append(conf)

        # perturb hypothesis
        perturbed_hyp = perturb_text(hypothesis, baseline_token=baseline_token)
        hyp_confidences = []
        for sent, _ in perturbed_hyp:
            inp = self.tokenizer(premise, text_pair=sent, return_tensors='pt')
            conf = torch.softmax(self.model(**inp.to(self.device)).logits[0],
                                 dim=-1)[target_class].item()
            conf = orig_confidence - conf
            hyp_confidences.append(conf)

        pre_topk = torch.tensor(pre_confidences).topk(
            k=min(k, len(pre_confidences)))[1].tolist()
        hyp_topk = torch.tensor(hyp_confidences).topk(
            k=min(k, len(hyp_confidences)))[1].tolist()
        topk_premises = [perturbed_premise[i] for i in pre_topk]
        topk_hyp = [perturbed_hyp[i] for i in hyp_topk]

        topk_pairs = list(itertools.product(topk_premises, topk_hyp))
        final_confidences = []
        for pair in topk_pairs:
            inp = self.tokenizer(pair[0][0], text_pair=pair[1][0], return_tensors='pt')
            conf = torch.softmax(self.model(**inp.to(self.device)).logits[0],
                                 dim=-1)[target_class].item()
            conf = orig_confidence - conf
            if conf > 0:  # if confidence rises:
                continue
            final_confidences.append(conf)

        return target_class, orig_confidence, final_confidences, topk_pairs

    def analyze_result(self, premise, hypothesis, prediction, confidence, conf_drops,
                       perturbations):
        print('premise:', premise)
        print('hypothesis:', hypothesis)
        print()
        class_map = ['contradiction', 'entailment', 'neutral']
        print(
            f'original prediction was {class_map[prediction]} / with confidence: {confidence}\n'
        )
        conf_drops = np.array(conf_drops)
        idx = conf_drops.argsort(axis=0)[::-1]
        conf_drops = conf_drops[idx]
        perturbations = [perturbations[i] for i in idx]
        pert_sents = [(s[0][0], s[1][0]) for s in perturbations]
        pert_pre_words = set([s[0][1] for s in perturbations])
        pert_hyp_words = set([s[1][1] for s in perturbations])

        for i, (pert, conf) in enumerate(zip(pert_sents, conf_drops), 1):
            print(f'{i}. {pert} | -{conf}')

        print()
        print('premise:', pert_pre_words, '\nhypothesis:', pert_hyp_words)
