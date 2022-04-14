"""
Explain with random masking.
"""
import string
import sys

sys.path.insert(0, '/data/schoiaj/repos/nli_explain')
from itertools import combinations, product

import torch
from tqdm import tqdm

from explainers.base_explainer import ExplainerInterface


class MaskExplainer(ExplainerInterface):

    def __init__(self, model_name, device='cpu', baseline_token='[MASK]'):
        super().__init__(model_name, device, baseline_token)
        if 'attention' in baseline_token:
            self.baseline_token = baseline_token.split('+')[1]
            self.attention_perturbation = True
        else:
            self.attention_perturbation = False
            self.baseline_token_id = self.tokenizer.encode(self.baseline_token,
                                                           add_special_tokens=False)[0]

    def get_masks(self, inputs, mask_p=0.5, mask_n=1000):
        """
        inputs: huggingface tokenizer encoded inputs. Assumes each value is a torch.LongTensor
        mask_p: max amount of tokens masked in the sentence.
        mask_n: number of randomly masked sequence to be generated.
        """
        input_ids = inputs['input_ids']  # [1, T]
        sent_length = input_ids.shape[1]
        sep_idx = torch.where(input_ids[0] == self.tokenizer.sep_token_id)[0]
        random_mask = torch.rand(mask_n, sent_length) <= mask_p

        # set special tokens to 1, so they are not masked
        random_mask[:, [0, sep_idx[0].item(), -1]] = 1

        processed_inputs = {}
        if self.attention_perturbation:
            for k, v in inputs.items():
                if k == 'attention_mask':
                    processed_inputs[k] = random_mask.long()
                else:
                    processed_inputs[k] = v.repeat(mask_n, 1)
        else:
            batch_input_ids = input_ids.repeat(mask_n, 1)
            masked_input_ids = torch.where(random_mask == 0, self.baseline_token_id,
                                           batch_input_ids)
            for k, v in inputs.items():
                if k == input_ids:
                    processed_inputs[k] = masked_input_ids
                else:
                    processed_inputs[k] = v.repeat(mask_n, 1)

        return processed_inputs, random_mask

    def get_batch(self, inputs, batch_size=32, verbose=False):
        num_batches = inputs['input_ids'].shape[0] // batch_size + 1
        iterator = range(num_batches)
        if verbose:
            iterator = tqdm(iterator, total=num_batches, desc="Mask Explainer")
        for idx in iterator:
            batch_idx = slice(idx * batch_size, (idx + 1) * batch_size)
            yield {k: v[batch_idx].to(self.device) for k, v in inputs.items()}

    def get_scores(self,
                   premise,
                   hypothesis,
                   batch_size=32,
                   target_class=None,
                   mask_p=0.5,
                   mask_n=1000):
        inputs = self.tokenizer(premise, text_pair=hypothesis, return_tensors='pt')
        logits = self.model(**inputs.to(self.device)).logits[0]
        pred_score, pred_class = torch.softmax(logits, -1).max(0)
        inputs.to('cpu')
        if target_class is None:
            target_class = pred_class

        masked_inputs, mask = self.get_masks(inputs, mask_p=mask_p, mask_n=mask_n)

        all_scores = []
        for batch in self.get_batch(masked_inputs, batch_size=batch_size, verbose=False):
            with torch.no_grad():
                logits = self.model(**batch).logits
            scores = torch.softmax(logits, -1)[:, target_class].cpu()
            all_scores.append(scores)

        all_scores = torch.cat(all_scores)

        return all_scores, mask, pred_class.item(), pred_score.item()

    def explain(self,
                premise,
                hypothesis,
                batch_size=32,
                target_class=None,
                interaction_order=1,
                top_p=0.5,
                mask_p=0.5,
                mask_n=1000,
                inverse_mask=False,
                no_correction=False):
        assert interaction_order <= 6, 'interaction order > 6 is too slow.'
        tokens = self.tokenizer.tokenize(premise,
                                         pair=hypothesis,
                                         add_special_tokens=True)
        scores, mask, pred_class, pred_score = self.get_scores(premise,
                                                               hypothesis,
                                                               batch_size=batch_size,
                                                               target_class=target_class,
                                                               mask_p=mask_p,
                                                               mask_n=mask_n)
        if inverse_mask:
            mask = ~mask
            scores = pred_score - scores
            skip_indices = set(torch.where(mask.sum(0) == 0)[0].tolist())
        else:
            skip_indices = set(torch.where(mask.sum(0) == mask.shape[0])[0].tolist())

        skip_indices.update(
            [i for i, tok in enumerate(tokens) if tok in string.punctuation])
        valid_indices = sorted(set(range(mask.shape[1])) - skip_indices)

        explanations = {}
        if interaction_order == 1:
            saliencies = (scores.unsqueeze(0) @ mask.float())  # [1, T]
            if not no_correction:
                freq = mask.float().mean(0, keepdims=True)  # [T]
                saliencies /= freq  # [1, T]

            saliencies = saliencies.squeeze()
            for i in valid_indices:
                explanations[(i,)] = saliencies[i].item()
        elif interaction_order >= 2:
            # pairwise
            sep_idx = tokens.index(self.tokenizer.sep_token)
            # only cross pairwise
            sep_pos = -1
            for i in valid_indices:
                if i >= sep_idx:
                    sep_pos = i
                    break

            feature_groups = product(valid_indices[:sep_pos], valid_indices[sep_pos:])
            for group in feature_groups:
                # 1 iff all indices are present
                explanations[group] = self.group_attribution(group,
                                                             scores,
                                                             mask,
                                                             no_correction=no_correction)
            # higher order
            for _ in range(interaction_order - 2):
                top_p_prev_int = sorted(explanations.items(),
                                        key=lambda x: x[1],
                                        reverse=True)[:int(top_p * len(explanations))]
                explanations = {}
                for prev_group, _ in top_p_prev_int:
                    for i in valid_indices:
                        if i in prev_group:
                            continue
                        group = tuple(sorted(prev_group + (i,)))
                        explanations[group] = self.group_attribution(
                            group, scores, mask, no_correction=no_correction)

        return explanations, tokens, pred_class

    def group_attribution(self, group, scores, mask, no_correction=False):
        interaction_mask = torch.prod(mask[:, group].float(), dim=1)
        if not no_correction:
            freq = interaction_mask.mean()
            return (interaction_mask @ scores / freq).item()
        return (interaction_mask @ scores).item()
