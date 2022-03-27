"""
A abstracted API for getting the API with only public config.
"""
import sys

sys.path.insert(0, '/data/schoiaj/repos/nli_explain')

import torch
from transformers import BertForSequenceClassification, BertTokenizer

from utils.utils import load_pretrained_config
from explainers.integrated_hessians.path_explain.explainers.embedding_explainer_torch import \
    EmbeddingExplainerTorch


class IHBertExplainer:

    def __init__(self, model_name, device='cpu', baseline_token='[MASK]'):
        config = load_pretrained_config(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(config['model_card'])
        self.model = BertForSequenceClassification.from_pretrained(
            config['model_card']).to(device)
        self.label_map = config['label_map']
        self.device = device
        self.baseline_token = baseline_token

        self.explainer = EmbeddingExplainerTorch(self.prediction_model)

    ### Here we define functions that represent two pieces of the model:
    ### embedding and prediction
    def embedding_model(self, **inputs):
        # TODO: This is only for BERT!
        batch_embedding = self.model.bert.embeddings(**inputs)
        return batch_embedding

    def prediction_model(self, batch_embedding):
        # Note: this isn't exactly the right way to use the attention mask.
        # It should actually indicate which words are real words. This
        # makes the coding easier however, and the output is fairly similar,
        # so it suffices for this tutorial.
        # attention_mask = torch.ones(batch_embedding.shape[:2]).to(device)
        # attention_mask = tf.cast(attention_mask, dtype=tf.float32)

        encoder_outputs = self.model.bert.encoder(batch_embedding,
                                                  output_hidden_states=True,
                                                  return_dict=False)
        sequence_output = encoder_outputs[0]
        pooled_output = self.model.bert.pooler(sequence_output)
        logits = self.model.classifier(pooled_output)
        return logits

    def explain(self,
                premise,
                hypothesis,
                output_indices=None,
                batch_size=32,
                num_samples=256,
                use_expectation=False):
        inputs = self.tokenizer(premise, text_pair=hypothesis,
                                return_tensors='pt').to(self.device)
        ### First we need to decode the tokens from the batch ids.
        batch_sentences = self.tokenizer.tokenize(
            f'[CLS] {premise} [SEP] {hypothesis} [SEP]')
        batch_ids = inputs['input_ids']
        inputs.pop('attention_mask')
        batch_embedding = self.embedding_model(**inputs).detach()

        baseline_ids = torch.where(batch_ids == self.tokenizer.sep_token_id,
                                   self.tokenizer.sep_token_id, self.baseline_token)
        baseline_ids[:, 0] = self.tokenizer.cls_token_id
        baseline_inputs = {k: v for k, v in inputs.items()}
        baseline_inputs['input_ids'] = baseline_ids
        baseline_embedding = self.embedding_model(**baseline_inputs).detach()

        logits = self.prediction_model(batch_embedding)
        pred_label = int(torch.argmax(logits, -1))

        ### We are finally ready to explain our model
        explainer = EmbeddingExplainerTorch(self.prediction_model)

        ### For interactions, the hessian is rather large so we use a very small batch size
        interactions = explainer.interactions(
            inputs=batch_embedding,
            baseline=baseline_embedding,
            batch_size=batch_size,
            num_samples=num_samples,
            use_expectation=use_expectation,
            output_indices=pred_label if output_indices is None else output_indices,
            verbose=False)

        return interactions, batch_sentences, pred_label
