# nli_explain
Explain BERT model's decision in NLI task.

## Description

This repository contains various methods and attempts of explaining BERT model's decision during Natural Language Inference (NLI) task 
with Feature Interaction methods. This contains:

* IH - [integrated Hessians](https://github.com/suinleelab/path_explain)
* [Archipelago](https://github.com/mtsang/archipelago)
* X-Archipelago (ours) - cross-sentence version (for NLI examples with two sentences) of Archipelgo
* MAFIA (ours) - MAsk-based Feature Interaction Attribution

## requirements

```
nltk
pandas
numpy
torch
transformers
```

## How to Use

1. download e-SNLI dataset from https://github.com/OanaMariaCamburu/e-SNLI/tree/master/dataset. Put the files in `data/` folder.
2. preprocess data by `python prepare_data.py`.
3. For each explainer (lime, IH, Arch (Archipelago), Mask (MAFIA)), you can find a script in `scripts/` folder. It will create a `.json` file containing explanation for each NLI example in the `explanations/` directory (created if not already).
  * for details, check `explainers/save_explanations.py`.
  * for implementation of each explainer, check `explainers/` directory.
4. For evaluation, checkout the `scripts/eval_explanation.sh` script. The template function and example usage are shown here. Modify the file according to your needs.
