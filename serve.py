import argparse

import gradio as gr
from explainers.mask_explain.mask_explainer import MaskExplainer


class ModelInterface:

    def __init__(self, model_name):
        self.explainer = MaskExplainer(model_name,
                                       device='cuda',
                                       baseline_token='attention+[MASK]')

        self.explanations = None
        self.tokens = None

    def infer(self, sent1, sent2, mask_p, mask_n, topk):
        del topk  # unused
        explanations, tokens, pred_class = self.explainer.explain(sent1,
                                                                  sent2,
                                                                  batch_size=32,
                                                                  target_class=None,
                                                                  interaction_order=(1,),
                                                                  mask_p=mask_p,
                                                                  mask_n=mask_n,
                                                                  inverse_mask=False)
        inv_label_map = self.explainer.get_label_map(inv=True)
        pred_label = inv_label_map[pred_class]

        self.explanations = explanations
        self.tokens = tokens

        return pred_label

    def interpret(self, *args):
        topk = args[-1]
        topk_tok = sorted(self.explanations.items(), key=lambda x: x[1],
                          reverse=True)[:topk]
        topk_tok = {i[0]: s for i, s in topk_tok if s > 0}
        token_exp1 = []
        token_exp2 = []
        second_sent = False
        for i, tok in enumerate(self.tokens):
            if i == 0 or i == len(self.tokens) - 1:
                continue
            if tok == self.explainer.tokenizer.sep_token:
                second_sent = True
                continue
            if second_sent:
                token_exp2.append((tok, topk_tok.get(i, 0)))
            else:
                token_exp1.append((tok, topk_tok.get(i, 0)))
        return token_exp1, token_exp2, [0], [0], [0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="bert-base")
    args = parser.parse_args()
    model = ModelInterface(args.model_name)

    iface = gr.Interface(
        fn=model.infer,
        inputs=[
            gr.inputs.Textbox(lines=2, placeholder="text here...", label='sentence1'),
            gr.inputs.Textbox(lines=2, placeholder="text here...", label='sentence2'),
            gr.inputs.Slider(minimum=0.1,
                             maximum=0.8,
                             step=0.1,
                             default=0.4,
                             label='mask_p'),
            gr.inputs.Slider(minimum=1000,
                             maximum=10000,
                             step=500,
                             default=5000,
                             label='mask_n'),
            gr.inputs.Slider(minimum=1, maximum=6, step=1, default=3,
                             label='topk tokens'),
        ],
        outputs=[
            gr.outputs.Textbox(type="auto", label="prediction"),
        ],
        title='NLI Explainer demo',
        theme='peach',
        interpretation=model.interpret,
    )
    iface.launch(server_name="0.0.0.0")


if __name__ == '__main__':
    main()
