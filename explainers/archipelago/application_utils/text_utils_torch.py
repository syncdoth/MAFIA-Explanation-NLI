import torch


class BertWrapperTorch:

    def __init__(self, model, device, merge_logits=False):
        self.model = model.to(device)
        self.device = device
        self.merge_logits = merge_logits

    def get_predictions(self, **inputs):
        if not isinstance(inputs['input_ids'], torch.Tensor):
            inputs = {k: torch.LongTensor(v).to(self.device) for k, v in inputs.items()}
        return self.model(**inputs).logits.detach().cpu()

    def __call__(self, **inputs):
        batch_predictions = self.get_predictions(**inputs)
        if self.merge_logits:
            batch_predictions2 = (batch_predictions[:, 1] - batch_predictions[:, 0])
            return batch_predictions2.unsqueeze(1).numpy()

        return batch_predictions.numpy()
