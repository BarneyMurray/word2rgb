from typing import Callable, List

import evaluate
import torch
from torch import nn
from transformers import Trainer


def compute_metrics(eval_preds):
    mse_metric = evaluate.load("mse", "multilist")
    logits, labels = (torch.from_numpy(p) for p in eval_preds)
    sigmoid = nn.Sigmoid()
    preds = sigmoid(logits)

    return mse_metric.compute(predictions=preds, references=labels)


class MultiClassRegressionTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")

        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        sigmoid = nn.Sigmoid()
        normalised_logits = sigmoid(logits).to(torch.float64)

        loss_fct = nn.MSELoss()
        loss = loss_fct(normalised_logits.view(-1), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


class ColorPredictionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: List[List[int]],
        tokenize_fn: Callable
        ):
        self.texts = texts
        self.labels = labels
        self.tokenize_fn = tokenize_fn         

    def __getitem__(self, idx):
        item = self.tokenize_fn(self.texts[idx])
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)