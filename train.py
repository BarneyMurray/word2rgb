import functools
import json

from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer, EarlyStoppingCallback,
                          IntervalStrategy, TrainingArguments)

from get_train_data import create_dataset, get_df, get_train_val_test_splits
from utils import MultiClassRegressionTrainer, compute_metrics

num_epochs = 3
batch_size = 16

config = AutoConfig.from_pretrained(
    'distilbert-base-uncased', 
    num_labels=3
    )

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tokenize_fn = functools.partial(tokenizer, truncation=True, max_length=64, padding='max_length')

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", config=config)

# create train data
df = get_df()
train_dataset, val_dataset, test_dataset = (
    create_dataset(split, tokenize_fn) 
    for split in get_train_val_test_splits(df, num_samples=200)
    )

eval_steps = len(train_dataset) // batch_size // 3  # evaluate three times per epoch

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=64,
    warmup_steps=len(train_dataset) // 2,  # warmup for half of first epoch
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy=IntervalStrategy.STEPS,
    save_steps=eval_steps*3,  # checkpoint every three evaluation steps
    save_total_limit=2,
    eval_steps=eval_steps,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss'
)

trainer = MultiClassRegressionTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer.train()
metrics = trainer.evaluate(test_dataset, metric_key_prefix='test')
json.dump(metrics, open('./results/test/test_results.json', 'w'))

trainer.save_model('./results/test')
tokenizer.save_pretrained('./results/test')



