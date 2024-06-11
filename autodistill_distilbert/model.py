import evaluate
import numpy as np
from autodistill.text_classification import TextClassificationTargetModel
from datasets import load_dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments,
                          pipeline)


class DistilBERT(TextClassificationTargetModel):
    def __init__(self, model_name = None):
        if model_name:
            self.model = pipeline(
                "text-classification",
                model=model_name,
                tokenizer="distilbert/distilbert-base-uncased",
                return_all_scores=True,
            )

    def predict(self, input: str) -> dict:
        result = self.model(input)

        return result

    def train(self, dataset_file, output_dir="output", epochs=2):
        accuracy = evaluate.load("accuracy")

        tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return accuracy.compute(predictions=predictions, references=labels)

        def preprocess_function(examples):
            return tokenizer(examples["text"], truncation=True)

        dataset = load_dataset("json", data_files=dataset_file, split="train")
        dataset = dataset.rename_column("content", "text")
        dataset = dataset.train_test_split(test_size=0.2)
        tokenized_dataset = dataset.map(preprocess_function, batched=True)
        # drop null labels
        tokenized_dataset = tokenized_dataset.filter(lambda example: example["classification"] is not None)

        # assign ids to each unique label
        labels = list(set(tokenized_dataset["train"]["classification"]))
        # order alphabetically
        labels = sorted([i for i in labels if i is not None])

        id2label = {i: label for i, label in enumerate(labels)}
        label2id = {v: k for k, v in id2label.items()}

        def convert_label_to_id(example):
            example["label"] = label2id[example["classification"]]
            return example

        tokenized_dataset = tokenized_dataset.map(convert_label_to_id)

        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert/distilbert-base-uncased",
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id,
        )

        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=epochs,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        trainer.evaluate()
