from argparse import ArgumentParser
import pandas as pd
from utils import clean
from transformers import AutoTokenizer
import torch
from transformers import Trainer, TrainingArguments
import numpy as np
import evaluate
from peft import PeftModel, get_peft_model
from peft import LoraConfig, TaskType
from transformers import AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
import os
from datasets import Dataset
import wandb


class Config:

    def __init__(
        self,
        model,
        max_length,
        batch_size,
        epochs,
        learning_rate,
        device,
        save_steps,
        eval_steps,
    ):
        self.model = model
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = device
        self.save_steps = save_steps
        self.eval_steps = eval_steps


def load_data(input_path):
    # Load the data from the input path
    data = pd.read_csv(input_path)
    data = data.astype({"generated": int})
    data = Dataset.from_pandas(data.sample(10000))
    data = data.map(clean)
    data = data.class_encode_column("generated")
    return data


def train_model(data, cfg):

    tokenizer = AutoTokenizer.from_pretrained(cfg.model, use_fast=True)

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=cfg.max_length,
            return_tensors="pt",
        )

    tokenized_data = data.map(tokenize, batched=True)

    tokenized_data = tokenized_data.remove_columns(column_names=["__index_level_0__"])

    tokenized_data = tokenized_data.rename_column("generated", "labels")
    tokenized_data = tokenized_data.remove_columns("text")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenized_data = tokenized_data.train_test_split(test_size=0.3)

    id2label = {0: "Human", 1: "AI"}
    label2id = {"Human": 0, "AI": 1}

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model, num_labels=2, id2label=id2label, label2id=label2id
    )
    peft_config = LoraConfig(
        task_type="SEQ_CLS",
        inference_mode=False,
        r=4,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules="all-linear",
    )
    model = get_peft_model(model=model, peft_config=peft_config)
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir="output",
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.epochs,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        save_total_limit=2,
        report_to="wandb",
        load_best_model_at_end=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"],
        data_collator=data_collator,
    )

    trainer.train()
    return


def save_model(model, model_path):
    # Save the model to the model path
    pass


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--input", type=str, help="Input file path", required=True)
    parser.add_argument(
        "--model",
        type=str,
        help="Model file path",
        default="distilbert/distilbert-base-uncased",
    )
    parser.add_argument("--batch_size", type=int, help="Batch size", default=8)
    parser.add_argument("--epochs", type=int, help="Number of epochs", default=3)
    parser.add_argument(
        "--learning_rate", type=float, help="Learning rate", default=2e-5
    )

    parser.add_argument("--max_length", type=int, help="Max Length", default=512)

    parser.add_argument("--save_steps", type=int, help="Save Steps", default=200)
    parser.add_argument("--eval_steps", type=int, help="Eval Steps", default=200)

    args = parser.parse_args()

    cfg = Config(
        model=args.model,
        max_length=args.max_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
    )

    wandb.init(
        # set the wandb project where this run will be logged
        project="ai-text-detector",
        # track hyperparameters and run metadata
        config=cfg,
        id="345",
        name="sd",
    )

    # Load the data
    data = load_data(args.input)

    # Train the model
    model = train_model(data, cfg)
