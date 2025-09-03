import logging
from pathlib import Path

import evaluate
import numpy as np
from datasets import DatasetDict, load_dataset
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    BatchEncoding,
    DataCollatorWithPadding,
    EvalPrediction,
    ModernBertForSequenceClassification,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from email_sorter import DEFAULT_OUTPUT_DIR

logger: logging.Logger = logging.getLogger(__name__)


def prepBaseModel(
    model_name: str,
) -> tuple[ModernBertForSequenceClassification, PreTrainedTokenizerBase]:
    """Initializes the specified model and associated tokenizer"""

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model: ModernBertForSequenceClassification = (
        ModernBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            device_map="auto",
            id2label={0: "HAM", 1: "SPAM"},
            label2id={"HAM": 0, "SPAM": 1},
        )
    )

    # Freeze BERT weights
    for layer in model.model.parameters():
        layer.requires_grad = False

    return model, tokenizer


def prepData(dataset_name: str, tokenizer: PreTrainedTokenizerBase) -> DatasetDict:
    """Imports the specified dataset and tokenizes it."""

    # Import dataset
    dataset = load_dataset(dataset_name)

    # Process dataset
    def tokenize(batch: Dataset) -> BatchEncoding:
        return tokenizer(batch["text"], truncation=True)

    dataset = dataset.map(tokenize, batched=True)

    return dataset  # type: ignore


def prepTrainer(
    model: ModernBertForSequenceClassification,
    tokenizer: PreTrainedTokenizerBase,
    data: DatasetDict,
    output_location: Path,
) -> Trainer:
    """Creates a Trainer object for the specified model and data."""
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=-1)
        return accuracy.compute(predictions=predictions, references=labels) or {}

    training_args = TrainingArguments(
        output_dir=str(output_location),
        per_device_train_batch_size=64,
        dataloader_pin_memory=True,
        dataloader_num_workers=6,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    return Trainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        train_dataset=data["train"],
        eval_dataset=data["test"],
        compute_metrics=compute_metrics,
    )


def train_pipeline(model_name: str, dataset_name: str, output_name: str) -> Path:
    model, tokenizer = prepBaseModel(model_name)
    data = prepData(dataset_name, tokenizer)
    output_location = Path(DEFAULT_OUTPUT_DIR, output_name)
    trainer = prepTrainer(model, tokenizer, data, output_location)
    trainer.train()
    trainer.save_model()
    return output_location
