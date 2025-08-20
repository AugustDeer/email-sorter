import logging

import evaluate
import numpy as np
import torch as t
from accelerate import Accelerator
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

logger: logging.Logger = logging.getLogger(__name__)

accelerator: Accelerator = Accelerator()
device: t.device = accelerator.device


def fetchModel(model_name: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Initializes the specified model and associated tokenizer"""

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name)
    model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        device_map="auto",
        torch_dtype=t.float16,
    )

    return model, tokenizer


def fetchData(
    dataset_name: str, tokenizer: PreTrainedTokenizerBase
) -> tuple[Dataset, Dataset]:
    """Imports the specified dataset and tokenizes it."""

    # Import dataset
    dataset: DatasetDict = load_dataset(dataset_name)  # pyright: ignore[reportAssignmentType]

    # Process dataset
    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[str]]:
        return tokenizer(batch["text"], padding="max_length", truncation=True).data

    dataset = dataset.map(tokenize, batched=True)

    return dataset["train"], dataset["test"]


def shrinkData(data: Dataset, size: int, seed: int = 42) -> Dataset:
    """Randomly select elements from the dataset to reduce size."""
    return data.shuffle(seed=seed).select(range(size))


def prepTrainer(model: PreTrainedModel, train: Dataset, eval: Dataset) -> Trainer:
    """Creates a Trainer object for the specified model and data."""
    # Define training arguments
    training_args = TrainingArguments(
        eval_strategy="epoch",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=16,
        batch_eval_metrics=True,
        per_device_eval_batch_size=4,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        torch_empty_cache_steps=8,
        # torch_compile=True,
        # torch_compile_backend="inductor",
    )

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred, compute_result: bool = True) -> dict | None:
        logits, labels = eval_pred
        predictions = t.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=labels)
        if compute_result:
            return metric.compute()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=eval,
        compute_metrics=compute_metrics,  # pyright: ignore[reportArgumentType]
    )

    return trainer


def testModel(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, data: list[str]
) -> list[str]:
    """Test model on given strings."""
    logger.info(f"Testing model on {len(data)} items.")

    inputs = tokenizer(
        data,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with t.no_grad():
        outputs = model(**inputs)

    logits: np.ndarray = outputs.logits.to("cpu")

    predictions: np.ndarray = np.argmax(logits, axis=-1)

    return predictions.tolist()
