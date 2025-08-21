import logging

import evaluate
import numpy as np
import torch as t
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    ModernBertForSequenceClassification,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import SequenceClassifierOutput

logger: logging.Logger = logging.getLogger(__name__)


def fetchModel(
    model_name: str,
) -> tuple[ModernBertForSequenceClassification, PreTrainedTokenizerBase]:
    """Initializes the specified model and associated tokenizer"""

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model: ModernBertForSequenceClassification = (
        ModernBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            device_map="auto",
            id2label={0: "ham", 1: "spam"},
            label2id={"ham": 0, "spam": 1},
        )
    )

    # Freeze BERT weights
    for layer in model.model.parameters():
        layer.requires_grad = False

    return model, tokenizer


def fetchData(
    dataset_name: str, tokenizer: PreTrainedTokenizerBase
) -> tuple[Dataset, Dataset]:
    """Imports the specified dataset and tokenizes it."""

    # Import dataset
    dataset = load_dataset(dataset_name)

    # Process dataset
    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[str]]:
        return tokenizer(batch["subject"], padding="max_length", truncation=True).data

    dataset = dataset.map(tokenize, batched=True)

    return dataset["train"], dataset["test"]  # type: ignore


def prepTrainer(
    model: ModernBertForSequenceClassification,
    tokenizer: PreTrainedTokenizerBase,
    train: Dataset,
    eval: Dataset,
) -> Trainer:
    """Creates a Trainer object for the specified model and data."""
    # Define training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=64,
        dataloader_pin_memory=True,
        dataloader_num_workers=6,
        eval_strategy="epoch",
        save_strategy="epoch",
        output_dir="./output/trained_model",
    )

    metric = evaluate.load("accuracy")

    trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train,
        eval_dataset=eval,
        compute_metrics=lambda eval_pred: metric.compute(
            predictions=np.argmax(eval_pred.predictions, -1),
            references=eval_pred.label_ids,
        )
        or {},
    )

    return trainer


def testModel(
    model: ModernBertForSequenceClassification,
    tokenizer: PreTrainedTokenizerBase,
    data: list[str],
) -> t.Tensor:
    """Test model on given strings."""
    logger.info(f"Testing model on {len(data)} items.")

    inputs = tokenizer(
        data,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    with t.no_grad():
        outputs: SequenceClassifierOutput = model(**inputs)

    logits = outputs.logits
    assert logits is not None
    predictions = logits.softmax(-1)
    return predictions.cpu()
