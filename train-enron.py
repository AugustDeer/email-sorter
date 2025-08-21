# %%
import torch as t

from email_sorter import fetchData, fetchModel, prepTrainer, testModel

# %%
model, tokenizer = fetchModel("answerdotai/ModernBERT-base")
train, eval = fetchData("SetFit/enron_spam", tokenizer)

# %%
trainer = prepTrainer(model, tokenizer, train, eval)
trainer.train()

model.eval()
with t.no_grad():
    metrics = trainer.evaluate()
    print("Accuracy:", metrics["eval_accuracy"])

# %%
test = ["Save 50% on new backpacks!", "Reminder: Homework due tomorrow."]
predictions = testModel(model, tokenizer, test)

for d, p in zip(test, predictions):
    print(f"Input: {d}")
    print(*(f"{k} {p[v]}" for k, v in model.config.label2id.items()))
