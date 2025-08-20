# %%
from email_sorter import fetchData, fetchModel, prepTrainer, shrinkData, testModel

# %%
model, tokenizer = fetchModel("answerdotai/ModernBERT-base")
train, eval = fetchData("SetFit/enron_spam", tokenizer)

# %%
small_train = shrinkData(train, 5000)
small_eval = shrinkData(eval, 500)

# %%
trainer = prepTrainer(model, small_train, small_eval)
trainer.train()

# %%
test = ["Save 50% on new backpacks!", "Reminder: Homework due tomorrow."]
predictions = testModel(model, tokenizer, test)

for d, p in zip(test, predictions):
    print(f"Input: {d}", f"Prediction: {'spam' if p else 'ham'}", sep="\n", end="\n\n")
