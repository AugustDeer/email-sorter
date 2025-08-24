# %%
from transformers import pipeline

from email_sorter.train import train_pipeline

output = train_pipeline(
    "answerdotai/ModernBERT-base", "SetFit/enron_spam", "enron_test_model"
)

classifier = pipeline("text-classification", model=str(output))

tests = ["Save 50% on new backpacks!", "Reminder: Homework due tomorrow."]

results = classifier(tests)

for test, result in zip(tests, results):
    print(test)
    print(result)
    print()
