from typing import Iterable

from . import Email


def labelEmail(email: Email) -> bool:
    while True:
        print("Subject:", email.subject, flush=True)
        response = input("Spam? [y/n] ").lower()
        if response in ("y", "yes"):
            return True
        elif response in ("n", "no"):
            return False
        elif response in ("?", "body"):
            print(email.body, end="\n\n")
        else:
            print("Please answer 'y' or 'n'.")
        input("Press enter to continue.")


def labelEmails(emails: Iterable[Email]):
    unlabeled_emails = [email for email in emails if email.spam is None]
    print(f"You have {len(unlabeled_emails)} unlabeled emails.")
    input("Press Enter to begin.")
    for email in unlabeled_emails:
        try:
            response = labelEmail(email)
            email.spam = response
        except EOFError:
            input("Skipping email.")
