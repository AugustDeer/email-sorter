import email
import email.message
import email.policy
import os
from imaplib import IMAP4_SSL

from rich import print
from rich.progress import Progress
from rich.prompt import IntPrompt, Prompt

from . import Email, saveEmails


def fetchEmails(
    user: str,
    passwd: str,
    *,
    num_emails: int = 100,
    domain: str = "imap.gmail.com",
    port: int = 993,
):
    with IMAP4_SSL(domain, port) as M:
        M.login(user, passwd)
        if M.state != "AUTH":
            raise
        M.select(readonly=True)
        _, data = M.uid("search", None, "ALL")  # type: ignore
        uids = data[0].split()
        n = len(uids)
        _, data = M.fetch(f"{max(1, n - num_emails)}:{n}", "(RFC822)")
    return [
        email.message_from_bytes(m[1], policy=email.policy.default)
        for m in data
        if isinstance(m, tuple)
    ]


def parseEmail(mail: email.message.EmailMessage) -> Email:
    body_obj = mail.get_body(("html", "plain"))
    subject = str(mail["Subject"])
    if body_obj is None:
        return Email(subject=subject)
    body = str(body_obj.get_content())
    return Email(subject=subject, body=body)


def main():
    address = Prompt.ask("Email address")
    passwd = Prompt.ask("App Password", password=True)
    while True:
        num_emails = IntPrompt.ask("Number of emails to download", default=100)
        if num_emails > 0:
            break
        print("[prompt.invalid]Answer must be positive")
    dest_dir = Prompt.ask("Output dir", default="./output/")
    path = os.path.join(dest_dir, "emails.json")

    with Progress(transient=True) as progress:
        with progress.console.status("Downloading emails...") as status:
            emails = fetchEmails(address, passwd, num_emails=num_emails)
            progress.console.print("Downloaded", len(emails), "emails.")
            status.update("Parsing emails...")
            parsed_emails = [parseEmail(e) for e in progress.track(emails)]
            status.update("Saving emails...")
            with open(path, "wb") as output:
                saveEmails(parsed_emails, output)
        progress.console.print("Saved", len(parsed_emails), "emails to", path)


if __name__ == "__main__":
    main()
