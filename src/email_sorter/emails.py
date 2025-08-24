import email
import email.message
import email.policy
import json
from dataclasses import dataclass
from imaplib import IMAP4_SSL

from html2text import html2text

from email_sorter import default_output_dir


@dataclass(frozen=True)
class Email:
    """Email parameters"""

    subject: str
    body: str | None = None


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


def parseEmail(mail: email.message.EmailMessage) -> dict[str, str]:
    output: dict[str, str] = {"Subject": mail["Subject"]}
    body_obj = mail.get_body(("html", "plain"))
    if body_obj is None:
        return output
    body_type = body_obj.get_content_subtype()
    body: str = body_obj.get_content()
    if body_type == "html":
        body = html2text(body)
    output["Body"] = body
    return output


def saveEmails(emails: list[email.message.EmailMessage], filename="emails.json"):
    parsed_emails = [parseEmail(e) for e in emails]
    with open(default_output_dir / filename, "w", newline="") as file:
        json.dump(parsed_emails, file, indent=2)
