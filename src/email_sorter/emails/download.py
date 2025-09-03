import email
import email.message
import email.policy
import imaplib
from imaplib import IMAP4_SSL

import gradio as gr
import msgspec

from . import DEFAULT_EMAIL_FILE, Email, saveEmails


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


def fetchParseSave(
    address: str,
    password: str,
    path: str = DEFAULT_EMAIL_FILE,
    num_emails: int = 100,
    domain: str = "imap.gmail.com",
    port: int = 993,
    progress: gr.Progress = gr.Progress(),
):
    try:
        emails = fetchEmails(
            address, password, num_emails=num_emails, domain=domain, port=port
        )
    except imaplib.IMAP4.error as e:
        raise gr.Error(str(e))
    parsed_emails: list[Email] = []
    for e in progress.tqdm(emails, desc="Parsing...", unit="emails"):
        parsed_emails.append(parseEmail(e))  # type: ignore
    with open(path, "wb") as output:
        saveEmails(parsed_emails, output)
    gr.Info(f"Downloaded emails to {path}")
    return [msgspec.structs.asdict(e) for e in parsed_emails]


demo = gr.Interface(
    fetchParseSave,
    inputs=[
        gr.Textbox(type="email"),
        gr.Textbox(type="password"),
    ],
    additional_inputs=[
        gr.Textbox(DEFAULT_EMAIL_FILE),
        gr.Number(100),
        gr.Textbox("imap.gmail.com"),
        gr.Number(993),
    ],
    outputs="json",
)


if __name__ == "__main__":
    demo.launch()
