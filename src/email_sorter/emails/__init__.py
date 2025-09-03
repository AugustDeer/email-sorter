import os
from typing import BinaryIO, Optional

import msgspec

from email_sorter import DEFAULT_OUTPUT_DIR

DEFAULT_EMAIL_FILE = os.path.join(DEFAULT_OUTPUT_DIR, "emails.json")


class Email(msgspec.Struct, omit_defaults=True, frozen=True):
    """Email parameters"""

    subject: str
    body: Optional[str] = None
    spam: Optional[bool] = None


def loadEmails(file: BinaryIO):
    data = file.read()
    emails = msgspec.json.decode(data, type=list[Email])
    return emails


def saveEmails(emails: list[Email], file: BinaryIO):
    data = msgspec.json.encode(emails)
    file.write(data)
