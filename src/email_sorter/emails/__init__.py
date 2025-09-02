from typing import BinaryIO, Optional

import msgspec

from .. import default_output_dir


class Email(msgspec.Struct, omit_defaults=True):
    """Email parameters"""

    subject: str
    body: Optional[str] = None
    spam: Optional[bool] = None


def loadEmails(filename="emails.json"):
    with open(default_output_dir / filename) as file:
        data = file.read()
    emails = msgspec.json.decode(data, type=list[Email])
    return emails


def saveEmails(emails: list[Email], file: BinaryIO):
    data = msgspec.json.encode(emails)
    file.write(data)
