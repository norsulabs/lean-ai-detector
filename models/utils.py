import re


def clean_paragraph(paragraph):
    # Remove all symbols except for alphanumeric characters and spaces
    paragraph = re.sub(r"[^a-zA-Z0-9\s]", "", paragraph)
    # Replace multiple spaces with a single space
    paragraph = re.sub(r"\s+", " ", paragraph)
    # Strip leading and trailing spaces
    paragraph = paragraph.strip()

    return paragraph
