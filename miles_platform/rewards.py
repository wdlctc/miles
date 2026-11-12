"""Bounded numeric grading for untrusted customer text; no symbolic evaluation."""

import re
from fractions import Fraction

NUMBER = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"


def numeric(value):
    value = str(value).strip()
    if len(value) > 128:
        raise ValueError("Numeric labels must be at most 128 characters")
    value = value.replace(",", "")
    value = re.sub(r"\\(?:d?frac)\{(" + NUMBER + r")\}\{(" + NUMBER + r")\}", r"\1/\2", value)
    if not re.fullmatch(NUMBER + r"(?:/" + NUMBER + r")?", value):
        raise ValueError("This recipe accepts numeric labels only")
    parts = value.split("/")
    try:
        return Fraction(parts[0]) / (Fraction(parts[1]) if len(parts) == 2 else 1)
    except ZeroDivisionError:
        raise ValueError("Zero denominator is not allowed") from None


def grade(response, label):
    text = str(response)[-16000:]
    start = text.rfind("\\boxed{")
    if start < 0:
        return False
    depth, answer = 1, []
    for char in text[start + 7 : start + 7 + 256]:
        depth += int(char == "{") - int(char == "}")
        if depth == 0:
            break
        answer.append(char)
    if depth:
        return False
    try:
        return numeric("".join(answer)) == numeric(label)
    except ValueError:
        return False


async def reward(args, sample, **kwargs):
    return int(grade(sample.response, sample.label))
