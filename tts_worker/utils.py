import re
import logging
from typing import List

from tts_preprocess_et.convert import convert_sentence

logger = logging.getLogger(__name__)

# define and compile all the patterns
PRE_REGEXES = (
    (re.compile(r'[`´’\']'), ''),
    (re.compile(r'[()]'), ', ')
)

POST_REGEXES = (
    (re.compile(r'[()[\]:;−­–…—]'), ', '),
    (re.compile(r'[«»“„”]'), '"'),
    (re.compile(r'[*\'\\/-]'), ' '),
    (re.compile(r'[`´’\']'), ''),
    (re.compile(r' +([.,!?])'), r'\g<1>'),
    (re.compile(r', ?([.,?!])'), r'\g<1>'),
    (re.compile(r'\.+'), ''),

    (re.compile(r' +'), ' '),
    (re.compile(r'^ | $'), ''),
    (re.compile(r'^, ?'), ''),
    (re.compile(r'\s+'), ' '),
)


def clean(sent: str, alphabet: List[str], frontend: str = 'est'):
    for regex, sub in PRE_REGEXES:
        sent = regex.sub(sub, sent)

    if frontend == 'est':
        try:
            sent = convert_sentence(sent)
        except Exception as ex:
            logger.error(str(ex), sent)

    for regex, sub in POST_REGEXES:
        sent = regex.sub(sub, sent)

    sent = sent.lower()
    sent = ''.join([char for char in sent if char in alphabet])  # TODO
    sent = ' '.join(sent.split())
    return sent


def split_sentence(sent, max_len):
    sub_sents = []

    while len(sent) > max_len:
        i = sent[:max_len].rfind(' ')
        sub_sents.append(sent[:i])
        sent = sent[i:]
    sub_sents.append(sent)

    return sub_sents
