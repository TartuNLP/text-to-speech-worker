from unidecode import unidecode
import re
import logging
from typing import List

from tts_preprocess_et.convert import convert_sentence

logger = logging.getLogger(__name__)

# define and compile all the patterns
EST_PRE_REGEXPS = (
    (re.compile(r'[`´’\']'), ''),
    (re.compile(r'[()]'), ', ')
)

VRO_PRE_REGEXPS = {
    (re.compile('y'), 'õ'),
    (re.compile('([ǴḰĹḾŃṔŔŚǵḱĺḿńṕŕśǘ])'), r'\g<1>\'')
}


POST_REGEXPS = (
    (re.compile(r'[()[\]:;−­–…—]'), ', '),
    (re.compile(r'[«»“„”]'), '"'),
    (re.compile(r'[*\'\\/-]'), ' '),
    (re.compile(r'[`´’]'), "'"),
    (re.compile(r' +([.,!?])'), r'\g<1>'),
    (re.compile(r', ?([.,?!])'), r'\g<1>'),
    (re.compile(r'\.+'), '.'),
    (re.compile(r' +'), ' '),
    (re.compile(r'^ | $'), ''),
    (re.compile(r'^, ?'), ''),
    (re.compile(r'\s+'), ' '),
)


def clean(sent: str, alphabet: List[str], frontend: str = 'est'):
    if frontend == 'est':
        for regex, sub in EST_PRE_REGEXPS:
            sent = regex.sub(sub, sent)

        try:
            sent = convert_sentence(sent)
        except Exception as ex:
            logger.error(str(ex), sent)

    elif frontend == "vro":
        for regex, sub in VRO_PRE_REGEXPS:
            sent = regex.sub(sub, sent)

        sent = unidecode(sent)

    for regex, sub in POST_REGEXPS:
        sent = regex.sub(sub, sent)

    sent = sent.lower()
    sent = ''.join([char for char in sent if char in alphabet])  # TODO
    sent = ' '.join(sent.split())
    return sent


def split_sentence(sent, max_len):
    sub_sents = []

    while len(sent) > max_len:
        i = sent[:max_len].rfind(' ')
        if i == -1:
            i = max_len
        sub_sents.append(sent[:i])
        sent = sent[i:]
    sub_sents.append(sent)

    return sub_sents
