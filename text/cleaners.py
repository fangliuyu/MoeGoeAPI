import re
from unidecode import unidecode
from phonemizer import phonemize
from pypinyin import Style, pinyin
from pypinyin.style._utils import get_finals, get_initials

# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
]]

def english_cleaners(text):
    text = unidecode(text)
    text = text.lower()
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    phonemes = phonemize(text, language='en-us', backend='espeak', strip=True)
    return re.sub(_whitespace_re, ' ', phonemes)


def english_cleaners2(text):
    text = unidecode(text)
    text = text.lower()
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    phonemes = phonemize(text, language='en-us', backend='espeak', strip=True, preserve_punctuation=True,
                         with_stress=True)
    return re.sub(_whitespace_re, ' ', phonemes)


def japanese_cleaners(text):
    from text.japanese import japanese_to_romaji_with_accent
    text = japanese_to_romaji_with_accent(text)
    text = re.sub(r'([A-Za-z])$', r'\1.', text)
    return text


def japanese_cleaners2(text):
    return japanese_cleaners(text).replace('ts', 'ʦ').replace('...', '…')


def korean_cleaners(text):
    '''Pipeline for Korean text'''
    from text.korean import latin_to_hangul, number_to_hangul, divide_hangul
    text = latin_to_hangul(text)
    text = number_to_hangul(text)
    text = divide_hangul(text)
    text = re.sub(r'([\u3131-\u3163])$', r'\1.', text)
    return text


def chinese_cleaners(text):
    '''Pipeline for Chinese text'''
    from text.mandarin import number_to_chinese, chinese_to_bopomofo, latin_to_bopomofo
    text = number_to_chinese(text)
    text = chinese_to_bopomofo(text)
    text = latin_to_bopomofo(text)
    text = re.sub(r'([ˉˊˇˋ˙])$', r'\1。', text)
    return text

def chinese_cleaners1(text):
    from pypinyin import Style, pinyin
    phones = [phone[0] for phone in pinyin(text, style=Style.TONE3)]
    return ' '.join(phones)


def chinese_cleaners2(text):
    phones = [
        p
        for phone in pinyin(text, style=Style.TONE3)
        for p in [
            get_initials(phone[0], strict=True),
            get_finals(phone[0][:-1], strict=True) + phone[0][-1]
            if phone[0][-1].isdigit()
            else get_finals(phone[0], strict=True)
            if phone[0][-1].isalnum()
            else phone[0],
        ]
        # Remove the case of individual tones as a phoneme
        if len(p) != 0 and not p.isdigit()
    ]
    return phones

def zh_ja_mixture_cleaners(text):
    from text.mandarin import chinese_to_romaji
    from text.japanese import japanese_to_romaji_with_accent
    text = re.sub(r'\[ZH\](.*?)\[ZH\]',
                  lambda x: chinese_to_romaji(x.group(1))+' ', text)
    text = re.sub(r'\[JA\](.*?)\[JA\]', lambda x: japanese_to_romaji_with_accent(
        x.group(1)).replace('ts', 'ʦ').replace('u', 'ɯ').replace('...', '…')+' ', text)
    text = re.sub(r'\s+$', '', text)
    text = re.sub(r'([^\.,!\?\-…~])$', r'\1.', text)
    return text


def sanskrit_cleaners(text):
    text = text.replace('॥', '।').replace('ॐ', 'ओम्')
    text = re.sub(r'([^।])$', r'\1।', text)
    return text


def cjks_cleaners(text):
    from text.mandarin import chinese_to_lazy_ipa
    from text.japanese import japanese_to_ipa
    from text.korean import korean_to_lazy_ipa
    from text.sanskrit import devanagari_to_ipa
    from text.english import english_to_lazy_ipa
    text = re.sub(r'\[ZH\](.*?)\[ZH\]',
                  lambda x: chinese_to_lazy_ipa(x.group(1))+' ', text)
    text = re.sub(r'\[JA\](.*?)\[JA\]',
                  lambda x: japanese_to_ipa(x.group(1))+' ', text)
    text = re.sub(r'\[KO\](.*?)\[KO\]',
                  lambda x: korean_to_lazy_ipa(x.group(1))+' ', text)
    text = re.sub(r'\[SA\](.*?)\[SA\]',
                  lambda x: devanagari_to_ipa(x.group(1))+' ', text)
    text = re.sub(r'\[EN\](.*?)\[EN\]',
                  lambda x: english_to_lazy_ipa(x.group(1))+' ', text)
    text = re.sub(r'\s+$', '', text)
    text = re.sub(r'([^\.,!\?\-…~])$', r'\1.', text)
    return text


def cjke_cleaners(text):
    from text.mandarin import chinese_to_lazy_ipa
    from text.japanese import japanese_to_ipa
    from text.korean import korean_to_ipa
    from text.english import english_to_ipa2
    text = re.sub(r'\[ZH\](.*?)\[ZH\]', lambda x: chinese_to_lazy_ipa(x.group(1)).replace(
        'ʧ', 'tʃ').replace('ʦ', 'ts').replace('ɥan', 'ɥæn')+' ', text)
    text = re.sub(r'\[JA\](.*?)\[JA\]', lambda x: japanese_to_ipa(x.group(1)).replace('ʧ', 'tʃ').replace(
        'ʦ', 'ts').replace('ɥan', 'ɥæn').replace('ʥ', 'dz')+' ', text)
    text = re.sub(r'\[KO\](.*?)\[KO\]',
                  lambda x: korean_to_ipa(x.group(1))+' ', text)
    text = re.sub(r'\[EN\](.*?)\[EN\]', lambda x: english_to_ipa2(x.group(1)).replace('ɑ', 'a').replace(
        'ɔ', 'o').replace('ɛ', 'e').replace('ɪ', 'i').replace('ʊ', 'u')+' ', text)
    text = re.sub(r'\s+$', '', text)
    text = re.sub(r'([^\.,!\?\-…~])$', r'\1.', text)
    return text


def cjke_cleaners2(text):
    from text.mandarin import chinese_to_ipa
    from text.japanese import japanese_to_ipa2
    from text.korean import korean_to_ipa
    from text.english import english_to_ipa2
    text = re.sub(r'\[ZH\](.*?)\[ZH\]',
                  lambda x: chinese_to_ipa(x.group(1))+' ', text)
    text = re.sub(r'\[JA\](.*?)\[JA\]',
                  lambda x: japanese_to_ipa2(x.group(1))+' ', text)
    text = re.sub(r'\[KO\](.*?)\[KO\]',
                  lambda x: korean_to_ipa(x.group(1))+' ', text)
    text = re.sub(r'\[EN\](.*?)\[EN\]',
                  lambda x: english_to_ipa2(x.group(1))+' ', text)
    text = re.sub(r'\s+$', '', text)
    text = re.sub(r'([^\.,!\?\-…~])$', r'\1.', text)
    return text


def thai_cleaners(text):
    from text.thai import num_to_thai, latin_to_thai
    text = num_to_thai(text)
    text = latin_to_thai(text)
    return text


def shanghainese_cleaners(text):
    from text.shanghainese import shanghainese_to_ipa
    text = shanghainese_to_ipa(text)
    text = re.sub(r'([^\.,!\?\-…~])$', r'\1.', text)
    return text


def chinese_dialect_cleaners(text):
    from text.mandarin import chinese_to_ipa2
    from text.japanese import japanese_to_ipa3
    from text.shanghainese import shanghainese_to_ipa
    from text.cantonese import cantonese_to_ipa
    from text.english import english_to_lazy_ipa2
    from text.ngu_dialect import ngu_dialect_to_ipa
    text = re.sub(r'\[ZH\](.*?)\[ZH\]',
                  lambda x: chinese_to_ipa2(x.group(1))+' ', text)
    text = re.sub(r'\[JA\](.*?)\[JA\]',
                  lambda x: japanese_to_ipa3(x.group(1)).replace('Q', 'ʔ')+' ', text)
    text = re.sub(r'\[SH\](.*?)\[SH\]', lambda x: shanghainese_to_ipa(x.group(1)).replace('1', '˥˧').replace('5',
                  '˧˧˦').replace('6', '˩˩˧').replace('7', '˥').replace('8', '˩˨').replace('ᴀ', 'ɐ').replace('ᴇ', 'e')+' ', text)
    text = re.sub(r'\[GD\](.*?)\[GD\]',
                  lambda x: cantonese_to_ipa(x.group(1))+' ', text)
    text = re.sub(r'\[EN\](.*?)\[EN\]',
                  lambda x: english_to_lazy_ipa2(x.group(1))+' ', text)
    text = re.sub(r'\[([A-Z]{2})\](.*?)\[\1\]', lambda x: ngu_dialect_to_ipa(x.group(2), x.group(
        1)).replace('ʣ', 'dz').replace('ʥ', 'dʑ').replace('ʦ', 'ts').replace('ʨ', 'tɕ')+' ', text)
    text = re.sub(r'\s+$', '', text)
    text = re.sub(r'([^\.,!\?\-…~])$', r'\1.', text)
    return text
