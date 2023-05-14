""" from https://github.com/keithito/tacotron """
from . import cleaners
from .symbols import symbols


def text_to_sequence(text,modelsymbols, cleaner_names):
  if  modelsymbols:
    _symbol_to_id = {s: i for i, s in enumerate(modelsymbols)}
  else:
    _symbol_to_id = {s: i for i, s in enumerate(symbols)}

  sequence = []

  clean_text = _clean_text(text, cleaner_names)
  for symbol in clean_text:
    if symbol not in _symbol_to_id.keys():
      continue
    symbol_id = _symbol_to_id[symbol]
    sequence += [symbol_id]
  return sequence


def cleaned_text_to_sequence(modelsymbols,cleaned_text):
  if  modelsymbols:
    _symbol_to_id = {s: i for i, s in enumerate(modelsymbols)}
  else:
    _symbol_to_id = {s: i for i, s in enumerate(symbols)}
  sequence = [_symbol_to_id[symbol] for symbol in cleaned_text]
  return sequence

def sequence_to_text(modelsymbols,sequence):
  if  modelsymbols:
    _id_to_symbol = {i: s for i, s in enumerate(modelsymbols)}
  else:
    _id_to_symbol = {i: s for i, s in enumerate(symbols)}
    result = ''
  for symbol_id in sequence:
      s = _id_to_symbol[symbol_id]
      result += s
  return result

def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text
