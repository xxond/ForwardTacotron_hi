""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. '''
from utils.text import cmudict

_pad = '_'
_punctuation = '!\'(),.:;? '
_special = '-'

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
#_arpabet = ['@' + s for s in cmudict.valid_symbols]

# Phonemes
#_vowels = 'iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɶɑɒᵻ'
#_non_pulmonic_consonants = 'ʘɓǀɗǃʄǂɠǁʛ'
#_pulmonic_consonants = 'pbtdʈɖcɟkɡqɢʔɴŋɲɳnɱmʙrʀⱱɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟ'
#_suprasegmentals = 'ˈˌːˑ'
#_other_symbols = 'ʍwɥʜʢʡɕʑɺɧ'
#_diacrilics = 'ɚ˞ɫ'

pre_path = r'/'.join(__file__.split(r'/')[:-1])
with open(pre_path + '/phonemes.txt', 'r') as fh:
    for line in fh:
        _chars = line[:-1]
        break

phonemes = sorted(list(_pad + _punctuation + _special + _chars))
   #_vowels + _non_pulmonic_consonants
   #+ _pulmonic_consonants + _suprasegmentals + _other_symbols + _diacrilics
