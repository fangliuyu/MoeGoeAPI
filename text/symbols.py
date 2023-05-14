_pad = '_'
_punctuation = ';:,.!?¡¿—…"«»“” '

_punctuation_zh = '；：，。！？-“”《》、（）ＢＰ…—~.\·『』・ '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

_numbers = '1234567890'
_others = ''

_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_punctuation_zh)+ list(_letters) + list(_numbers)+ list(_others)+ list(_letters_ipa)

# Special symbol ids
SPACE_ID = symbols.index(" ")
