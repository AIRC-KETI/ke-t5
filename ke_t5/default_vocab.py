import t5.data
DEFAULT_SPM_PATH = "gs://ket5/vocabs/ket5.64000/sentencepiece.model"
DEFAULT_EXTRA_IDS = 100

DEFAULT_VOCAB = t5.data.SentencePieceVocabulary(
    DEFAULT_SPM_PATH, DEFAULT_EXTRA_IDS)