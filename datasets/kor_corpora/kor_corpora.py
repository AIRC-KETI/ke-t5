"""kor_corpora dataset."""
import os
import csv
import hashlib
import functools
import textwrap

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


_VERSION = tfds.core.Version('1.0.0')

# Korean dataset
# ---------------------------------------------
# NSMC sentiment analysis: https://github.com/e9t/nsmc
_KO_NSMC_URL='https://github.com/e9t/nsmc'
_KO_NSMC_ROOT='https://github.com/e9t/nsmc/raw/master'
_KO_NSMC_TRAIN_LINK=os.path.join(_KO_NSMC_ROOT, 'ratings_train.txt')
_KO_NSMC_TEST_LINK=os.path.join(_KO_NSMC_ROOT, 'ratings_test.txt')
_KO_NSMC_DEFAULT_SPLIT={'train': _KO_NSMC_TRAIN_LINK, 'test': _KO_NSMC_TEST_LINK}
_KO_NSMC_CITATION=textwrap.dedent("""\
@misc{e9t_nsmc,
    author       = {e9t},
    title        = {{Naver sentiment movie corpus}},
    month        = jun,
    year         = 2016,
    version      = {1.0},
    url          = {https://github.com/e9t/nsmc}
    }""")
_KO_NSMC_TEXT_FEATURES = {'sentence': 'document'}
_KO_NSMC_LABEL_CLASSES = ['negative', 'positive']
_KO_NSMC_LABEL_COL = 'label' # (0: negative, 1: positive)
_KO_NSMC_DESCRIPTION = textwrap.dedent("""\
Naver sentiment movie corpus v1.0
...
""")
_KO_NSMC_SPLIT = {
  'source': {
    tfds.Split.TRAIN: ['train'],
    tfds.Split.VALIDATION: ['train'],
    tfds.Split.TEST: ['test'],
  },
  'split': {
    tfds.Split.TRAIN: lambda x: x % 10 != 0,
    tfds.Split.VALIDATION: lambda x: x % 10 == 0,
    tfds.Split.TEST: lambda x: True,
}}

# Question pair: https://github.com/aisolab/nlp_classification/tree/master/BERT_pairwise_text_classification/qpair
_KO_QPAIR_URL='https://github.com/songys/Question_pair'
_KO_QPAIR_ROOT='https://github.com/aisolab/nlp_classification/raw/master/BERT_pairwise_text_classification/qpair'
_KO_QPAIR_TRAIN_LINK=os.path.join(_KO_QPAIR_ROOT, 'train.txt')
_KO_QPAIR_VALIDATION_LINK=os.path.join(_KO_QPAIR_ROOT, 'validation.txt')
_KO_QPAIR_TEST_LINK=os.path.join(_KO_QPAIR_ROOT, 'test.txt')
_KO_QPAIR_DEFAULT_SPLIT={
  'train': _KO_QPAIR_TRAIN_LINK,
  'validation': _KO_QPAIR_VALIDATION_LINK,
  'test': _KO_QPAIR_TEST_LINK}
_KO_QPAIR_CITATION=textwrap.dedent("""\
@misc{songys_question_pair,
    author       = {songys},
    title        = {{Paired Question}},
    month        = jun,
    year         = 2020,
    version      = {2.0},
    url          = {https://github.com/e9t/nsmc}
    }""")
_KO_QPAIR_TEXT_FEATURES = {'question1':'question1', 'question2':'question2'}
_KO_QPAIR_LABEL_CLASSES = ['duplicated', 'not_duplicate']
_KO_QPAIR_LABEL_COL = 'is_duplicate' # (0: duplicated, 1: not_duplicate)
_KO_QPAIR_DESCRIPTION = textwrap.dedent("""\
Paired Question v.2
...
""")
_KO_QPAIR_SPLIT = {
  'source': {
    tfds.Split.TRAIN: ['train'],
    tfds.Split.VALIDATION: ['validation'],
    tfds.Split.TEST: ['test'],
  },
  'split': {
    tfds.Split.TRAIN: lambda x: True,
    tfds.Split.VALIDATION: lambda x: True,
    tfds.Split.TEST: lambda x: True,
}}

# NLI: https://github.com/kakaobrain/KorNLUDatasets
_KO_NLI_URL='https://github.com/kakaobrain/KorNLUDatasets'
_KO_NLI_ROOT='https://github.com/kakaobrain/KorNLUDatasets/raw/master/KorNLI'
_KO_MNLI_TRAIN_LINK = os.path.join(_KO_NLI_ROOT, 'multinli.train.ko.tsv')
_KO_SNLI_TRAIN_LINK = os.path.join(_KO_NLI_ROOT, 'snli_1.0_train.ko.tsv')
_KO_XNLI_DEV_LINK = os.path.join(_KO_NLI_ROOT, 'xnli.dev.ko.tsv')
_KO_XNLI_TEST_LINK = os.path.join(_KO_NLI_ROOT, 'xnli.test.ko.tsv')
_KO_NLI_DEFAULT_SPLIT={
  'mnli_train': _KO_MNLI_TRAIN_LINK,
  'snli_train': _KO_SNLI_TRAIN_LINK,
  'xnli_dev': _KO_XNLI_DEV_LINK,
  'xnli_test': _KO_XNLI_TEST_LINK}
_KO_NLI_CITATION=textwrap.dedent("""\
@article{ham2020kornli,
  title={KorNLI and KorSTS: New Benchmark Datasets for Korean Natural Language Understanding},
  author={Ham, Jiyeon and Choe, Yo Joong and Park, Kyubyong and Choi, Ilji and Soh, Hyungjoon},
  journal={arXiv preprint arXiv:2004.03289},
  year={2020}
}""")
_KO_NLI_TEXT_FEATURES = {'premise':'sentence1', 'hypothesis':'sentence2'}
_KO_NLI_LABEL_CLASSES = ['entailment', 'neutral', 'contradiction']
_KO_NLI_LABEL_COL = 'gold_label' # (0: entailment, 1: neutral, 2: contradiction)
_KO_NLI_DESCRIPTION = textwrap.dedent("""\
KorNLI
...
""")
_KO_NLI_SPLIT = {
  'source': {
    tfds.Split.TRAIN: ['mnli_train', 'snli_train'],
    tfds.Split.VALIDATION: ['xnli_dev'],
    tfds.Split.TEST: ['xnli_test'],
  },
  'split': {
    tfds.Split.TRAIN: lambda x: True,
    tfds.Split.VALIDATION: lambda x: True,
    tfds.Split.TEST: lambda x: True,
}}


# STS: https://github.com/kakaobrain/KorNLUDatasets
_KO_STS_URL='https://github.com/kakaobrain/KorNLUDatasets'
_KO_STS_ROOT='https://github.com/kakaobrain/KorNLUDatasets/raw/master/KorSTS'
_KO_STS_TRAIN_LINK = os.path.join(_KO_STS_ROOT, 'sts-train.tsv')
_KO_STS_DEV_LINK = os.path.join(_KO_STS_ROOT, 'sts-dev.tsv')
_KO_STS_TEST_LINK = os.path.join(_KO_STS_ROOT, 'sts-test.tsv')
_KO_STS_DEFAULT_SPLIT={
  'train': _KO_STS_TRAIN_LINK,
  'dev': _KO_STS_DEV_LINK,
  'test': _KO_STS_TEST_LINK}
_KO_STS_CITATION=textwrap.dedent("""\
@article{ham2020kornli,
  title={KorNLI and KorSTS: New Benchmark Datasets for Korean Natural Language Understanding},
  author={Ham, Jiyeon and Choe, Yo Joong and Park, Kyubyong and Choi, Ilji and Soh, Hyungjoon},
  journal={arXiv preprint arXiv:2004.03289},
  year={2020}
}""")
_KO_STS_TEXT_FEATURES = {'sentence1':'sentence1', 'sentence2':'sentence2'}
_KO_STS_LABEL_CLASSES = None
_KO_STS_LABEL_COL = 'score' # (0-5)
_KO_STS_DESCRIPTION = textwrap.dedent("""\
KorSTS
...
""")
_KO_STS_SPLIT = {
  'source': {
    tfds.Split.TRAIN: ['train'],
    tfds.Split.VALIDATION: ['dev'],
    tfds.Split.TEST: ['test'],
  },
  'split': {
    tfds.Split.TRAIN: lambda x: True,
    tfds.Split.VALIDATION: lambda x: True,
    tfds.Split.TEST: lambda x: True,
}}

# Korean HateSpeech Dataset: https://github.com/kocohub/korean-hate-speech
_KO_KHSD_URL='https://github.com/kocohub/korean-hate-speech'
_KO_KHSD_ROOT='https://github.com/kocohub/korean-hate-speech/raw/master/'
_KO_KHSD_TRAIN_LINK = os.path.join(_KO_KHSD_ROOT, 'labeled', 'train.tsv')
_KO_KHSD_DEV_LINK = os.path.join(_KO_KHSD_ROOT, 'labeled', 'dev.tsv')
_KO_KHSD_DEFAULT_SPLIT={
  'train': _KO_KHSD_TRAIN_LINK,
  'dev': _KO_KHSD_DEV_LINK}
_KO_KHSD_CITATION = textwrap.dedent("""\
@inproceedings{moon-etal-2020-beep,
    title = "{BEEP}! {K}orean Corpus of Online News Comments for Toxic Speech Detection",
    author = "Moon, Jihyung  and
      Cho, Won Ik  and
      Lee, Junbum",
    booktitle = "Proceedings of the Eighth International Workshop on Natural Language Processing for Social Media",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.socialnlp-1.4",
    pages = "25--31",
    abstract = "Toxic comments in online platforms are an unavoidable social issue under the cloak of anonymity. Hate speech detection has been actively done for languages such as English, German, or Italian, where manually labeled corpus has been released. In this work, we first present 9.4K manually labeled entertainment news comments for identifying Korean toxic speech, collected from a widely used online news platform in Korea. The comments are annotated regarding social bias and hate speech since both aspects are correlated. The inter-annotator agreement Krippendorff{'}s alpha score is 0.492 and 0.496, respectively. We provide benchmarks using CharCNN, BiLSTM, and BERT, where BERT achieves the highest score on all tasks. The models generally display better performance on bias identification, since the hate speech detection is a more subjective issue. Additionally, when BERT is trained with bias label for hate speech detection, the prediction score increases, implying that bias and hate are intertwined. We make our dataset publicly available and open competitions with the corpus and benchmarks.",
}""")
_KO_KHSD_TEXT_FEATURES = {'sentence':'comments'}
_KO_KHSD_LABEL_CLASSES = ['hate', 'offensive', 'none']
_KO_KHSD_LABEL_COL = 'hate'
_KO_KHSD_ADD_FEAT={
  'bias': {
    'key': 'bias',
    'feature': tfds.features.ClassLabel(
      names=['gender', 'others', 'none'])
  }
}
_KO_KHSD_DESCRIPTION = textwrap.dedent("""\
Korean HateSpeech Dataset
...
""")
_KO_KHSD_SPLIT = {
  'source': {
    tfds.Split.TRAIN: ['train'],
    tfds.Split.VALIDATION: ['train'],
    tfds.Split.TEST: ['dev'],
  },
  'split': {
    tfds.Split.TRAIN: lambda x: x % 10 != 0,
    tfds.Split.VALIDATION: lambda x: x % 10 == 0,
    tfds.Split.TEST: lambda x: True,
}}
# ---------------------------------------------

def _update_split(file_dict, split_dict):
  source_dict = split_dict['source']
  return_dict = {}
  for k, v in source_dict.items():
    flist = []
    for vv in v:
      flist.extend(file_dict[vv] if isinstance(file_dict[vv], list) else [file_dict[vv]])
    return_dict[k] = flist
  return return_dict

def _hash_text(text):
  return hashlib.md5(tf.compat.as_text(text).encode("utf-8")).hexdigest()

def _filter_fn_hash_id(uid, split_fn):
  hash_id = _hash_text(str(uid))
  val = int(hash_id, 16)
  return split_fn(val)

def _get_additional_feat_dict(additional_feat):
  return {k:v['feature'] for k, v in additional_feat.items()}

def _get_feat_dict(additional_feat, row):
  return {k:row[v['key']] for k, v in additional_feat.items()}

class KorCorporaConfig(tfds.core.BuilderConfig):
  def __init__( self,
                name,
                text_features,
                label_column,
                data_url,
                data_dir,
                citation,
                url,
                label_classes=None,
                process_label=lambda x: x,
                additional_feat=None,
                manual_split=None,
                **kwargs):
    super(KorCorporaConfig, self).__init__(
      name=name,
      version=_VERSION,
      **kwargs
    )
    self.text_features=text_features
    self.label_column=label_column
    self.data_url = data_url
    self.data_dir = data_dir
    self.citation = citation
    self.url = url
    self.label_classes = label_classes
    self.process_label = process_label
    self.additional_feat = additional_feat
    self.manual_split = manual_split


class KorCorpora(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for kor_corpora dataset."""
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  BUILDER_CONFIGS = [
    KorCorporaConfig(
      name="nsmc",
      text_features=_KO_NSMC_TEXT_FEATURES,
      label_column=_KO_NSMC_LABEL_COL,
      data_url=_KO_NSMC_DEFAULT_SPLIT,
      data_dir='nsmc',
      citation=_KO_NSMC_CITATION,
      url=_KO_NSMC_URL,
      label_classes=_KO_NSMC_LABEL_CLASSES,
      description=_KO_NSMC_DESCRIPTION),
    KorCorporaConfig(
      name="nsmc.split",
      text_features=_KO_NSMC_TEXT_FEATURES,
      label_column=_KO_NSMC_LABEL_COL,
      data_url=_KO_NSMC_DEFAULT_SPLIT,
      data_dir='nsmc',
      citation=_KO_NSMC_CITATION,
      url=_KO_NSMC_URL,
      label_classes=_KO_NSMC_LABEL_CLASSES,
      description=_KO_NSMC_DESCRIPTION,
      manual_split=_KO_NSMC_SPLIT),
    KorCorporaConfig(
      name="qpair",
      text_features=_KO_QPAIR_TEXT_FEATURES,
      label_column=_KO_QPAIR_LABEL_COL,
      data_url=_KO_QPAIR_DEFAULT_SPLIT,
      data_dir='qpair',
      citation=_KO_QPAIR_CITATION,
      url=_KO_QPAIR_URL,
      label_classes=_KO_QPAIR_LABEL_CLASSES,
      description=_KO_QPAIR_DESCRIPTION),
    KorCorporaConfig(
      name="qpair.split",
      text_features=_KO_QPAIR_TEXT_FEATURES,
      label_column=_KO_QPAIR_LABEL_COL,
      data_url=_KO_QPAIR_DEFAULT_SPLIT,
      data_dir='qpair',
      citation=_KO_QPAIR_CITATION,
      url=_KO_QPAIR_URL,
      label_classes=_KO_QPAIR_LABEL_CLASSES,
      description=_KO_QPAIR_DESCRIPTION,
      manual_split=_KO_QPAIR_SPLIT),
    KorCorporaConfig(
      name="kornli",
      text_features=_KO_NLI_TEXT_FEATURES,
      label_column=_KO_NLI_LABEL_COL,
      data_url=_KO_NLI_DEFAULT_SPLIT,
      data_dir='kornli',
      citation=_KO_NLI_CITATION,
      url=_KO_NLI_URL,
      label_classes=_KO_NLI_LABEL_CLASSES,
      description=_KO_NLI_DESCRIPTION),
    KorCorporaConfig(
      name="kornli.split",
      text_features=_KO_NLI_TEXT_FEATURES,
      label_column=_KO_NLI_LABEL_COL,
      data_url=_KO_NLI_DEFAULT_SPLIT,
      data_dir='kornli',
      citation=_KO_NLI_CITATION,
      url=_KO_NLI_URL,
      label_classes=_KO_NLI_LABEL_CLASSES,
      description=_KO_NLI_DESCRIPTION,
      manual_split=_KO_NLI_SPLIT),
    KorCorporaConfig(
      name="korsts",
      text_features=_KO_STS_TEXT_FEATURES,
      label_column=_KO_STS_LABEL_COL,
      data_url=_KO_STS_DEFAULT_SPLIT,
      data_dir='korsts',
      citation=_KO_STS_CITATION,
      url=_KO_STS_URL,
      label_classes=_KO_STS_LABEL_CLASSES,
      description=_KO_STS_DESCRIPTION,
      process_label=np.float32),
    KorCorporaConfig(
      name="korsts.split",
      text_features=_KO_STS_TEXT_FEATURES,
      label_column=_KO_STS_LABEL_COL,
      data_url=_KO_STS_DEFAULT_SPLIT,
      data_dir='korsts',
      citation=_KO_STS_CITATION,
      url=_KO_STS_URL,
      label_classes=_KO_STS_LABEL_CLASSES,
      description=_KO_STS_DESCRIPTION,
      process_label=np.float32,
      manual_split=_KO_STS_SPLIT),
    KorCorporaConfig(
      name="khsd",
      text_features=_KO_KHSD_TEXT_FEATURES,
      label_column=_KO_KHSD_LABEL_COL,
      data_url=_KO_KHSD_DEFAULT_SPLIT,
      data_dir='khsd',
      citation=_KO_KHSD_CITATION,
      url=_KO_KHSD_URL,
      label_classes=_KO_KHSD_LABEL_CLASSES,
      description=_KO_KHSD_DESCRIPTION,
      additional_feat=_KO_KHSD_ADD_FEAT),
    KorCorporaConfig(
      name="khsd.split",
      text_features=_KO_KHSD_TEXT_FEATURES,
      label_column=_KO_KHSD_LABEL_COL,
      data_url=_KO_KHSD_DEFAULT_SPLIT,
      data_dir='khsd',
      citation=_KO_KHSD_CITATION,
      url=_KO_KHSD_URL,
      label_classes=_KO_KHSD_LABEL_CLASSES,
      description=_KO_KHSD_DESCRIPTION,
      additional_feat=_KO_KHSD_ADD_FEAT,
      manual_split=_KO_KHSD_SPLIT),
  ]

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    features = {
        text_feature: tfds.features.Text()
        for text_feature in self.builder_config.text_features.keys()
    }
    if self.builder_config.label_classes:
      features["label"] = tfds.features.ClassLabel(
          names=self.builder_config.label_classes)
    else:
      features["label"] = tf.float32
    features["idx"] = tf.int32
    if self.builder_config.additional_feat is not None:
      features.update(_get_additional_feat_dict(self.builder_config.additional_feat))

    return tfds.core.DatasetInfo(
        builder=self,
        description=self.builder_config.description,
        features=tfds.features.FeaturesDict(features),
        homepage=self.builder_config.url,
        citation=self.builder_config.citation,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    path = dl_manager.download_and_extract(self.builder_config.data_url)

    if self.builder_config.manual_split is not None:
      path = _update_split(path, self.builder_config.manual_split)
      split_fn = self.builder_config.manual_split['split']
      return {k:self._generate_examples(v, split_fn[k]) for k, v in path.items()}

    # TODO(kor_corpora): Returns the Dict[split names, Iterator[Key, Example]]
    return {k:self._generate_examples(v) for k, v in path.items()}

  def _generate_examples(self, path_list, split_fn=None):
    """Yields examples."""
    process_label = self.builder_config.process_label
    label_classes = self.builder_config.label_classes
    additional_feat = self.builder_config.additional_feat

    if split_fn is not None:
      split_filter = functools.partial(_filter_fn_hash_id, split_fn=split_fn)
    else:
      split_filter = lambda x: True

    if not isinstance(path_list, list):
      path_list = [path_list]

    _hash_set = set()
    idx = 0

    for path in path_list:
      with tf.io.gfile.GFile(path) as f:
        if self.builder_config.name.startswith('qpair'):
          reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        else:
          reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
          example = {
            k: row[col] for k, col in self.builder_config.text_features.items()
          }
          example['idx'] = idx
          idx += 1
          if self.builder_config.label_column in row:
            label = row[self.builder_config.label_column]
            if label_classes and label not in label_classes:
              label = int(label) if label else None
            example["label"] = process_label(label)
            if additional_feat is not None:
              example.update(_get_feat_dict(additional_feat, row))
          else:
            example["label"] = process_label(-1)

          # Filter out corrupted rows.
          for value in example:
            if value is None:
              break
          else:
            if split_filter(str(example['idx'])) and str(example['idx']) not in _hash_set:
              _hash_set.add(str(example['idx']))
              yield example['idx'], example

# KorCorpora
# tfds build --data_dir ../../tmp/tensorflow_datasets --config nsmc