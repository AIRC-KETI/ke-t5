# coding=utf-8
# Copyright 2021 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# modified by kimsan0622@keti.re.kr
"""korquad dataset."""
import os
import json
import copy
import hashlib
import functools

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.question_answering import qa_utils

# KorQuad: https://korquad.github.io/
# ---------------------------------------------
_KORQUAD_URL='https://korquad.github.io/'
# https://github.com/korquad/korquad.github.io/raw/master/dataset/KorQuAD_v1.0_dev.json
_KORQUAD_ROOT='https://github.com/korquad/korquad.github.io/raw/master/dataset/'
_KORQUADV1_TRAIN_LINK=[os.path.join(_KORQUAD_ROOT, 'KorQuAD_v1.0_train.json')]
_KORQUADV1_DEV_LINK=[os.path.join(_KORQUAD_ROOT, 'KorQuAD_v1.0_dev.json')]
_KORQUADV1_DEFAULT_SPLIT={'train': _KORQUADV1_TRAIN_LINK, 'dev': _KORQUADV1_DEV_LINK}
_KORQUADV1_DESCRIPTION = """
KorQuAD1.0
"""
_KORQUADV1_CITATION = """
@article{DBLP:journals/corr/abs-1909-07005,
  author    = {Seungyoung Lim and
               Myungji Kim and
               Jooyoul Lee},
  title     = {KorQuAD1.0: Korean {QA} Dataset for Machine Reading Comprehension},
  journal   = {CoRR},
  volume    = {abs/1909.07005},
  year      = {2019},
  url       = {http://arxiv.org/abs/1909.07005},
  archivePrefix = {arXiv},
  eprint    = {1909.07005},
  timestamp = {Mon, 23 Sep 2019 18:07:15 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1909-07005.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

# https://github.com/korquad/korquad.github.io/raw/master/dataset/KorQuAD_2.1/train/KorQuAD_2.1_train_00.zip
_KORQUADV2_TRAIN_LINK=[os.path.join(_KORQUAD_ROOT,'KorQuAD_2.1/train', 'KorQuAD_2.1_train_{0:02d}.zip'.format(idx)) for idx in range(13)]
_KORQUADV2_DEV_LINK=[os.path.join(_KORQUAD_ROOT,'KorQuAD_2.1/dev', 'KorQuAD_2.1_dev_{0:02d}.zip'.format(idx)) for idx in range(2)]
_KORQUADV2_DEFAULT_SPLIT={'train': _KORQUADV2_TRAIN_LINK, 'dev': _KORQUADV2_DEV_LINK}
_KORQUADV2_DESCRIPTION = """
KorQuAD2.1
"""
_KORQUADV2_CITATION = """
김영민, 임승영, 이현정, 박소윤, 김명지. (2020). KorQuAD 2.0: 웹문서 기계독해를 위한 한국어 질의응답 데이터셋. 정보과학회논문지, 47(6), 577-586.
"""

_KORQUADV2_KEY_MAP={'context':'context', 'answer_start': 'answer_start', 'text':'text'}
_KORQUADV2_HTML_KEY_MAP={'context':'raw_html', 'answer_start': 'html_answer_start', 'text':'html_answer_text'}

def generate_korquadv2_examples(filepath, KEY_MAP):
  qas = {}
  with tf.io.gfile.GFile(filepath) as f:
    squad = json.load(f)
    for article in squad["data"]:
      title = article.get("title", "").strip()
      context = article[KEY_MAP['context']]
      for qa in article["qas"]:
        qa["title"] = title
        qa["context"] = context
        id_ = qa["id"]
        qa["answers"] = [copy.deepcopy(qa["answer"])]
        del qa["answer"]
        qas[id_] = qa

    for id_, qa in qas.items():
      answer_starts = [answer[KEY_MAP['answer_start']] for answer in qa["answers"]]
      answers = [answer[KEY_MAP['text']] for answer in qa["answers"]]
      yield id_, {
          "title": qa["title"],
          "context": qa["context"],
          "question": qa["question"].strip(),
          "id": id_,
          "answers": {
              "answer_start": answer_starts,
              "text": answers,
          },
      }

_KORQUAD_MANUAL_SPLIT = {
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


_VERSION = tfds.core.Version('1.0.0')

class KorquadConfig(tfds.core.BuilderConfig):
  def __init__( self,
                name,
                data_url,
                description,
                citation,
                manual_split=None,
                **kwargs):
    super(KorquadConfig, self).__init__(
      name=name,
      version=_VERSION,
      **kwargs
    )
    self.data_url=data_url
    self.description=description
    self.citation=citation
    self.manual_split=manual_split

class Korquad(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for korquad dataset."""
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  BUILDER_CONFIGS = [
    KorquadConfig(
      'v1.0',
      data_url=_KORQUADV1_DEFAULT_SPLIT,
      description=_KORQUADV1_DESCRIPTION,
      citation=_KORQUADV1_CITATION,
    ),
    KorquadConfig(
      'v1.0.split',
      data_url=_KORQUADV1_DEFAULT_SPLIT,
      description=_KORQUADV1_DESCRIPTION,
      citation=_KORQUADV1_CITATION,
      manual_split=_KORQUAD_MANUAL_SPLIT,
    ),
    KorquadConfig(
      'v2.1',
      data_url=_KORQUADV2_DEFAULT_SPLIT,
      description=_KORQUADV2_DESCRIPTION,
      citation=_KORQUADV2_CITATION,
    ),
    KorquadConfig(
      'v2.1.split',
      data_url=_KORQUADV2_DEFAULT_SPLIT,
      description=_KORQUADV2_DESCRIPTION,
      citation=_KORQUADV2_CITATION,
      manual_split=_KORQUAD_MANUAL_SPLIT,
    ),
    KorquadConfig(
      'v2.1.html',
      data_url=_KORQUADV2_DEFAULT_SPLIT,
      description=_KORQUADV2_DESCRIPTION,
      citation=_KORQUADV2_CITATION,
    ),
    KorquadConfig(
      'v2.1.html.split',
      data_url=_KORQUADV2_DEFAULT_SPLIT,
      description=_KORQUADV2_DESCRIPTION,
      citation=_KORQUADV2_CITATION,
      manual_split=_KORQUAD_MANUAL_SPLIT,
    ),
  ]

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    features_dict = qa_utils.SQUADLIKE_FEATURES

    return tfds.core.DatasetInfo(
        builder=self,
        description=self.builder_config.description,
        features=features_dict,
        homepage=_KORQUAD_URL,
        citation=self.builder_config.citation,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""

    path_kv = {k:dl_manager.download_and_extract(v) for k, v in self.builder_config.data_url.items()}
    if not self.builder_config.name.startswith("v1.0"):
      for k, v in path_kv.items():
        file_names = []
        for vv in v:
          file_names.extend(tf.io.gfile.glob(os.path.join(vv, "*.json")))
        path_kv[k] = file_names

    if self.builder_config.manual_split is not None:
      path_kv = _update_split(path_kv, self.builder_config.manual_split)
      split_fn = self.builder_config.manual_split['split']
      return {k:self._generate_examples(v, split_fn[k]) for k, v in path_kv.items()}

    # TODO(korquad): Returns the Dict[split names, Iterator[Key, Example]]
    return {k:self._generate_examples(v) for k, v in path_kv.items()}

  def _generate_examples(self, path_list, split_fn=None):
    """Yields examples."""
    # TODO(korquad): Yields (key, example) tuples from the dataset
    if self.builder_config.name.startswith("v2.1.html"):
      gen_fn = functools.partial(generate_korquadv2_examples, KEY_MAP=_KORQUADV2_HTML_KEY_MAP)
    elif self.builder_config.name.startswith("v2.1"):
      gen_fn = functools.partial(generate_korquadv2_examples, KEY_MAP=_KORQUADV2_KEY_MAP)
    else:
      gen_fn = qa_utils.generate_squadlike_examples
    
    if split_fn is not None:
      split_filter = functools.partial(_filter_fn_hash_id, split_fn=split_fn)
    else:
      split_filter = lambda x: True
    
    _hash_set = set()

    for fpath in path_list:
      for example in iter(gen_fn(fpath)):
        uid, _ = example
        if split_filter(str(uid)) and str(uid) not in _hash_set:
            _hash_set.add(str(uid))
            yield example

# tfds build --data_dir ../../tmp/tensorflow_datasets --config v1.0.split