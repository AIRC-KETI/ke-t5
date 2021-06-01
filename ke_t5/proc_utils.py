import functools
import collections

import seqio
import tensorflow as tf
from t5.data import preprocessors
from t5.data import postprocessors
from seqio.utils import map_over_dataset
from t5.evaluation import metrics
import t5.data.glue_utils

import numpy as np

from ke_t5.default_vocab import DEFAULT_VOCAB

def ndarray2string(ndarray):
    return DEFAULT_VOCAB._decode(ndarray.tolist())

def check_string(string, **unused_kwargs):
    if isinstance(string, np.ndarray):
        return ndarray2string(string)
    elif isinstance(string, str):
        return string
    raise TypeError("string(must be str of ndarray): {}".format(string))

def string_to_float(string, default=-1., **unused_kwargs):
    string = check_string(string)
    try:
        return float(string)
    except ValueError:
        return default
    except TypeError as e:
        raise TypeError("TypeError - string: {}, type: {}".format(string, type(string)))


def string_label_to_class_id(string_label, label_classes, default=-1, **unused_kwargs):
    string_label = check_string(string_label)
    if string_label in label_classes:
        return label_classes.index(string_label)
    else:
        return default

def multirc(string_label, example=None, is_target=False):
  """Returns dict containing the class with the question index for grouping."""
  res = {
      "value":
          string_label_to_class_id(
              string_label, label_classes=("False", "True"), example=example)
  }
  # Add the group, if present, since the model outputs will not have it.
  if is_target:
    res["group"] = example["idx/question"]
  return res


def qa(answer, example=None, is_target=False):
  """Returns answer, or all answers if the full example is provided."""
  if is_target:
    return [tf.compat.as_text(a) for a in example["answers"]]
  return answer

def get_glue_postprocess_fn(builder_config):
    if builder_config.name == "stsb":
        return string_to_float
    elif builder_config.name == "multirc":
        return multirc
    elif builder_config.name == "record":
        return qa
    else:
        return functools.partial(
            string_label_to_class_id,
            label_classes=builder_config.label_classes,
        )

@map_over_dataset
def summarize_split(x, article_key, summary_key, summary_type='summary'):
  """Convert a summarization dataset to a text2text pair.

  For example, say the dataset returns examples of this format:
    {'article': <article>, 'highlights': <summary>}
  If article_key = 'article', summary_key = 'highlights', then the outputs will
  have the format:
    {'inputs': 'summarize': <article>, 'targets': <summary>}

  Args:
    x: an example to process.
    article_key: the feature key for the article to summarize.
    summary_key: the feature key for the target summary.
  Returns:
    A preprocessed example with the format listed above.
  """
  strs_to_join = ['summarize_{}:'.format(summary_type), x[article_key]]
  return {
      'inputs': tf.strings.join(strs_to_join, separator=' '),
      'targets': x[summary_key],
  }


_KO_NSMC_LABEL_CLASSES = ['negative', 'positive']
_KO_NSMC_FEATURE_NAMES = ('sentence',)

_KO_QPAIR_LABEL_CLASSES = ['duplicated', 'not_duplicate']
_KO_QPAIR_FEATURE_NAMES = ('question1', 'question2',)

_KO_NLI_LABEL_CLASSES = ['entailment', 'neutral', 'contradiction']
_KO_NLI_FEATURE_NAMES = ('premise', 'hypothesis',)

_KO_KHSD_LABEL_CLASSES = ['hate', 'offensive', 'none']
_KO_KHSD_FEATURE_NAMES = ('sentence',)

_KO_COLA_LABEL_CLASSES = ["unacceptable", "acceptable"]
_KO_COLA_FEATURE_NAMES = ('sentence',)

#a = (nsmc.split, 134799, qpair, 6136, nil, 942854, sts, 5749, khsd, 7065, cola, 15876)
KO_WEIGHT_MAPPING = {
    "ke_t5_kor_copora_nsmc": 134_799.,
    "ke_t5_kor_copora_qpair": 6_136.,
    "ke_t5_kor_copora_kornli": 942_854.,
    "ke_t5_kor_copora_korsts": 5_749.,
    "ke_t5_kor_copora_khsd": 7_065.,
    "ke_t5_nikl_cola": 15_876.,
    "ke_t5_korquad_allanswers": 54_477.,
    "ke_t5_nikl_summarization": 6_952.,
}

KET5_KO_EN_WEIGHT_MAPPING = {
    "ke_t5_ted_multi_en_ko": 258_098.,
    "ke_t5_ted_multi_ko_en": 258_098.,
}

KET5_EN_WEIGHT_MAPPING = {
    "ke_t5_squad_v010_allanswers": 87_599.,
    "ke_t5_cnn_dailymail_v002": 287_113.,
    "ke_t5_trivia_qa_v010": 87_622.,
}



KET5_GLUE_WEIGHT_MAPPING = {
    f"ke_t5_{k}": v for k, v in t5.data.glue_utils.get_glue_weight_mapping().items()}
KET5_SUPER_GLUE_WEIGHT_MAPPING = {
    f"ke_t5_{k}": v for k, v in t5.data.glue_utils.get_super_glue_weight_mapping().items()}

KET5_ALL_WEIGHT_MAPPING = {**KO_WEIGHT_MAPPING, **KET5_KO_EN_WEIGHT_MAPPING, **KET5_EN_WEIGHT_MAPPING, **KET5_GLUE_WEIGHT_MAPPING, **KET5_SUPER_GLUE_WEIGHT_MAPPING}

_KO_TEXT_CLASSIFICATION = [
    "ke_t5_kor_copora_nsmc",
    "ke_t5_kor_copora_qpair",
    "ke_t5_kor_copora_kornli",
    "ke_t5_kor_copora_korsts",
    "ke_t5_kor_copora_khsd",
    "ke_t5_nikl_cola"
]

def get_ket5_ko_text_classification():
    return _KO_TEXT_CLASSIFICATION

def get_ket5_super_glue_weight_mapping():
    return KET5_SUPER_GLUE_WEIGHT_MAPPING

def get_ket5_glue_weight_mapping():
    return KET5_GLUE_WEIGHT_MAPPING

def get_ket5_all_weight_mapping():
    return KET5_ALL_WEIGHT_MAPPING

def dedupe(name):
    rate = None
    if name in KET5_ALL_WEIGHT_MAPPING:
        rate = KET5_ALL_WEIGHT_MAPPING[name]
    if "glue" in name and "rte" in name:
        rate *= 0.5
    return rate


KOR_CORPORA_DICT = {
    'nsmc': {
        'label_names': _KO_NSMC_LABEL_CLASSES,
        'feature_names': _KO_NSMC_FEATURE_NAMES
    },
    'qpair': {
        'label_names': _KO_QPAIR_LABEL_CLASSES,
        'feature_names': _KO_QPAIR_FEATURE_NAMES
    },
    'kornli': {
        'label_names': _KO_NLI_LABEL_CLASSES,
        'feature_names': _KO_NLI_FEATURE_NAMES
    },
    'khsd': {
        'label_names': _KO_KHSD_LABEL_CLASSES,
        'feature_names': _KO_KHSD_FEATURE_NAMES
    },
    'korsts': None,
}

_KO_TEXT_CLASSIFICATION_TASKS = []


def get_kor_copora_preprocess_fn(task_name):
    if task_name not in KOR_CORPORA_DICT:
        raise AssertionError(
            "can't find preprocess job for {0} in get_kor_copora_preprocess_fn!".format(task_name))
    if task_name == 'korsts':
        return korsts
    else:
        return functools.partial(
            preprocessors.glue,
            benchmark_name=task_name,
            label_names=KOR_CORPORA_DICT[task_name]['label_names'],
            feature_names=KOR_CORPORA_DICT[task_name]['feature_names'])


def nikl_cola_preprocess_fn():
    return functools.partial(
        preprocessors.glue,
        benchmark_name='niklcola',
        label_names=_KO_COLA_LABEL_CLASSES,
        feature_names=_KO_COLA_FEATURE_NAMES)


def nikl_cola_postprocess_fn():
    return functools.partial(
        string_label_to_class_id,
        label_classes=_KO_COLA_LABEL_CLASSES)


@map_over_dataset
def korsts(x):
    strs_to_join = [
        'korsts sentence1:', x['sentence1'], 'sentence2:', x['sentence2']
    ]
    label_string = tf.as_string(tf.round(x['label'] * 5) / 5, precision=1)
    joined = tf.strings.join(strs_to_join, separator=' ')
    return {'inputs': joined, 'targets': label_string, 'idx': x['idx']}


def get_kor_copora_postprocess_fn(task_name):
    if task_name not in KOR_CORPORA_DICT:
        raise AssertionError(
            "can't find preprocess job for {0} in get_kor_copora_preprocess_fn!".format(task_name))
    if task_name == 'korsts':
        return string_to_float
    else:
        return functools.partial(
            string_label_to_class_id,
            label_classes=KOR_CORPORA_DICT[task_name]['label_names'],
        )


_KOR_CORPORA_METRICS = collections.OrderedDict([
    ("nsmc", [metrics.accuracy]),
    ("qpair", [metrics.f1_score_with_invalid, metrics.accuracy]),
    ("kornli", [metrics.accuracy]),
    ("korsts", [metrics.pearson_corrcoef, metrics.spearman_corrcoef]),
    ("khsd", [metrics.accuracy]),
])


def get_kor_corpora_metric(task_name):
    return _KOR_CORPORA_METRICS[task_name]


def _pad_punctuation_kor(text):
    # Add space around punctuation. Hangul Syllable(\uAC00-\uD7AF)
    text = tf.strings.regex_replace(
        text, '([^A-Za-z0-9_\uAC00-\uD7AF])', r' \1 ')
    # Collapse consecutive whitespace into one space.
    text = tf.strings.regex_replace(text, r'\s+', ' ')
    return text

# imported from multilingual t5 github


def _pad_punctuation_general(text):
    # Pad everything except for: underscores (_), whitespace (\s),
    # numbers (\p{N}), letters (\p{L}) and accent characters (\p{M}).
    text = tf.strings.regex_replace(text, r'([^_\s\p{N}\p{L}\p{M}])', r' \1 ')
    # Collapse consecutive whitespace into one space.
    text = tf.strings.regex_replace(text, r'\s+', ' ')
    return text

# imported from multilingual t5 github


def _string_join(lst):
    # Join on space, but collapse consecutive spaces.
    out = tf.strings.join(lst, separator=' ')
    return tf.strings.regex_replace(out, r'\s+', ' ')


def _escape_html(text):
    # &apos; -> '
    # &lt; -> <
    # &gt; -> >
    # &amp; -> &
    # &quot; -> "
    for reg, rep in [('[\s]*&apos;', '\''), ('&lt;', '<'), ('&gt;', '>'), ('&amp;', '&'), ('&quot;', '"')]:
        text = tf.strings.regex_replace(text, reg, rep)
    text = tf.strings.regex_replace(text, r'\s+', ' ')
    return text


@map_over_dataset
def preprocess_korquad(x, include_context=True):
    a = _pad_punctuation_general(x['answers']['text'])
    q = _pad_punctuation_general(x['question'])
    c = _pad_punctuation_general(x['context'])
    if include_context:
        inputs = _string_join(['question:', q, 'context:', c])
    else:
        inputs = _string_join(['korquad trivia question:', q])
    return {
        'inputs': inputs,
        'targets': a[0],
        'id': x['id'],
        'context': c,
        'question': q,
        'answers': a
    }


def _ted_multi_translate_preprocess(dataset, source_language, target_language):
    def _grep_languages(x, source_language, target_language):
        languages = x['translations']['language']
        translations = x['translations']['translation']

        src_script = tf.gather_nd(translations, tf.where(
            tf.math.equal(tf.constant(source_language), languages)))
        tgt_script = tf.gather_nd(translations, tf.where(
            tf.math.equal(tf.constant(target_language), languages)))

        return {
            source_language: src_script,
            target_language: tgt_script,
        }

    def _process(x, source_language, target_language):
        src_str = 'translate {}'.format(source_language)
        tgt_str = ' to {}: '.format(target_language)
        src_trans = _escape_html(x[source_language][0])
        tgt_trans = _escape_html(x[target_language][0])
        return {
            'inputs': tf.strings.join([src_str, tgt_str, src_trans]),
            'targets': tgt_trans,
        }

    dataset = dataset.map(
        functools.partial(_grep_languages, source_language=source_language, target_language=target_language), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.filter(
        lambda x: tf.size(x[source_language]) > 0 and tf.size(x[target_language]) > 0)
    dataset = dataset.map(
        functools.partial(_process, source_language=source_language, target_language=target_language), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return dataset



_KLUE_TC_CLASSES = [
    '정치',  # politics
    '경제',  # economy
    '사회',  # society
    '생활문화',  # culture
    '세계',  # world
    'IT과학',  # IT/science
    '스포츠',  # sports
    '해당없음'  # OOD(out-of-distribution)
]

_KLUE_NLI_CLASSES = [
    'entailment',
    'contradiction',
    'neutral'
]

_KLUE_NER_TAGS = [
    'PS',  # person
    'LC',  # location
    'OG',  # organization
    'DT',  # date
    'TI',  # time
    'QT'  # quantity
]

_KLUE_NER_IOB2_TAGS = [
    'O',
    'B-PS',
    'I-PS',
    'B-LC',
    'I-LC',
    'B-OG',
    'I-OG',
    'B-DT',
    'I-DT',
    'B-TI',
    'I-TI',
    'B-QT',
    'I-QT',
]

_KLUE_RE_RELATIONS = [
    "no_relation",
    "org:dissolved",
    "org:founded",
    "org:place_of_headquarters",
    "org:alternate_names",
    "org:member_of",
    "org:members",
    "org:political/religious_affiliation",
    "org:product",
    "org:founded_by",
    "org:top_members/employees",
    "org:number_of_employees/members",
    "per:date_of_birth",
    "per:date_of_death",
    "per:place_of_birth",
    "per:place_of_death",
    "per:place_of_residence",
    "per:origin",
    "per:employee_of",
    "per:schools_attended",
    "per:alternate_names",
    "per:parents",
    "per:children",
    "per:siblings",
    "per:spouse",
    "per:other_family",
    "per:colleagues",
    "per:product",
    "per:religion",
    "per:title"
]

_KLUE_RE_ENTITY_TYPE = [
    "PER",
    "ORG",
    "POH",
    "DAT",
    "LOC",
    "NOH"
]

_KLUE_DP_SYNTAX = [
    "NP",  # Noun Phrase
    "VP",  # Verb Phrase
    "AP",  # Adverb Phrase
    "VNP",  # Copula Phrase
    "DP",  # Adnoun Phrase
    "IP",  # Interjection Phrase
    "X",  # Pseudo Phrase
    "L",  # Left Parenthesis and Quotation Mark
    "R"  # Right Parenthesis and Quotation Mark
]
_KLUE_DP_FUNC = [
    "SBJ",  # Subject
    "OBJ",  # Object
    "MOD",  # Noun Modifier
    "AJT",  # Predicate Modifier
    "CMP",  # Complement
    "CNJ",  # Conjunction
]

_KLUE_DP_DEPREL_TAGS = [
    "NP",
    "NP_AJT",
    "VP",
    "NP_SBJ",
    "VP_MOD",
    "NP_OBJ",
    "AP",
    "NP_CNJ",
    "NP_MOD",
    "VNP",
    "DP",
    "VP_AJT",
    "VNP_MOD",
    "NP_CMP",
    "VP_SBJ",
    "VP_CMP",
    "VP_OBJ",
    "VNP_CMP",
    "AP_MOD",
    "X_AJT",
    "VNP_AJT",
    "VP_CNJ",
    "IP",
    "X",
    "VNP_OBJ",
    "X_SBJ",
    "X_OBJ",
    "VNP_SBJ",
    "L",
    "AP_AJT",
    "X_CMP",
    "X_CNJ",
    "X_MOD",
    "AP_CMP",
    "R",
    "VNP_CNJ",
    "AP_SBJ",
    "NP_SVJ"
]

KLUE_META={
    'tc_classes': _KLUE_TC_CLASSES,
    'nli_classes': _KLUE_NLI_CLASSES,
    'ner_tags': _KLUE_NER_TAGS,
    'ner_iob2_tags': _KLUE_NER_IOB2_TAGS,
    're_relations': _KLUE_RE_RELATIONS,
    're_entity_type': _KLUE_RE_ENTITY_TYPE,
    'dp_deprels': _KLUE_DP_DEPREL_TAGS,
    'dp_syntax': _KLUE_DP_SYNTAX,
    'dp_func': _KLUE_DP_FUNC
}

@seqio.map_over_dataset
def base_preproc_for_classification(x, benchmark_name, input_keys, label_names=None, no_label_idx=0, with_feature_key=True):
    strs_to_join = []
    for key in input_keys:
        if with_feature_key:
            strs_to_join.append('{}:'.format(key))
        strs_to_join.append(x[key])

    ex = {}

    if label_names is not None:
        # put the name of benchmark if the model is generative
        strs_to_join.insert(0, benchmark_name)
        ex['targets'] = tf.cond(
            # When no label is provided (label == -1), use "<unk>"
            tf.equal(x['label'], -1),
            lambda: tf.constant('<unk>'),
            # Otherwise grab the label text from label_names
            lambda: tf.gather(label_names, x['label']),
        )
    else:
        ex['targets'] = tf.cond(
            # When no label is provided (label == -1), use no_label_idx
            tf.equal(x['label'], -1),
            lambda: tf.constant(no_label_idx, tf.int32),
            # Otherwise set the label
            lambda: x['label'],
        )

    joined = tf.strings.join(strs_to_join, separator=' ')
    ex['inputs'] = joined
    ex['id'] = x['id']

    return ex


@seqio.map_over_dataset
def base_preproc_for_regression(x, benchmark_name, input_keys, is_string_tgt=True, with_feature_key=True):
    strs_to_join = []
    for key in input_keys:
        if with_feature_key:
            strs_to_join.append('{}:'.format(key))
        strs_to_join.append(x[key])

    ex = {}

    if is_string_tgt:
        # put the name of benchmark if the model is generative
        strs_to_join.insert(0, benchmark_name)
        ex['targets'] = tf.as_string(tf.round(x['label'] * 5) / 5, precision=1)
    else:
        ex['targets'] = x['label']

    joined = tf.strings.join(strs_to_join, separator=' ')
    ex['inputs'] = joined
    ex['id'] = x['id']

    return ex


@seqio.map_over_dataset
def re_preproc_for_classification(x, benchmark_name, label_names=None, no_label_idx=0, with_feature_key=True):
    # mark span using start index of the entity
    def _mark_span(text, span_str, span_idx, mark):
        pattern_tmpl = r'^((?:[\S\s]){N})(W)'
        pattern = tf.strings.regex_replace(pattern_tmpl, 'N',
                                           tf.as_string(span_idx))
        pattern = tf.strings.regex_replace(pattern, 'W', span_str)
        return tf.strings.regex_replace(text, pattern, r'\1{0}\2{0}'.format(mark))

    # '*' for subejct entity '#' for object entity.

    text = x["sentence"]
    text = _mark_span(text, x['subject_entity']['word'],
                      x['subject_entity']['start_idx'], '*')
    # Compensate for 2 added "words" added in previous step.
    span2_index = x['object_entity']['start_idx'] + 2 * tf.cast(
        x['subject_entity']['start_idx'] < x['object_entity']['start_idx'], tf.int32)
    text = _mark_span(text, x['object_entity']['word'], span2_index, '#')

    strs_to_join = []
    if with_feature_key:
        strs_to_join.append('{}:'.format('text'))
    strs_to_join.append(text)

    ex = {}

    if label_names is not None:
        # put the name of benchmark if the model is generative
        strs_to_join.insert(0, benchmark_name)
        ex['targets'] = tf.cond(
            # When no label is provided (label == -1), use "<unk>"
            tf.equal(x['label'], -1),
            lambda: tf.constant('<unk>'),
            # Otherwise grab the label text from label_names
            lambda: tf.gather(label_names, x['label']),
        )
    else:
        ex['targets'] = tf.cond(
            # When no label is provided (label == -1), use no_label_idx
            tf.equal(x['label'], -1),
            lambda: tf.constant(no_label_idx, tf.int32),
            # Otherwise set the label
            lambda: x['label'],
        )

    joined = tf.strings.join(strs_to_join, separator=' ')
    ex['inputs'] = joined
    ex['id'] = x['id']

    return ex


@seqio.map_over_dataset
def preprocess_quad(x, benchmark_name, include_context=True, impossible_answer_text='impossible'):
    a = _pad_punctuation_general(x['answers']['text'])
    q = _pad_punctuation_general(x['question'])
    c = _pad_punctuation_general(x['context'])

    strs_to_join = []
    if include_context:
        strs_to_join.extend(['question:', q, 'context:', c])
    else:
        strs_to_join.extend(['trivia question:', q])

    strs_to_join.insert(0, benchmark_name)
    inputs = _string_join(strs_to_join)

    if 'is_impossible' in x:
        if x['is_impossible']:
            label = tf.constant(impossible_answer_text)
        else:
            label = a[0]
    else:
        label = a[0]

    return {
        'inputs': inputs,
        'targets': label,
        'id': x['id'],
        'context': c,
        'question': q,
        'answers': a
    }

def tokenize_with_offsets(
        dataset: tf.data.Dataset,
        output_features: seqio.preprocessors.OutputFeaturesType,
        copy_pretokenized: bool = True,
    ) -> tf.data.Dataset:

    def _tokenize(features):
        ret = {}
        for k, v in features.items():
            if k in output_features:
                if copy_pretokenized:
                    ret[f'{k}_pretokenized'] = v
                vocab = output_features[k].vocabulary
                (tokens, start_offsets, end_offsets) = vocab.tf_tokenizer.tokenize_with_offsets(v)
                v = tokens
                ret[f'{k}_start_offsets'] = start_offsets
                ret[f'{k}_end_offsets'] = end_offsets
            ret[k] = v
        return ret

    return dataset.map(
        _tokenize, num_parallel_calls=tf.data.experimental.AUTOTUNE)