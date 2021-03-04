
"""Add Tasks to registry."""
import functools


import t5.data
from t5.evaluation import metrics
import tensorflow_datasets as tfds


DEFAULT_SPM_PATH = "gs://ket5/vocabs/ket5.64000/sentencepiece.model"
DEFAULT_EXTRA_IDS = 100

DEFAULT_TEMPERATURE = 1.0 / 0.3
DEFAULT_MIX_RATE = functools.partial(
    t5.data.rate_num_examples, temperature=DEFAULT_TEMPERATURE)

DEFAULT_VOCAB = t5.data.SentencePieceVocabulary(
    DEFAULT_SPM_PATH, DEFAULT_EXTRA_IDS)
DEFAULT_OUTPUT_FEATURES = {
    "inputs": t5.data.Feature(
        vocabulary=DEFAULT_VOCAB, add_eos=True, required=False),
    "targets": t5.data.Feature(
        vocabulary=DEFAULT_VOCAB, add_eos=True)
}


# ==================================== KE-T5 ======================================
t5.data.TaskRegistry.add(
    "ke.ke_v100_span_corruption",
    t5.data.TfdsTask,
    tfds_name="ke_dataset/ke:1.0.0",
    text_preprocessor=functools.partial(
        t5.data.preprocessors.rekey, key_map={"inputs": None, "targets": "text"}),
    token_preprocessor=t5.data.preprocessors.span_corruption,
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[])

t5.data.TaskRegistry.add(
    "ke.ko_v100_span_corruption",
    t5.data.TfdsTask,
    tfds_name="ke_dataset/ko:1.0.0",
    text_preprocessor=functools.partial(
        t5.data.preprocessors.rekey, key_map={"inputs": None, "targets": "text"}),
    token_preprocessor=t5.data.preprocessors.span_corruption,
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[])

t5.data.TaskRegistry.add(
    "ke.ke.newslike_v100_span_corruption",
    t5.data.TfdsTask,
    tfds_name="ke_dataset/ke.newslike:1.0.0",
    text_preprocessor=functools.partial(
        t5.data.preprocessors.rekey, key_map={"inputs": None, "targets": "text"}),
    token_preprocessor=t5.data.preprocessors.span_corruption,
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[])

t5.data.TaskRegistry.add(
    "ke.ko.newslike_v100_span_corruption",
    t5.data.TfdsTask,
    tfds_name="ke_dataset/ko.newslike:1.0.0",
    text_preprocessor=functools.partial(
        t5.data.preprocessors.rekey, key_map={"inputs": None, "targets": "text"}),
    token_preprocessor=t5.data.preprocessors.span_corruption,
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[])
