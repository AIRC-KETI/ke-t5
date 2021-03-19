

"""Add Tasks to registry."""
from ke_t5.proc_utils import _ted_multi_translate_preprocess
from ke_t5.proc_utils import preprocess_korquad
from ke_t5.proc_utils import get_kor_copora_preprocess_fn, get_kor_copora_postprocess_fn, get_kor_corpora_metric
import functools


import t5.data
from t5.evaluation import metrics
import tensorflow_datasets as tfds

from ke_t5.default_vocab import DEFAULT_VOCAB

DEFAULT_TEMPERATURE = 1.0 / 0.3
DEFAULT_MIX_RATE = functools.partial(
    t5.data.rate_num_examples, temperature=DEFAULT_TEMPERATURE)

DEFAULT_OUTPUT_FEATURES = {
    "inputs": t5.data.Feature(
        vocabulary=DEFAULT_VOCAB, add_eos=True, required=False),
    "targets": t5.data.Feature(
        vocabulary=DEFAULT_VOCAB, add_eos=True)
}

# ==============================================================================
# =============================== Single Task ==================================
# ==============================================================================

# ==================================== KE pre-training ======================================

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


# =================================== NSMC =====================================
t5.data.TaskRegistry.add(
    "ke_t5_kor_copora_nsmc",
    t5.data.TfdsTask,
    tfds_name="kor_corpora/nsmc.split:1.0.0",
    text_preprocessor=get_kor_copora_preprocess_fn('nsmc'),
    postprocess_fn=get_kor_copora_postprocess_fn('nsmc'),
    metric_fns=get_kor_corpora_metric('nsmc'),
    output_features=DEFAULT_OUTPUT_FEATURES,
    splits=None)

# ============================== Question Pair ================================
t5.data.TaskRegistry.add(
    "ke_t5_kor_copora_qpair",
    t5.data.TfdsTask,
    tfds_name="kor_corpora/qpair.split:1.0.0",
    text_preprocessor=get_kor_copora_preprocess_fn('qpair'),
    postprocess_fn=get_kor_copora_postprocess_fn('qpair'),
    metric_fns=get_kor_corpora_metric('qpair'),
    output_features=DEFAULT_OUTPUT_FEATURES,
    splits=None)

# =================================== KorNLI =====================================
t5.data.TaskRegistry.add(
    "ke_t5_kor_copora_kornli",
    t5.data.TfdsTask,
    tfds_name="kor_corpora/kornli.split:1.0.0",
    text_preprocessor=get_kor_copora_preprocess_fn('kornli'),
    postprocess_fn=get_kor_copora_postprocess_fn('kornli'),
    metric_fns=get_kor_corpora_metric('kornli'),
    output_features=DEFAULT_OUTPUT_FEATURES,
    splits=None)

# =================================== KorSTS =====================================
t5.data.TaskRegistry.add(
    "ke_t5_kor_copora_korsts",
    t5.data.TfdsTask,
    tfds_name="kor_corpora/korsts.split:1.0.0",
    text_preprocessor=get_kor_copora_preprocess_fn('korsts'),
    postprocess_fn=get_kor_copora_postprocess_fn('korsts'),
    metric_fns=get_kor_corpora_metric('korsts'),
    output_features=DEFAULT_OUTPUT_FEATURES,
    splits=None)

# ============================== Korean Hate Speech ================================
t5.data.TaskRegistry.add(
    "ke_t5_kor_copora_khsd",
    t5.data.TfdsTask,
    tfds_name="kor_corpora/khsd.split:1.0.0",
    text_preprocessor=get_kor_copora_preprocess_fn('khsd'),
    postprocess_fn=get_kor_copora_postprocess_fn('khsd'),
    metric_fns=get_kor_corpora_metric('khsd'),
    output_features=DEFAULT_OUTPUT_FEATURES,
    splits=None)

from ke_t5.proc_utils import nikl_cola_preprocess_fn, nikl_cola_postprocess_fn
# =================================== NIKL CoLA ====================================
t5.data.TaskRegistry.add(
    "ke_t5_nikl_cola",
    t5.data.TfdsTask,
    tfds_name="nikl/cola.v1.0:1.0.0",
    text_preprocessor=nikl_cola_preprocess_fn(),
    postprocess_fn=nikl_cola_postprocess_fn(),
    metric_fns=[metrics.sklearn_metrics_wrapper(
        "matthews_corrcoef", metric_post_process_fn=lambda x: 100 * x)],
    output_features=DEFAULT_OUTPUT_FEATURES,
    splits=None)

# =================================== Korquad ====================================
# Maximized evaluation metrics over all answers.
t5.data.TaskRegistry.add(
    "ke_t5_korquad_allanswers",
    t5.data.TfdsTask,
    tfds_name="korquad/v1.0.split:1.0.0",
    text_preprocessor=preprocess_korquad,
    postprocess_fn=t5.data.postprocessors.qa,
    metric_fns=[metrics.squad],
    output_features=DEFAULT_OUTPUT_FEATURES)

# Maximized evaluation metrics over all answers.
t5.data.TaskRegistry.add(
    "ke_t5_korquad_context_free",
    t5.data.TfdsTask,
    tfds_name="korquad/v1.0.split:1.0.0",
    text_preprocessor=functools.partial(
        preprocess_korquad, include_context=False),
    postprocess_fn=t5.data.postprocessors.qa,
    metric_fns=[metrics.squad],
    output_features=DEFAULT_OUTPUT_FEATURES)

from ke_t5.proc_utils import check_string
from ke_t5.proc_utils import summarize_split
# =================================== NIKL Summarization ====================================
t5.data.TaskRegistry.add(
    "ke_t5_nikl_summarization",
    t5.data.TfdsTask,
    tfds_name="nikl/summarization.v1.0.split:1.0.0",
    text_preprocessor=functools.partial(t5.data.preprocessors.summarize,
                                        article_key="article",
                                        summary_key="highlights"),
    postprocess_fn=check_string,
    metric_fns=[metrics.bleu, metrics.rouge],
    output_features=DEFAULT_OUTPUT_FEATURES)

t5.data.TaskRegistry.add(
    "ke_t5_nikl_summarization_summary",
    t5.data.TfdsTask,
    tfds_name="nikl/summarization.v1.0.summary.split:1.0.0",
    text_preprocessor=functools.partial(summarize_split,
                                        article_key="article",
                                        summary_key="highlights",
                                        summary_type="summary"),
    postprocess_fn=check_string,
    metric_fns=[metrics.bleu, metrics.rouge],
    output_features=DEFAULT_OUTPUT_FEATURES)

t5.data.TaskRegistry.add(
    "ke_t5_nikl_summarization_topic",
    t5.data.TfdsTask,
    tfds_name="nikl/summarization.v1.0.topic.split:1.0.0",
    text_preprocessor=functools.partial(summarize_split,
                                        article_key="article",
                                        summary_key="highlights",
                                        summary_type="topic"),
    postprocess_fn=check_string,
    metric_fns=[metrics.bleu, metrics.rouge],
    output_features=DEFAULT_OUTPUT_FEATURES)

# ============================== TED Open Translation project ===============================
t5.data.TaskRegistry.add(
    "ke_t5_ted_multi_en_ko",
    t5.data.TfdsTask,
    tfds_name="ted_multi_translate:1.1.0",
    text_preprocessor=functools.partial(_ted_multi_translate_preprocess,
                                        source_language="en",
                                        target_language="ko"),
    postprocess_fn=check_string,
    metric_fns=[metrics.bleu, metrics.rouge],
    output_features=DEFAULT_OUTPUT_FEATURES)

t5.data.TaskRegistry.add(
    "ke_t5_ted_multi_ko_en",
    t5.data.TfdsTask,
    tfds_name="ted_multi_translate:1.1.0",
    text_preprocessor=functools.partial(_ted_multi_translate_preprocess,
                                        source_language="ko",
                                        target_language="en"),
    postprocess_fn=check_string,
    metric_fns=[metrics.bleu, metrics.rouge],
    output_features=DEFAULT_OUTPUT_FEATURES)

# =================================== WNLI =====================================
t5.data.TaskRegistry.add(
    "ke_t5_glue_wnli_v002_simple_eval",
    t5.data.TfdsTask,
    tfds_name="glue/wnli:1.0.0",
    text_preprocessor=t5.data.preprocessors.wnli_simple,
    postprocess_fn=t5.data.postprocessors.wsc_simple,
    metric_fns=[metrics.accuracy],
    output_features=DEFAULT_OUTPUT_FEATURES,
    splits=["validation", "test"])

from ke_t5.proc_utils import get_glue_postprocess_fn
# ============================== GLUE ===============================
for b in tfds.text.glue.Glue.builder_configs.values():
    t5.data.TaskRegistry.add(
        "ke_t5_glue_%s_v002" % b.name,
        t5.data.TfdsTask,
        tfds_name="glue/%s:1.0.0" % b.name,
        text_preprocessor=t5.data.glue_utils.get_glue_text_preprocessor(b),
        metric_fns=t5.data.glue_utils.get_glue_metric(b.name),
        output_features=DEFAULT_OUTPUT_FEATURES,
        postprocess_fn=get_glue_postprocess_fn(b),
        splits=["test"] if b.name == "ax" else None,
    )

# ================================= SuperGlue ==================================
for b in tfds.text.super_glue.SuperGlue.builder_configs.values():
  # We use a simplified version of WSC, defined below
  if "wsc" in b.name:
    continue
  if b.name == "axb":
    text_preprocessor = [
        functools.partial(
            t5.data.preprocessors.rekey,
            key_map={
                "premise": "sentence1",
                "hypothesis": "sentence2",
                "label": "label",
                "idx": "idx",
            }),
        t5.data.glue_utils.get_glue_text_preprocessor(b)
    ]
  else:
    text_preprocessor = t5.data.glue_utils.get_glue_text_preprocessor(b)
  t5.data.TaskRegistry.add(
      "ke_t5_super_glue_%s_v102" % b.name,
      t5.data.TfdsTask,
      tfds_name="super_glue/%s:1.0.2" % b.name,
      text_preprocessor=text_preprocessor,
      metric_fns=t5.data.glue_utils.get_super_glue_metric(b.name),
      output_features=DEFAULT_OUTPUT_FEATURES,
      postprocess_fn=get_glue_postprocess_fn(b),
      splits=["test"] if b.name in ["axb", "axg"] else None)

# ======================== Definite Pronoun Resolution =========================
t5.data.TaskRegistry.add(
    "ke_t5_dpr_v001_simple",
    t5.data.TfdsTask,
    tfds_name="definite_pronoun_resolution:1.1.0",
    text_preprocessor=t5.data.preprocessors.definite_pronoun_resolution_simple,
    metric_fns=[metrics.accuracy],
    output_features=DEFAULT_OUTPUT_FEATURES)

# =================================== WSC ======================================
t5.data.TaskRegistry.add(
    "ke_t5_super_glue_wsc_v102_simple_train",
    t5.data.TfdsTask,
    tfds_name="super_glue/wsc.fixed:1.0.2",
    text_preprocessor=functools.partial(
        t5.data.preprocessors.wsc_simple, correct_referent_only=True),
    metric_fns=[],
    output_features=DEFAULT_OUTPUT_FEATURES,
    splits=["train"])
t5.data.TaskRegistry.add(
    "ke_t5_super_glue_wsc_v102_simple_eval",
    t5.data.TfdsTask,
    tfds_name="super_glue/wsc.fixed:1.0.2",
    text_preprocessor=functools.partial(
        t5.data.preprocessors.wsc_simple, correct_referent_only=False),
    postprocess_fn=t5.data.postprocessors.wsc_simple,
    metric_fns=[metrics.accuracy],
    output_features=DEFAULT_OUTPUT_FEATURES,
    splits=["validation", "test"])

# ============================== CNN/DailyMail ===============================
t5.data.TaskRegistry.add(
    "ke_t5_cnn_dailymail_v002",
    t5.data.TfdsTask,
    tfds_name="cnn_dailymail:3.1.0",
    text_preprocessor=functools.partial(
        t5.data.preprocessors.summarize,
        article_key="article",
        summary_key="highlights"),
    postprocess_fn=check_string,
    metric_fns=[metrics.rouge],
    output_features=DEFAULT_OUTPUT_FEATURES)

# =================================== Squad ====================================
# Maximized evaluation metrics over all answers.
t5.data.TaskRegistry.add(
    "ke_t5_squad_v010_allanswers",
    t5.data.TfdsTask,
    tfds_name="squad/v1.1:2.0.0",
    text_preprocessor=t5.data.preprocessors.squad,
    postprocess_fn=t5.data.postprocessors.qa,
    metric_fns=[metrics.squad],
    output_features=DEFAULT_OUTPUT_FEATURES)


# Maximized evaluation metrics over all answers.
t5.data.TaskRegistry.add(
    "ke_t5_squad_v010_context_free",
    t5.data.TfdsTask,
    tfds_name="squad/v1.1:2.0.0",
    text_preprocessor=functools.partial(
        t5.data.preprocessors.squad, include_context=False),
    postprocess_fn=t5.data.postprocessors.qa,
    metric_fns=[metrics.squad],
    output_features=DEFAULT_OUTPUT_FEATURES)

# ================================= TriviaQA ===================================
t5.data.TaskRegistry.add(
    "ke_t5_trivia_qa_v010",
    t5.data.TfdsTask,
    tfds_name="trivia_qa/rc:1.1.0",
    text_preprocessor=t5.data.preprocessors.trivia_qa,
    metric_fns=[],
    token_preprocessor=t5.data.preprocessors.trivia_qa_truncate_inputs,
    output_features=DEFAULT_OUTPUT_FEATURES)

# ===============================================================================
# ================================= Mixtures ====================================
# ===============================================================================
import ke_t5.proc_utils
_GLUE_WEIGHT_MAPPING = t5.data.glue_utils.get_glue_weight_mapping()
_SUPER_GLUE_WEIGHT_MAPPING = t5.data.glue_utils.get_super_glue_weight_mapping()


_glue_tasks_with_weight = list(ke_t5.proc_utils.get_ket5_glue_weight_mapping().items())

_wsc_dpr_tasks = [
    "ke_t5_dpr_v001_simple",
    "ke_t5_super_glue_wsc_v102_simple_train",
    "ke_t5_super_glue_wsc_v102_simple_eval",
]

_super_glue_tasks_with_weight = list(ke_t5.proc_utils.get_ket5_super_glue_weight_mapping().items())

_all_tasks = list(ke_t5.proc_utils.get_ket5_all_weight_mapping().keys())

_ko_text_classification = ke_t5.proc_utils.get_ket5_ko_text_classification()

# ============================== GLUE Mixture =================================
t5.data.MixtureRegistry.add(
    "ke_t5_glue_v002_proportional",
    _glue_tasks_with_weight)

t5.data.MixtureRegistry.add(
    "ke_t5_glue_v002_equal",
    [k for k, v in _glue_tasks_with_weight],
    default_rate=1.0)

# ============================== Super GLUE Mixture =================================
t5.data.MixtureRegistry.add(
    "ke_t5_super_glue_v102_proportional",
    _super_glue_tasks_with_weight)

t5.data.MixtureRegistry.add(
    "ke_t5_super_glue_v102_equal",
    [k for k, v in _super_glue_tasks_with_weight],
    default_rate=1.0)

# ========================== Korean Text Classification Mixture =============================
t5.data.MixtureRegistry.add(
    "ke_t5_ko_text_classification_proportional",
    [(t, ke_t5.proc_utils.dedupe(t)) for t in _ko_text_classification])

t5.data.MixtureRegistry.add(
    "ke_t5_ko_text_classification_equal",
    _ko_text_classification,
    default_rate=1.0)

# ========================== Ko Summary Mixture =============================
t5.data.MixtureRegistry.add(
    "ke_t5_nikl_summary_mixture_equal",
    ["ke_t5_nikl_summarization_summary", "ke_t5_nikl_summarization_topic"],
    default_rate=1.0)

# ========================== Ko En Summary Mixture =============================
t5.data.MixtureRegistry.add(
    "ke_t5_ko_en_summary_proportional",
    [(t, ke_t5.proc_utils.dedupe(t)) for t in ["ke_t5_cnn_dailymail_v002", "ke_t5_nikl_summarization"]])

t5.data.MixtureRegistry.add(
    "ke_t5_ko_en_summary_equal",
    ["ke_t5_cnn_dailymail_v002", "ke_t5_nikl_summarization"],
    default_rate=1.0)

# ========================== Ko En QA Mixture =============================
t5.data.MixtureRegistry.add(
    "ke_t5_ko_en_qa_proportional",
    [(t, ke_t5.proc_utils.dedupe(t)) for t in ["ke_t5_korquad_allanswers", "ke_t5_squad_v010_allanswers", "ke_t5_trivia_qa_v010"]])

t5.data.MixtureRegistry.add(
    "ke_t5_ko_en_qa_equal",
    ["ke_t5_korquad_allanswers", "ke_t5_squad_v010_allanswers", "ke_t5_trivia_qa_v010"],
    default_rate=1.0)

# ========================== Ko En Translation Mixture =============================
t5.data.MixtureRegistry.add(
    "ke_t5_ko_translation_proportional",
    [(t, ke_t5.proc_utils.dedupe(t)) for t in ["ke_t5_ted_multi_en_ko", "ke_t5_ted_multi_ko_en"]])

t5.data.MixtureRegistry.add(
    "ke_t5_ko_translation_equal",
    ["ke_t5_ted_multi_en_ko", "ke_t5_ted_multi_ko_en"],
    default_rate=1.0)

# ============================== All tasks Mixture =================================
t5.data.MixtureRegistry.add(
    "ke_t5_all_proportional",
    [(t, ke_t5.proc_utils.dedupe(t)) for t in _all_tasks])

t5.data.MixtureRegistry.add(
    "ke_t5_all_equal",
    _all_tasks,
    default_rate=1.0)
