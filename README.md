# KE-T5: Korean-English T5

KE-T5는 [Text-to-Text Transfer Transformer](https://arxiv.org/abs/1910.10683) 모델을 한국어와 영어 코퍼스를 이용하여 사전학습한 모델입니다.

Vocabulary는 64,000개의 sub-word token으로 이루어져 있으며, Google의 [sentencepiece](https://github.com/google/sentencepiece)를 이용하여 만들었습니다. Sentencepiece 모델은 한국어와 영어가 약 7:3의 비율로 섞인 30GB의 코퍼스를 99.95% 커버하도록 학습되었습니다.

사전학습에 사용된 비정형 코퍼스에 대한 구성과 정보는 [Datasets](#datasets)을 참고하시기 바랍니다.

<br>

## Long-KE-T5 Downstreams

기존 KE-T5보다 많은 양의 토큰을 입력으로 받고 출력으로 생성 가능한 [Long KE-T5](https://github.com/AIRC-KETI/long-ke-t5)가 공개되었습니다. Long KE-T5는 입력 기준 4096 토큰, 출력 기준 1024 토큰으로 학습되었습니다.
또한 Long KE-T5를 활용한 간단한 생성 모델(Downstream Tasks) 몇 개가 공개 되었으며 지속적으로 추가 모델들이 공개될 예정입니다.

## KE-T5 Downstreams

Pytorch와 Huggingface transformers를 사용하시는 분들이 Downstream task들을 비교적 쉽게 학습시킬 수 있는 [KE-T5 Downstreams](https://github.com/AIRC-KETI/ke-t5-downstreams)가 공개되었습니다. NER, Summarization, Sequence Classification등 여러 task를 학습시킬 수 있습니다.

## Fine-tuning

KE-T5를 이용하여 Downstream task를 학습하기 위해서는 먼저 pre-trained model을 다운 받아야 합니다. [pre-trained models](#pretrained-models)에서 다운 받으실 수 있습니다

이 섹션에서는 [Google Cloud Platform](https://cloud.google.com/)에서 TPU를 이용하여 학습하는 방법만 설명합니다. GPU에서 사용하실 경우 [T5 github](https://github.com/google-research/text-to-text-transfer-transformer#gpu-usage)에서 설명하듯이 [gin-config](https://github.com/google/gin-config)의 `utils.run.mesh_devices`와 `utils.run.mesh_shape` parameter를 오버라이드 해주시면 됩니다.\
[TFRC](https://www.tensorflow.org/tfrc)를 이용하시면 TPU를 한달간 무료로 사용하실 수 있습니다.

먼저 되도록 TPU Zone과 동일한 곳에 버킷과 VM instance를 생성합니다. VM instance와 TPU의 area는 버킷보다 세분화 되어 있습니다.
생성한 VM instance에 접속한 후 아래와 같이 repository를 clone하고 필요한 python package들을 설치합니다.

### Install packages
```bash
    git clone https://github.com/AIRC-KETI/ke-t5.git
    cd ke-t5
    pip3 install -r requirements.txt
```

datasets 폴더에 있는 script로 데이터셋을 생성하고 Bucket에 복사합니다. 아래의 스크립트에서는 생성된 데이터셋을 버킷의 root에 복사한다고 가정했습니다. 데이터셋 생성방법은 [여기](datasets/README.md)를 확인하시기 바랍니다.\
또한 다운받은 pre-trained 모델도 버킷에 저장하고 해당 디렉토리를 `PRETRAINED_MODEL_DIR` 변수에 assign합니다.\
학습하고자 하는 task를 `TASK` 변수에 assign합니다.\
아래 스크립트와 같이 TPU를 생성하고 모델을 학습합니다.

### Run downstream tasks on your TPU
```bash
export PROJECT_ID=your_project_id
export TPU_NAME=your_tpu_name
export ZONE=your_project_zone
export TPU_SIZE=v3-8

# create TPU
ctpu up --name=$TPU_NAME --project=$PROJECT_ID \
--zone=$ZONE --tpu-size=$TPU_SIZE --tf-version=2.4.1 --tpu-only --noconf

export PROJECT=your_project_name
export BUCKET=gs://yourbucket
export PRETRAINED_MODEL_DIR="${BUCKET}/your_pretrained_model_dir"
export DATA_DIR="${BUCKET}/tensorflow_datasets"
export MODEL_DIR="${BUCKET}/your_model_dir"

export TOKENS_PER_BATCH=65536
export PRETRAINED_STEPS=1000000
export TRAIN_STEPS=10000

export TASK=ke_t5_ko_translation_proportional
export TASK_DIR="${MODEL_DIR}/${TASK}"

export OP_CFG="${PRETRAINED_MODEL_DIR}/operative_config.gin"
export SHELL_PATH=`pwd -P`

python3 -m t5.models.mesh_transformer_main \
        --tpu="${TPU_NAME}" \
        --gcp_project="${PROJECT}" \
        --tpu_zone="${ZONE}" \
        --model_dir="${TASK_DIR}" \
        --gin_file="dataset.gin" \
        --gin_file="${OP_CFG}" \
        --gin_file="${SHELL_PATH}/ke_t5/gin/sequence_lengths/${TASK}.gin" \
        --gin_param="MIXTURE_NAME = '${TASK}'" \
        --gin_param="utils.run.batch_size = ('tokens_per_batch', ${TOKENS_PER_BATCH})" \
        --gin_param="utils.tpu_mesh_shape.model_parallelism = 1" \
        --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'" \
        --gin_param="utils.run.train_steps=$((PRETRAINED_STEPS+TRAIN_STEPS))" \
        --gin_param="utils.run.init_checkpoint='${PRETRAINED_MODEL_DIR}/model.ckpt-${PRETRAINED_STEPS}'" \
        --gin_file="learning_rate_schedules/constant_0_001.gin" \
        --t5_tfds_data_dir="${DATA_DIR}" \
        --module_import="ke_t5.tasks"
```

<br>

## How to use a saved model (exported tensorflow model)

Pre-trained 모델의 체크포인트뿐만 아니라 일부 Downstream tasks의 saved model도 공개했습니다.\
Tensorflow saved 모델은 `tensorflow`와 `tensorflow_text`가 설치되어 있으면 바로 사용하실 수 있습니다.
아래 예는 python에서 saved model을 사용하는 방법이며, `ke_t5_nikl_summary_mixture_equal` task를 다운받았다고 가정하고 있습니다.

`ke_t5_nikl_summary_mixture_equal` task는 `ke_t5_nikl_summarization_summary`와 `ke_t5_nikl_summarization_topic`의 mixture tasks이며, 각각 `summarize_topic`과 `summarize_summary`로 입력 텍스트가 시작합니다.
`ke_t5_nikl_summarization_summary`은 사람이 요약한 데이터를 이용하여 학습하며, `ke_t5_nikl_summarization_topic`은 topic sentences를 생성하도록 학습됩니다.


```python
# pip install tensorflow tensorflow_text
import numpy as np
import tensorflow as tf
import tensorflow_text

model_path = "path to exported model dir"
loaded = tf.saved_model.load(model_path)
infer = loaded.signatures["serving_default"]

# source: https://news.naver.com/main/read.nhn?mode=LSD&mid=shm&sid1=102&oid=081&aid=0003173411
# press: 서울신문(www.seoul.co.kr)
# author: 이주원 기자 starjuwon@seoul.co.kr
input_str = """“처음에는 ‘금방 끝나겠지’라고 생각했는데 어느덧 100일이 됐네요. \
그동안 춥고 아프고 힘들었지만 인간으로서 대우를 받을 수만 있다면 끝까지 버틸 수 있습니다.” \
LG트윈타워 청소 노동자들이 고용승계를 주장하며 파업에 나선지 100일째를 하루 앞둔 24일 \
서울 여의도 LG트윈타워 앞 ‘행복한 고용승계 텐트촌’에서 만난 박상설(63)씨는 힘들었던 투쟁 과정을 \
회상하며 눈시울을 붉혔다. 박씨는 2017년부터 LG트윈타워에서 청소 노동을 했지만 지난 1월 1일부로 \
계약이 종료돼 직장을 떠났다. 자동차 소음과 불편한 잠자리로 텐트에서 매일 밤잠을 설치지만 투쟁을 \
포기할 수 없다고 한다. 그는 “LG가 그동안 사회적 책임과 정도경영을 강조해 왔기에 파업이 금방 끝날 \
줄 알았다”며 “버티지 못하고 점점 떠나는 동지들을 바라볼 때마다 마음이 아프지만 정당한 노동 권리를 \
인정받기 위해 끝까지 투쟁할 것”이라고 강조했다. 지난해 11월 26일부터 파업에 돌입한 청소 \
노동자들은 25일 파업 100일째를 맞는다. 건물 1층 로비에서 시위를 하던 25명의 청소 노동자들은 지난 \
22일부터 정문 앞 도보에 텐트촌을 설치하고 장소를 옮겼다. 파업 100일에 맞춰 25일까지 시민연대와 \
함께 텐트 100개를 설치하고 주·야간 연대 시위를 이어가겠다는 뜻에서다. 노동자들은 한 명이 간신히 \
누울 수 있는 크기의 텐트 안에서 딱딱한 시멘트 바닥에 몸을 기대 쪽잠을 청하고 있다. LG트윈타워를 \
관리하는 LG그룹 계열사 ‘에스엔아이코퍼레이션’은 지난해 말 ‘지수아이앤씨’와 청소 용역 계약을 \
끝내고 다른 업체와 새로 계약했다. 사측은 ‘품질 저하’를 이유로 들었다. 반면 노동자들은 2019년 \
노조를 결성하고 권리를 주장하기 시작하면서 사측 눈 밖에 났다고 주장한다. 그동안 업체가 \
변경되더라도 기존 업체 노동자들이 새 업체에 고용승계가 되는 게 관례였지만 새 업체는 고용승계를 \
보장할 수 없다고 밝혔다. 지난달까지 고용노동부 중재로 수차례 노사 교섭이 있었지만 상황은 달라지지 \
않았다. 사측은 대신 노동자들에게 다른 사업장에서 일을 하게 해주겠다고 권유했다. 하지만 노동자들은 \
노조를 인정하지 않는 대기업의 행태를 묵인한 채 사측의 권유에 따른다면 어느 사업장에서 일을 하던 \
똑같은 행태가 반복될 수밖에 없다고 목소리를 높인다. 때문에 반드시 LG트윈타워에서 정당한 권리를 \
인정받고 노동을 이어가야만 한다고 말한다. 이들은 구광모 LG그룹 회장이 나서 문제를 해결해야 한다고 \
주장한다. 이혜정 LG트윈타워 공동대책위원회 집행위원은 “구 회장이 책임있는 답변을 내놓을 때까지 \
시민사회 단위와 함께 결의를 담아 끝까지 텐트촌을 유지할 것”이라고 강조했다."""

input_str_topic = "summarize_topic: " + input_str
input_str_summary = "summarize_summary: " + input_str

x = tf.constant([input_str_topic])

result = infer(x)
print([out.decode('utf-8') for out in result['inputs'].numpy()])
print([out.decode('utf-8') for out in result['outputs'].numpy()])

# summarize_topic
# 'LG트윈타워 청소 노동자가 고용승계를 주장하며 파업에 나선지 100일째를 하루 앞둔 24일 서울 \
# 여의도 LG트윈타워 앞 ‘행복한 고용승계 텐트촌’에서 만난 박상설(63)씨는 힘들었던 투쟁 과정을 \
# 회상하며 눈시울을 붉혔다. 반면 노동자들은 2019년 노조를 결성하고 권리를 주장하기 시작하면서 사측 \
# 눈 밖에 났다고 주장한다. 때문에 반드시 LG트윈타워에서 정당한 권리를 인정받고 노동을 이어가야 \
# 한다고 말한다.

# summarize_summary
# 'LG트윈타워 청소 노동자가 고용승계를 주장하며 파업에 나선지 100일째를 맞았다. LG트윈타워를 \
# 관리하는 LG그룹 계열사 ‘에스엔아이코퍼레이션’은 지난해 말 ‘지수아이앤씨’와 청소 용역 계약을 \
# 끝내고 다른 업체와 새로 계약했다. 그러나 노동자들은 노조를 인정하지 않는 대기업의 행태를 묵인한 \
# 채 사측의 권유에 따라 노동을 이어가라고 주장한다.'

```

<br>

## Datasets

데이터셋은 한국어의 경우 센터에서 확보하고 있는 비정형 코퍼스 중 Manual Cleaning을 여러번 진행한 데이터셋과 NIKL 데이터셋(**국립 국어원 모두의 말뭉치**)의 비정형 코퍼스 일부를 사용하였습니다.
영어의 경우 [RealNews](https://github.com/rowanz/grover/tree/master/realnews) 데이터셋을 사용하였습니다.
데이터셋 구성은 아래와 같습니다.

### `ke`
**Dataset size**: `92.02GiB`\
**Corpus type**: `ko` (Newspaper, Written, Web text, Messenger, Spoken) `en` (Newspaper)\
**Split**\
'train': total **39,344,633** examples (ko: 25,544,795, en: 13,799,838)\
'validation': total **39,283** examples (ko: 25,420, en: 13,863)

### `ke.newslike`
**Dataset size**: `81.74 GiB`\
**Corpus type**: `ko` (Newspaper) `en` (Newspaper)\
**Split**\
'train': total **36,534,568** examples (ko: 22,735,237, en: 13,799,838)\
'validation': total **36,452** examples (ko: 22,619, en: 13,863)

### `ko`
**Dataset size**: `57.77 GiB`\
**Corpus type**: `ko` (Newspaper, Written, Web text, Messenger, Spoken)\
**Split**\
'train': total **25,545,302** examples\
'validation': total **25,450** examples

### `ko.newslike`
**Dataset size**: `47.49 GiB`\
**Corpus type**: `ko` (Newspaper)\
**Split**\
'train': total **22,735,237** examples\
'validation': total **22,619** examples

<br>

## Models

### Pretrained models

HuggingFace 모델 허브에 모델이 추가되어 사용이 가능합니다. \
Huggingface TFT5ForConditionalGeneration, T5ForConditionalGeneration를 사용하실 경우 아래 코드와 같이 사용하시면 됩니다.

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = 'KETI-AIR/ke-t5-base'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
```


<table>
    <thead>
        <tr>
            <th>Dataset</th>
            <th>Size</th>
            <th>steps</th>
            <th>Download URL(Tensorflow)</th>
            <th>Huggingface Transformers</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=3>ke</td>
            <td rowspan=1>small</td>
            <td> 5.5M </td>
            <td> <a href='https://drive.google.com/file/d/1VtyD6i7lJRnrQVYN_y_pZMg7pFUsDrmn/view?usp=sharing'> Download </a>  </td>
            <td> KETI-AIR/ke-t5-small </td>
        </tr>
        <tr>
            <td rowspan=1>base</td>
            <td> 2.2M </td>
            <td> <a href='https://drive.google.com/file/d/1niTeiy7pzrOB20u8oq0az-A9gUpKP_Vo/view?usp=sharing'> Download </a>  </td>
            <td> KETI-AIR/ke-t5-base </td>
        </tr>
        <tr>
            <td rowspan=1>large</td>
            <td> 1.7M </td>
            <td> <a href='https://drive.google.com/file/d/1Iu4XgLNc99wPkl8KgYD3JIOLBWXDzveo/view?usp=sharing'> Download </a>  </td>
            <td> KETI-AIR/ke-t5-large </td>
        </tr>
        <tr>
            <td rowspan=3>ke.newslike</td>
            <td rowspan=1>small</td>
            <td> 5.4M </td>
            <td> <a href='https://drive.google.com/file/d/1_q_vInAVxgKbUbVHuc_DXIEefJ8eoVTY/view?usp=sharing'> Download </a>  </td>
            <td> KETI-AIR/ke-t5-small-newslike </td>
        </tr>
        <tr>
            <td rowspan=1>base</td>
            <td> 2.2M </td>
            <td> <a href='https://drive.google.com/file/d/1dVibwqYmoMQElRUR53vIaKWSJ3rpqUDI/view?usp=sharing'> Download </a>  </td>
            <td> KETI-AIR/ke-t5-base-newslike </td>
        </tr>
        <tr>
            <td rowspan=1>large</td>
            <td> 1.7M </td>
            <td> <a href='https://drive.google.com/file/d/1AstaSL2NKYZJ1fGFmgSyRxnAxS3aAbyd/view?usp=sharing'> Download </a>  </td>
            <td> KETI-AIR/ke-t5-large-newslike </td>
        </tr>
        <tr>
            <td rowspan=3>ko</td>
            <td rowspan=1>small</td>
            <td> 5.3M </td>
            <td> <a href='https://drive.google.com/file/d/1cFR4NHXa5mJEwtPN8OPslbVL18BLZ30F/view?usp=sharing'> Download </a>  </td>
            <td> KETI-AIR/ke-t5-small-ko </td>
        </tr>
        <tr>
            <td rowspan=1>base</td>
            <td> 2.3M </td>
            <td> <a href='https://drive.google.com/file/d/1nSVTCfDeoQN2M1fPXxPxRZ-_boYjSTqR/view?usp=sharing'> Download </a>  </td>
            <td> KETI-AIR/ke-t5-base-ko </td>
        </tr>
        <tr>
            <td rowspan=1>large</td>
            <td> 1M </td>
            <td> <a href='https://drive.google.com/file/d/12cKWNExMtOtmUBqO745YsoAou1eZr2Yz/view?usp=sharing'> Download </a>  </td>
            <td> KETI-AIR/ke-t5-large-ko </td>
        </tr>
    </tbody>
</table>

<br>

### Downstream models (Tensorflow saved model)

<table>
    <thead>
        <tr>
            <th>Dataset</th>
            <th>Size</th>
            <th>steps</th>
            <th>ke_t5_nikl_summary_mixture_equal</th>
            <th>ke_t5_ko_en_qa_equal</th>
            <th>ke_t5_ko_translation_proportional</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=3>ke</td>
            <td>small</td>
            <td> 1M </td>
            <td>  <a href='https://drive.google.com/file/d/1qZM7jcpeaao6NOo88aLgOYRi5lfxIZcZ/view?usp=sharing'> Download </a>  </td> <!-- ke_t5_nikl_summary_mixture_equal -->
            <td>  <a href='https://drive.google.com/file/d/1mnxToYnIfDFPxd4fmEuBMLRTBLR2YrrT/view?usp=sharing'> Download </a>  </td> <!-- ke_t5_ko_en_qa_equal -->
            <td>  <a href='https://drive.google.com/file/d/1IN2KSPh_Ul7iNZt57dF7odcbQ8VREsGl/view?usp=sharing'> Download </a>  </td> <!-- ke_t5_ko_translation_proportional -->
        </tr>
        <tr>
            <td>base</td>
            <td> 600K </td>
            <td>  <a href='https://drive.google.com/file/d/1lKi2-UTj2GVWih44mrGv7PWj1O-cNxDw/view?usp=sharing'> Download </a>  </td> <!-- ke_t5_nikl_summary_mixture_equal -->
            <td>  <a href='https://drive.google.com/file/d/1j8QmAkShctCSEPOodEJ3R88uzVV0L5vw/view?usp=sharing'> Download </a>  </td> <!-- ke_t5_ko_en_qa_equal -->
            <td>  <a href='https://drive.google.com/file/d/1NVwzcRSv3OeA4LLRSuZ9rJifmkRb08KJ/view?usp=sharing'> Download </a>  </td> <!-- ke_t5_ko_translation_proportional -->
        </tr>
        <tr>
            <td>large</td>
            <td> 600K </td>
            <td>  <a href='https://drive.google.com/file/d/14LvoUfxDaVaY9sp-SVdK7gPSXi_4HJ_9/view?usp=sharing'> Download </a>  </td> <!-- ke_t5_nikl_summary_mixture_equal -->
            <td>  <a href='https://drive.google.com/file/d/1yVU1B5DA7Vzl_XRXvbKXiNSf0LWHjlbQ/view?usp=sharing'> Download </a>  </td> <!-- ke_t5_ko_en_qa_equal -->
            <td>  <a href='https://drive.google.com/file/d/1UoNhljvHny2bsA8xYy317EmTllN4y2J5/view?usp=sharing'> Download </a>  </td> <!-- ke_t5_ko_translation_proportional -->
        </tr>
        <tr>
            <td rowspan=3>ke.newslike</td>
            <td>small</td>
            <td> 1M </td>
            <td>  <a href='https://drive.google.com/file/d/1gzjcbtNhrkDuMx75dL3XED0Iy0otipUD/view?usp=sharing'> Download </a>  </td> <!-- ke_t5_nikl_summary_mixture_equal -->
            <td>  <a href='https://drive.google.com/file/d/1PoQ8LgXJDQ4Bxud_ed7SeUZsLNtdzgy6/view?usp=sharing'> Download </a>  </td> <!-- ke_t5_ko_en_qa_equal -->
            <td>  <a href='https://drive.google.com/file/d/1Qovz2iv4hnM7mNC16X3TxRuHj4ILs_rf/view?usp=sharing'> Download </a>  </td> <!-- ke_t5_ko_translation_proportional -->
        </tr>
        <tr>
            <td>base</td>
            <td> 600K </td>
            <td>  <a href='https://drive.google.com/file/d/1VK2baHMcUki4Vlk1_Uwg0S1cauNTbf-o/view?usp=sharing'> Download </a>  </td> <!-- ke_t5_nikl_summary_mixture_equal -->
            <td>  <a href='https://drive.google.com/file/d/1EQ4giXlMsOT6WW3wx8TzctSrJDz3gtTi/view?usp=sharing'> Download </a>  </td> <!-- ke_t5_ko_en_qa_equal -->
            <td>  <a href='https://drive.google.com/file/d/1QHNQdv7XxaBhbeXdq2SMUq4Q6jkdVfXg/view?usp=sharing'> Download </a>  </td> <!-- ke_t5_ko_translation_proportional -->
        </tr>
        <tr>
            <td>large</td>
            <td> 600K </td>
            <td>  <a href='https://drive.google.com/file/d/1V_zedd21fEx1omIObfBcIjbqHoSY6zK3/view?usp=sharing'> Download </a>  </td> <!-- ke_t5_nikl_summary_mixture_equal -->
            <td>  <a href='https://drive.google.com/file/d/1_HovEuEr-fyp7J_2O5SaIS0Oo0FyhEFq/view?usp=sharing'> Download </a>  </td> <!-- ke_t5_ko_en_qa_equal -->
            <td>  <a href='https://drive.google.com/file/d/1QLO4yjQVzo2VNFYiKPWZ2uifYa9MXEh-/view?usp=sharing'> Download </a>  </td> <!-- ke_t5_ko_translation_proportional -->
        </tr>
        <tr>
            <td rowspan=3>ko</td>
            <td>small</td>
            <td> 1M </td>
            <td>  <a href='https://drive.google.com/file/d/16PptjLp3QKM2i39kPZnv1hyTwoL1FjVh/view?usp=sharing'> Download </a>  </td> <!-- ke_t5_nikl_summary_mixture_equal -->
            <td>  <a href='https://drive.google.com/file/d/1hYwyOr1vKRqlBx5uDMvGxJMmHrHd4uIK/view?usp=sharing'> Download </a>  </td> <!-- ke_t5_ko_en_qa_equal -->
            <td>  <a href='https://drive.google.com/file/d/1gA70apMtEJX2svtSrhKcPwPDEWKu68i6/view?usp=sharing'> Download </a>  </td> <!-- ke_t5_ko_translation_proportional -->
        </tr>
        <tr>
            <td>base</td>
            <td> 600K </td>
            <td>  <a href='https://drive.google.com/file/d/1w2pCZnBJn_h5sXAUrCxhaNFeZzNNPmXT/view?usp=sharing'> Download </a>  </td> <!-- ke_t5_nikl_summary_mixture_equal -->
            <td>  <a href='https://drive.google.com/file/d/1SGUvuFZSURVCcb8owDTrJAlHl3HXZIyC/view?usp=sharing'> Download </a>  </td> <!-- ke_t5_ko_en_qa_equal -->
            <td>  <a href='https://drive.google.com/file/d/1CDqrXBsCd-frvOOuKkZxrcbT1CGu4DRy/view?usp=sharing'> Download </a>  </td> <!-- ke_t5_ko_translation_proportional -->
        </tr>
    </tbody>
</table>

<br>

## Downstream Tasks

### Tasks

KE-T5에서 지원하는 모든 task들의 목록입니다.

<br>

#### Korean
| Task Name  | Input Format | Output Format | Task Type | Language |
| ------------- | ------------- | ------------- | ------------- |------------- |
| ke_t5_kor_copora_nsmc  | `nsmc sentence:{sentence}`  | `negative` or `positive`  | Classification  | ko |
| ke_t5_kor_copora_qpair  | `qpair question1: {question1} question2: {question2}`  | `duplicated` or `not_duplicate`  | Classification  | ko |
| ke_t5_kor_copora_kornli  | `kornli premise: {premise} hypothesis: {hypothesis}`  | `entailment` or `neutral` or `contradiction`  | Classification  | ko |
| ke_t5_kor_copora_korsts  | `korsts sentence1: {sentence1} sentence2: {sentence2}`  | `{score}` e.g. `1.4`  | Predict similarity score  | ko |
| ke_t5_kor_copora_khsd  | `khsd sentence: {sentence}`  | `hate` or `offensive` or `none` | Classification  | ko |
| ke_t5_nikl_cola  | `niklcola sentence: {sentence}`  | `unacceptable` or `acceptable`  | Classification  | ko |
| ke_t5_korquad_allanswers  | `question: {question} context: {article}`  | `{answer}` | Extractive QA  | ko |
| ke_t5_nikl_summarization  | `summarize: {text}`  | `{summary}` | Summarization  | ko |
| ke_t5_nikl_summarization_summary  | `summarize_summary: {text}`  | `{summary}`  | Summarization  | ko |
| ke_t5_nikl_summarization_topic  | `summarize_topic: {text}`  | `{summary}`  | Select topic sentences  | ko |
| klue_tc_gen  | `klue_tc title: {text}`  | `{topic}`  | Select topic  | ko |
| klue_nli_gen  | `klue_nli premise: {premise} hypothesis: {hypothesis}`  | `entailment` or `neutral` or `contradiction`  | Classification  | ko |
| klue_sts_gen  | `klue_sts sentence1: {sentence1} sentence2: {sentence2}`  | `{score}` e.g. `1.4`  | Predict similarity score  | ko |
| klue_re_gen  | `klue_re text: ~~~*{subject}*~~~#{object}#~~`  | `{relation}`  | Select relation  | ko |
| klue_mrc_gen  | `klue_mrc question: {question} context: {article}`  | `{answer}`  | Extractive QA  | ko |

<br>

#### English


<details>
<summary>ke_t5_glue_cola_v002</summary>

inputs
```
cola sentence: That picture of Susan offended her.
```

targets
```
"unacceptable", "acceptable"
```
</details>



<details>
<summary>ke_t5_glue_sst2_v002</summary>

inputs
```
sst2 sentence: a valueless kiddie paean to pro basketball underwritten by the nba.
```

targets
```
"negative", "positive"
```
</details>



<details>
<summary>ke_t5_glue_mrpc_v002</summary>

inputs
```
mrpc sentence1: The show's closure affected third-quarter earnings per share by a penny. sentence2: The company said this impacted earnings by a penny a share.
```

targets
```
"not_equivalent", "equivalent"
```
</details>



<details>
<summary>ke_t5_glue_qqp_v002</summary>

inputs
```
qqp question1: Who is going to be a better president - Hillary Clinton or Donald Trump? question2: In what aspects is Hillary Clinton better than Trump?
```

targets
```
"not_duplicate", "duplicate"
```
</details>



<details>
<summary>ke_t5_glue_stsb_v002</summary>

inputs
```
stsb sentence1: For what it's worth, dirt on the lens might not be as detrimental as you would think. sentence2: Regular dish washing fluid is great for removing grease, as it not only dissolves it but also binds it.
```

targets
```
0.4
```
</details>



<details>
<summary>ke_t5_glue_mnli_v002</summary>

inputs
```
mnli hypothesis: The other villages and settlements are not as developed. premise: Other villages are much less developed, and therein lies the essence of many delights.
```

targets
```
"entailment", "neutral", "contradiction"
```
</details>



<details>
<summary>ke_t5_glue_mnli_mismatched_v002</summary>

inputs
```
mnli hypothesis: SEND has renovated dozens of homes to sell at affordable prices. premise: In the past eight years SEND has renovated over 70 homes for affordable resale to families, created more than 100 high quality apartments, assisted 20 businesses to expand or improve their facilities, created three new parks, and taught more than 200 youth work ethics and skills.
```

targets
```
"entailment", "neutral", "contradiction"
```
</details>



<details>
<summary>ke_t5_glue_mnli_matched_v002</summary>

inputs
```
mnli hypothesis: The other villages and settlements are not as developed. premise: Other villages are much less developed, and therein lies the essence of many delights.
```

targets
```
"entailment", "neutral", "contradiction"
```
</details>



<details>
<summary>ke_t5_glue_qnli_v002</summary>

inputs
```
qnli question: When were the finalists announced? sentence: The South Florida/Miami area has previously hosted the event 10 times (tied for most with New Orleans), with the most recent one being Super Bowl XLIV in 2010.
```

targets
```
"entailment", "not_entailment"
```
</details>



<details>
<summary>ke_t5_glue_rte_v002</summary>

inputs
```
rte sentence1: Tropical Storm Irene on August 11, 2005 at 16:15 UTC. Tropical Storm Irene will increase in strength over the next several days, possibly developing into a hurricane that will hit the east coast of the United States, said the National Hurricane Center of Miami, Florida in a report today. Irene was located approximately 975 kilometers south-southeast of Bermuda at 16:00 UTC today. Forecasters say that the storm is now moving in a west- northwest direction with top sustained winds of 40 miles per hour. sentence2: A storm called Irene is going to approach the east coast of the US.
```

targets
```
"entailment", "not_entailment"
```
</details>


<details>
<summary>ke_t5_glue_ax_v002</summary>

inputs
```
mnli hypothesis: {hypothesis} premise: {premise}
```

targets
```
"entailment", "neutral", "contradiction"
```
</details>


<details>
<summary>ke_t5_glue_wnli_v002_simple_eval</summary>

pre-processor description (t5)
```
A typical example from WNLI might look like:
  {
    'sentence1': 'The fish ate the worm. It was tasty.',
    'sentence2': 'The worm was tasty.',
    'label': 1,
  }

This will be transformed to:
{
'inputs': 'wsc: The fish ate the worm. *It* was tasty.',
'targets': 'The worm',
'premise': 'The fish ate the worm. It was tasty.,
'hypothesis': 'The worm was tasty.',
'label': 1,
}
```

inputs
```
wsc: The fish ate the worm. *It* was tasty.
```

targets
```
The worm
```
</details>



<details>
<summary>ke_t5_super_glue_boolq_v102</summary>

inputs
```
boolq passage: Demon Drop -- Demon Drop is a drop tower amusement ride at Dorney Park & Wildwater Kingdom. Designed by Intamin, it is a Freefall model that was originally located at Cedar Point when it first opened to the public in 1983. It was relocated to Dorney Park following the 2009 season and reopened in 2010. It is one of the oldest of its kind still in operation. question: does cedar point still have the demon drop
```

targets
```
Yes or No
```
</details>



<details>
<summary>ke_t5_super_glue_cb_v102</summary>

inputs
```
cb hypothesis: in the circumstances she became very uneasy premise: Obeying his instruction, I proffered my hand, open palm upwards, towards the animal. The ratbird climbed on and began to preen its fur unconcernedly. Nobody will blame me if I say that in the circumstances I became very uneasy.
```

targets
```
"entailment", "contradiction", "neutral"
```
</details>



<details>
<summary>ke_t5_super_glue_copa_v102</summary>

inputs
```
copa choice1: The branch moved downstream. choice2: The river's current became stronger. premise: The tree branch landed in the river. question: effect
```

targets
```
"choice1", "choice2"
```
</details>



<details>
<summary>ke_t5_super_glue_multirc_v102</summary>

inputs
```
multirc question: Who began charging a $50 an hour minimum instead of $25 for legal services? answer: Lawyer's market paragraph: For most lawyers, full waiting rooms and appointments booked out to mid-July would equate to a lucrative law practice. But Frank Smith drives a 6-year-old car with 140,000 miles on it, and paying his senior paralegal minimum wage the last few months has put him in the red. Hoped-for federal grants haven"t come through, so he"s had to raise his rates. As of last week he charges $50 an hour minimum instead of $25 for the services of his yearling Northern Utah Legal Aid Foundation. That"s in a lawyer"s market where fees range in the $150 to $250 an hour range in the Ogden area, and up to $400 an hour in the Salt Lake area. Smith"s one-lawyer foundation basically helps the folks who have too much money to qualify for the federally funded Utah Legal Services, but not enough money to afford a lawyer.
```

targets
```
{'value': 0, 'group': 283}
```
</details>



<details>
<summary>ke_t5_super_glue_record_v102</summary>

inputs
```
record query: It has been widely speculated Mr Morrison could serve as Treasurer in Mr @placeholder's cabinet. entities: Abbott, Bill Shorten, Joe Hockey, Julie Bishop, Liberal Party, Malcolm Turnbull, Prime Ministership, Scott Morrison, Social Services, Tony Abbott, Turnbull passage: Malcolm Turnbull's triumph over Tony Abbott in a Liberal Party leadership ballot will have ramifications far beyond the Prime Minister's office. A number of government ministers and backbenchers will have their fortunes changed by the vote, which Prime Minister-elect Turnbull won 54 to 44 on Monday night. Members within the government and also the opposition are expected to be impacted by the change of Prime Minister, including Treasurer Joe Hockey, deputy leader Julie Bishop, Social Services Minister Scott Morrison and Opposition Leader Bill Shorten. Mr Turnbull, 60, has coveted the Prime Ministership for many years. His victory comes seven years after he was first appointed Liberal Party leader, only to be booted from the position by Tony Abbott in December, 2009. Malcolm Turnbull's win over Tony Abbott has created winners and losers. The Prime Minister-elect and deputy leader Julie Bishop are two winners. Opposition Leader Bill Shorten and Joe Hockey could be hurt by the vote. Mr Turnbull defeated Mr Abbott is a leadership ballot 54 votes to 44
```

targets
```
['Malcolm Turnbull', 'Turnbull']
```
</details>



<details>
<summary>ke_t5_super_glue_rte_v102</summary>

inputs
```
rte hypothesis: A storm called Irene is going to approach the east coast of the US. premise: Tropical Storm Irene on August 11, 2005 at 16:15 UTC. Tropical Storm Irene will increase in strength over the next several days, possibly developing into a hurricane that will hit the east coast of the United States, said the National Hurricane Center of Miami, Florida in a report today. Irene was located approximately 975 kilometers south-southeast of Bermuda at 16:00 UTC today. Forecasters say that the storm is now moving in a west- northwest direction with top sustained winds of 40 miles per hour.
```

targets
```
"entailment", "not_entailment"
```
</details>



<details>
<summary>ke_t5_super_glue_wic_v102</summary>

inputs
```
wic sentence1: Had unusual longevity in the company. sentence2: Her longevity as a star. word: longevity
```

targets (Sense match)
```
True or False
```
</details>



<details>
<summary>ke_t5_super_glue_wsc_v102</summary>

inputs
```
wsc: Alice was dusting the living room and trying to find the button that Mama had hidden. No time today to look at old pictures in her favorite photo album. Today she had to hunt for a button, so she put the album on a chair without even opening *it*.
```

targets (Coreference)
```
True of False
```
</details>



<details>
<summary>ke_t5_super_glue_wsc.fixed_v102</summary>

inputs
```
wsc: The stable was very roomy, with four good stalls; a large swinging window opened into the yard, which made *it* pleasant and airy.
```

targets (Coreference)
```
True or False
```
</details>



<details>
<summary>ke_t5_super_glue_axb_v102</summary>

inputs
```
rte hypothesis: {sentence1} premise: {sentence2}
```

targets
```
"entailment", "not_entailment"
```
</details>



<details>
<summary>ke_t5_super_glue_axg_v102</summary>

inputs
```
rte hypothesis: {sentence1} premise: {sentence2}
```

targets
```
"entailment", "not_entailment"
```
</details>



<details>
<summary>ke_t5_dpr_v001_simple</summary>

pre-processor description (t5)
```
A typical example from the definite pronoun resolution dataset might look like
  {
     'sentence': 'Bob asked Tom if he can lend some money.',
     'pronoun': 'he',
     'candidates': ['Bob', 'Tom'],
     'label': 1,
  }

This will be transformed to
{
    'inputs': 'wsc: Bob asked Tom if *he* can lend some money.'
    'targets': 'Tom',
}
```

inputs
```
wsc: Bob asked Tom if *he* can lend some money.
```

targets
```
Tom
```
</details>



<details>
<summary>ke_t5_super_glue_wsc_v102_simple_train</summary>

pre-processor description (t5)
```
 A typical example from SuperGLUE WSC might look like
  {
    'text': 'Mitchell asked Tom if he could lend some money.',
    'span1_text': 'Tom',
    'span2_text': 'he',
    'span2_index': 4,
  }

This will be transformed to
{
    'inputs': 'wsc: Bob asked Tom if *he* can lend some money.'
    'targets': 'Tom',
}
```

inputs
```
wsc: Bob asked Tom if *he* can lend some money.
```

targets
```
Tom
```
</details>



<details>
<summary>ke_t5_super_glue_wsc_v102_simple_eval</summary>

pre-processor description (t5)
```
 A typical example from SuperGLUE WSC might look like
  {
    'text': 'Mitchell asked Tom if he could lend some money.',
    'span1_text': 'Tom',
    'span2_text': 'he',
    'span2_index': 4,
  }

This will be transformed to
{
    'inputs': 'wsc: Bob asked Tom if *he* can lend some money.'
    'targets': 'Tom',
}
```

inputs
```
wsc: Bob asked Tom if *he* can lend some money.
```

targets
```
Tom
```
</details>



<details>
<summary>ke_t5_cnn_dailymail_v002</summary>

inputs
```
summarize: Sally Forrest, an actress-dancer who graced the silver screen throughout the '40s and '50s in MGM musicals and films such as the 1956 noir While the City Sleeps died on March 15 at her home in Beverly Hills, California. Forrest, whose birth name was Katherine Feeney, was 86 and had long battled cancer. Her publicist, Judith Goffin, announced the news Thursday. Scroll down for video. Actress: Sally Forrest was in the 1951 Ida Lupino-directed film 'Hard, Fast and Beautiful' (left) and the 1956 Fritz Lang movie 'While the City Sleeps' A San Diego native, Forrest became a protege of Hollywood trailblazer Ida Lupino, who cast her in starring roles in films including the critical and commercial success Not Wanted, Never Fear and Hard, Fast and Beautiful. Some of Forrest's other film credits included Bannerline, Son of Sinbad, and Excuse My Dust, according to her iMDB page. The page also indicates Forrest was in multiple Climax! and Rawhide television episodes. Forrest appeared as herself in an episode of The Ed Sullivan Show and three episodes of The Dinah Shore Chevy Show, her iMDB page says. She also starred in a Broadway production of The Seven Year Itch. City News Service reported that other stage credits included As You Like It, No, No, Nanette and Damn Yankees. Forrest married writer-producer Milo Frank in 1951. He died in 2004. She is survived by her niece, Sharon Durham, and nephews, Michael and Mark Feeney. Career: A San Diego native, Forrest became a protege of Hollywood trailblazer Ida Lupino, who cast her in starring roles in films.
```

targets
```
Sally Forrest, an actress-dancer who graced the silver screen throughout the '40s and '50s in MGM musicals and films died on March 15 . Forrest, whose birth name was Katherine Feeney, had long battled cancer . A San Diego native, Forrest became a protege of Hollywood trailblazer Ida Lupino, who cast her in starring roles in films .
```
</details>



<details>
<summary>ke_t5_squad_v010_allanswers</summary>

pre-processor description (t5)
```
SQuAD produces examples with this form:
    {'id': <id>, context': <article>, 'question': <question>,
     'answers': { 'text': [<n answers>] }}
This function will return examples of the format:
{'inputs': 'question: <question> context: <article>',
    'targets': '<answer_0>',
    'id': <id>, 'question': <question>, 'context': <context>,
    'answers': [<n answers>]},
```

inputs
```
question: {question} context: {article}
```

targets
```
{answer}
```
</details>



<details>
<summary>ke_t5_trivi_qa_v010</summary>

pre-processor description (t5)
```
TriviaQA produces examples with this form:
    {'entity_pages': {dict of wiki entities},
     'search_results': <dict of web search results>,
     'answer': {dict of all answers}, 'question': <question>,
     'question_id': <question_id>, 'question_source': <question_source>}
This function will return flattend examples of the format:
{'inputs': 'question: <question> context: <article>'
    'targets': 'answer: <sampled answer>'}
```

inputs
```
question: {question} context: {article}
```

targets
```
answer: {sampled answer}
```
</details>


<br>

### Mixture Tasks
\
Mixture Task는 여러 Task들의 묶음입니다. 

| Name  | Tasks |
| ------------- | ------------- |
| `ke_t5_glue_v002_proportional`, `ke_t5_glue_v002_equal`  | `ke_t5_glue_cola_v002`, `ke_t5_glue_sst2_v002`, `ke_t5_glue_mrpc_v002`, `ke_t5_glue_qqp_v002`, `ke_t5_glue_stsb_v002`, `ke_t5_glue_mnli_v002`, `ke_t5_glue_qnli_v002`, `ke_t5_glue_rte_v002`, `ke_t5_glue_mnli_mismatched_v002`, `ke_t5_glue_mnli_matched_v002`, `ke_t5_glue_ax_v002` |
| `ke_t5_super_glue_v102_proportional`, `ke_t5_super_glue_v102_equal`  | `ke_t5_dpr_v001_simple`, `ke_t5_super_glue_wsc_v102_simple_train`, `ke_t5_super_glue_wsc_v102_simple_eval`, `ke_t5_super_glue_boolq_v102`, `ke_t5_super_glue_cb_v102`, `ke_t5_super_glue_copa_v102`, `ke_t5_super_glue_multirc_v102`, `ke_t5_super_glue_record_v102`, `ke_t5_super_glue_rte_v102`, `ke_t5_super_glue_wic_v102`, `ke_t5_super_glue_axb_v102`, `ke_t5_super_glue_axg_v102` |
| `ke_t5_ko_text_classification_proportional`, `ke_t5_ko_text_classification_equal`  | `ke_t5_kor_copora_nsmc`, `ke_t5_kor_copora_qpair`, `ke_t5_kor_copora_kornli`, `ke_t5_kor_copora_korsts`, `ke_t5_kor_copora_khsd`, `ke_t5_nikl_cola` |
| `ke_t5_nikl_summary_mixture_equal`  | `ke_t5_nikl_summarization_summary`, `ke_t5_nikl_summarization_topic` |
| `ke_t5_ko_en_summary_proportional`, `ke_t5_ko_en_summary_equal`  | `ke_t5_cnn_dailymail_v002`, `ke_t5_nikl_summarization` |
| `ke_t5_ko_en_qa_proportional`, `ke_t5_ko_en_qa_equal`  | `ke_t5_korquad_allanswers`, `ke_t5_squad_v010_allanswers`, `ke_t5_trivia_qa_v010` |
| `ke_t5_ko_translation_proportional`, `ke_t5_ko_translation_equal`  | `ke_t5_ted_multi_en_ko`, `ke_t5_ted_multi_ko_en` |
| `ke_t5_all_proportional`, `ke_t5_ko_en_qa_equal`  | all tasks |


<br>

## Samples

몇몇 Downstream Task들의 I/O 샘플입니다.
해당 샘플을 생성한 모델은 아래와 같습니다.

### Model Spec
Dataset: `ke.newslike`\
Model Size: `small`\
steps: `1M`

<br>

### Summarization

<br>

#### NIKL Topic sentences

- source: https://news.naver.com/main/read.nhn?mode=LSD&mid=shm&sid1=102&oid=081&aid=0003173411
- press: 서울신문(www.seoul.co.kr)
- author: 이주원 기자 starjuwon@seoul.co.kr

`Input`
```
summarize_topic: “처음에는 ‘금방 끝나겠지’라고 생각했는데 어느덧 100일이 됐네요. \
그동안 춥고 아프고 힘들었지만 인간으로서 대우를 받을 수만 있다면 끝까지 버틸 수 있습니다.” \
LG트윈타워 청소 노동자들이 고용승계를 주장하며 파업에 나선지 100일째를 하루 앞둔 24일 \
서울 여의도 LG트윈타워 앞 ‘행복한 고용승계 텐트촌’에서 만난 박상설(63)씨는 힘들었던 투쟁 과정을 \
회상하며 눈시울을 붉혔다. 박씨는 2017년부터 LG트윈타워에서 청소 노동을 했지만 지난 1월 1일부로 \
계약이 종료돼 직장을 떠났다. 자동차 소음과 불편한 잠자리로 텐트에서 매일 밤잠을 설치지만 투쟁을 \
포기할 수 없다고 한다. 그는 “LG가 그동안 사회적 책임과 정도경영을 강조해 왔기에 파업이 금방 끝날 \
줄 알았다”며 “버티지 못하고 점점 떠나는 동지들을 바라볼 때마다 마음이 아프지만 정당한 노동 권리를 \
인정받기 위해 끝까지 투쟁할 것”이라고 강조했다. 지난해 11월 26일부터 파업에 돌입한 청소 \
노동자들은 25일 파업 100일째를 맞는다. 건물 1층 로비에서 시위를 하던 25명의 청소 노동자들은 지난 \
22일부터 정문 앞 도보에 텐트촌을 설치하고 장소를 옮겼다. 파업 100일에 맞춰 25일까지 시민연대와 \
함께 텐트 100개를 설치하고 주·야간 연대 시위를 이어가겠다는 뜻에서다. 노동자들은 한 명이 간신히 \
누울 수 있는 크기의 텐트 안에서 딱딱한 시멘트 바닥에 몸을 기대 쪽잠을 청하고 있다. LG트윈타워를 \
관리하는 LG그룹 계열사 ‘에스엔아이코퍼레이션’은 지난해 말 ‘지수아이앤씨’와 청소 용역 계약을 \
끝내고 다른 업체와 새로 계약했다. 사측은 ‘품질 저하’를 이유로 들었다. 반면 노동자들은 2019년 \
노조를 결성하고 권리를 주장하기 시작하면서 사측 눈 밖에 났다고 주장한다. 그동안 업체가 \
변경되더라도 기존 업체 노동자들이 새 업체에 고용승계가 되는 게 관례였지만 새 업체는 고용승계를 \
보장할 수 없다고 밝혔다. 지난달까지 고용노동부 중재로 수차례 노사 교섭이 있었지만 상황은 달라지지 \
않았다. 사측은 대신 노동자들에게 다른 사업장에서 일을 하게 해주겠다고 권유했다. 하지만 노동자들은 \
노조를 인정하지 않는 대기업의 행태를 묵인한 채 사측의 권유에 따른다면 어느 사업장에서 일을 하던 \
똑같은 행태가 반복될 수밖에 없다고 목소리를 높인다. 때문에 반드시 LG트윈타워에서 정당한 권리를 \
인정받고 노동을 이어가야만 한다고 말한다. 이들은 구광모 LG그룹 회장이 나서 문제를 해결해야 한다고 \
주장한다. 이혜정 LG트윈타워 공동대책위원회 집행위원은 “구 회장이 책임있는 답변을 내놓을 때까지 \
시민사회 단위와 함께 결의를 담아 끝까지 텐트촌을 유지할 것”이라고 강조했다.
```

\
`Prediction`
```
'LG트윈타워 청소 노동자가 고용승계를 주장하며 파업에 나선지 100일째를 하루 앞둔 24일 서울 \
여의도 LG트윈타워 앞 ‘행복한 고용승계 텐트촌’에서 만난 박상설(63)씨는 힘들었던 투쟁 과정을 \
회상하며 눈시울을 붉혔다. 반면 노동자들은 2019년 노조를 결성하고 권리를 주장하기 시작하면서 사측 \
눈 밖에 났다고 주장한다. 때문에 반드시 LG트윈타워에서 정당한 권리를 인정받고 노동을 이어가야 \
한다고 말한다.
```

<br>

#### NIKL Summary

- source: https://news.naver.com/main/read.nhn?mode=LSD&mid=shm&sid1=102&oid=081&aid=0003173411
- press: 서울신문(www.seoul.co.kr)
- author: 이주원 기자 starjuwon@seoul.co.kr

`Input`
```
summarize_topic: “처음에는 ‘금방 끝나겠지’라고 생각했는데 어느덧 100일이 됐네요. \
그동안 춥고 아프고 힘들었지만 인간으로서 대우를 받을 수만 있다면 끝까지 버틸 수 있습니다.” \
LG트윈타워 청소 노동자들이 고용승계를 주장하며 파업에 나선지 100일째를 하루 앞둔 24일 \
서울 여의도 LG트윈타워 앞 ‘행복한 고용승계 텐트촌’에서 만난 박상설(63)씨는 힘들었던 투쟁 과정을 \
회상하며 눈시울을 붉혔다. 박씨는 2017년부터 LG트윈타워에서 청소 노동을 했지만 지난 1월 1일부로 \
계약이 종료돼 직장을 떠났다. 자동차 소음과 불편한 잠자리로 텐트에서 매일 밤잠을 설치지만 투쟁을 \
포기할 수 없다고 한다. 그는 “LG가 그동안 사회적 책임과 정도경영을 강조해 왔기에 파업이 금방 끝날 \
줄 알았다”며 “버티지 못하고 점점 떠나는 동지들을 바라볼 때마다 마음이 아프지만 정당한 노동 권리를 \
인정받기 위해 끝까지 투쟁할 것”이라고 강조했다. 지난해 11월 26일부터 파업에 돌입한 청소 \
노동자들은 25일 파업 100일째를 맞는다. 건물 1층 로비에서 시위를 하던 25명의 청소 노동자들은 지난 \
22일부터 정문 앞 도보에 텐트촌을 설치하고 장소를 옮겼다. 파업 100일에 맞춰 25일까지 시민연대와 \
함께 텐트 100개를 설치하고 주·야간 연대 시위를 이어가겠다는 뜻에서다. 노동자들은 한 명이 간신히 \
누울 수 있는 크기의 텐트 안에서 딱딱한 시멘트 바닥에 몸을 기대 쪽잠을 청하고 있다. LG트윈타워를 \
관리하는 LG그룹 계열사 ‘에스엔아이코퍼레이션’은 지난해 말 ‘지수아이앤씨’와 청소 용역 계약을 \
끝내고 다른 업체와 새로 계약했다. 사측은 ‘품질 저하’를 이유로 들었다. 반면 노동자들은 2019년 \
노조를 결성하고 권리를 주장하기 시작하면서 사측 눈 밖에 났다고 주장한다. 그동안 업체가 \
변경되더라도 기존 업체 노동자들이 새 업체에 고용승계가 되는 게 관례였지만 새 업체는 고용승계를 \
보장할 수 없다고 밝혔다. 지난달까지 고용노동부 중재로 수차례 노사 교섭이 있었지만 상황은 달라지지 \
않았다. 사측은 대신 노동자들에게 다른 사업장에서 일을 하게 해주겠다고 권유했다. 하지만 노동자들은 \
노조를 인정하지 않는 대기업의 행태를 묵인한 채 사측의 권유에 따른다면 어느 사업장에서 일을 하던 \
똑같은 행태가 반복될 수밖에 없다고 목소리를 높인다. 때문에 반드시 LG트윈타워에서 정당한 권리를 \
인정받고 노동을 이어가야만 한다고 말한다. 이들은 구광모 LG그룹 회장이 나서 문제를 해결해야 한다고 \
주장한다. 이혜정 LG트윈타워 공동대책위원회 집행위원은 “구 회장이 책임있는 답변을 내놓을 때까지 \
시민사회 단위와 함께 결의를 담아 끝까지 텐트촌을 유지할 것”이라고 강조했다.
```

\
`Prediction`
```
'LG트윈타워 청소 노동자가 고용승계를 주장하며 파업에 나선지 100일째를 맞았다. LG트윈타워를 \
관리하는 LG그룹 계열사 ‘에스엔아이코퍼레이션’은 지난해 말 ‘지수아이앤씨’와 청소 용역 계약을 \
끝내고 다른 업체와 새로 계약했다. 그러나 노동자들은 노조를 인정하지 않는 대기업의 행태를 묵인한 \
채 사측의 권유에 따라 노동을 이어가라고 주장한다.'
```

<br>

#### CNN/DM

`Input`
```
summarize: Dougie Freedman is on the verge of agreeing a new two-year deal to remain at Nottingham 
Forest. Freedman has stabilised Forest since he replaced cult hero Stuart Pearce and the club's 
owners are pleased with the job he has done at the City Ground. Dougie Freedman is set to sign a new 
deal at Nottingham Forest. Freedman has impressed at the City Ground since replacing Stuart Pearce in 
February. They made an audacious attempt on the play-off places when Freedman replaced Pearce but 
have tailed off in recent weeks. That has not prevented Forest's ownership making moves to secure 
Freedman on a contract for the next two seasons.
```

\
`Target`
```
Nottingham Forest are close to extending Dougie Freedman's contract . The Forest boss took over from 
ormer manager Stuart Pearce in February . Freedman has since lead the club to ninth in the 
hampionship .
```

\
`Prediction`
```
Dougie Freedman set to sign a two-year deal at Nottingham Forest . Freedman replaced Stuart Pearce in 
February . Forest have made an audacious bid on the play-off places .
```

<br>

### Translation

```
en: Our goal is to make sure that no child is left behind , not because of cost or access .
ko: 우리의 목표는 어떤 아이도 소외시키지 않는 겁니다 . 비용이나 접근성 때문에 그런 일이 생기지 않게요 .
```

#### Ko2En
```
We aim is not to marginalize any child because of the cost or the access .
```

#### En2Ko
```
우리의 목표는 어떤 아이도 뒤처지지 않는 것입니다 . 원조나 지원에 의해서가 아니라요 .
```

<br>

## Performance

Downstream task들의 성능입니다. Mixture task들로 성능을 측정하였으며 시간 관계상 튜닝은 하지 않았습니다. \
성능을 높이기 위해서는 mixture task가 아닌 개별 task로 학습을 진행하십시오. (예를 들어 NIKL CoLA의 경우 개별 task로 학습을 진행할 경우 튜닝을 하지 않아도 small model에서 약 20의 성능이 나옵니다.)


Summarization과 Extractive QA의 경우 입력이 모델들의 최대 시퀀스 길이인 512를 넘을 수가 있습니다. 이 경우 넘는 부분을 그냥 버리고 학습과 성능측정을 진행했습니다. 즉, 요약의 경우 512 토큰을 초과하는 부분은 요약이 되지 않고, extractive QA의 경우 question에 대한 정답이 512 토큰을 초과하는 부분에 존재하면 답을 찾을 수 없습니다. 이러한 경우를 처리하여 성능을 높이기 위해서는 직접 학습 프로그램을 만들어 사용하셔야 합니다. (e.g. Extractive QA의 경우, BERT에서처럼 document span으로 분리하여 해당 span에 정답이 있는지 없는지로 학습.)


### Korean

Task: `ke_t5_ko_text_classification_proportional`

<details>
<summary>Click</summary>

<table>
    <thead>
        <tr>
            <th rowspan=2>Dataset</th>
            <th rowspan=2>Size</th>
            <th rowspan=2>steps</th>
            <th>NIKL CoLA</th>
            <th>NSMC</th>
            <th colspan=2>Question-pair</th>
            <th>KorNLI</th>
            <th colspan=2>KorSTS</th>
            <th>Hate Speech</th>
        </tr>
        <tr>
            <th>Mattew's</th>
            <th>Accuracy</th>
            <th>F1</th>
            <th>Accuracy</th>
            <th>Accuracy</th>
            <th>Pearson</th>
            <th>Spearman</th>
            <th>Accuracy</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=3>ke</td>
            <td>small</td>
            <td> 1M </td>
            <td> -3.716 </td> <!--NIKL CoLA Mattew's-->
            <td> 87.9 </td> <!--NSMC Accuracy-->
            <td> 87.9 </td> <!--Question-pair F1-->
            <td> 91.5 </td> <!--Question-pair Accuracy-->
            <td> 73.41 </td> <!--KorNLI Accuracy-->
            <td> 78.19 </td> <!--KorSTS Pearson-->
            <td> 77.9 </td> <!--KorSTS Spearman-->
            <td> 60.65 </td> <!--KHSD Accuracy-->
        </tr>
        <tr>
            <td>base</td>
            <td> 600K </td>
            <td> 12.51 </td> <!--NIKL CoLA Mattew's-->
            <td> 88.95 </td> <!--NSMC Accuracy-->
            <td> 93.7 </td> <!--Question-pair F1-->
            <td> 91.49 </td> <!--Question-pair Accuracy-->
            <td> 78.67 </td> <!--KorNLI Accuracy-->
            <td> 80.02 </td> <!--KorSTS Pearson-->
            <td> 79.73 </td> <!--KorSTS Spearman-->
            <td> 64.14 </td> <!--KHSD Accuracy-->
        </tr>
        <tr>
            <td>large</td>
            <td> 600K </td>
            <td> 13.31 </td> <!--NIKL CoLA Mattew's-->
            <td> 89.7 </td> <!--NSMC Accuracy-->
            <td> 89.74 </td> <!--Question-pair F1-->
            <td> 92.52 </td> <!--Question-pair Accuracy-->
            <td> 79.76 </td> <!--KorNLI Accuracy-->
            <td> 83.65 </td> <!--KorSTS Pearson-->
            <td> 83.25 </td> <!--KorSTS Spearman-->
            <td> 62.82 </td> <!--KHSD Accuracy-->
        </tr>
        <tr>
            <td rowspan=3>ke.newslike</td>
            <td>small</td>
            <td> 1M </td>
            <td> 4.088 </td> <!--NIKL CoLA Mattew's-->
            <td> 88.09 </td> <!--NSMC Accuracy-->
            <td> 90.55 </td> <!--Question-pair F1-->
            <td> 91.94 </td> <!--Question-pair Accuracy-->
            <td> 74.82 </td> <!--KorNLI Accuracy-->
            <td> 76.76 </td> <!--KorSTS Pearson-->
            <td> 77.15 </td> <!--KorSTS Spearman-->
            <td> 60.89 </td> <!--KHSD Accuracy-->
        </tr>
        <tr>
            <td>base</td>
            <td> 600K </td>
            <td> 8.773 </td> <!--NIKL CoLA Mattew's-->
            <td> 87.07 </td> <!--NSMC Accuracy-->
            <td> 89.52 </td> <!--Question-pair F1-->
            <td> 92.38 </td> <!--Question-pair Accuracy-->
            <td> 76.1 </td> <!--KorNLI Accuracy-->
            <td> 79.38 </td> <!--KorSTS Pearson-->
            <td> 79.25 </td> <!--KorSTS Spearman-->
            <td> 61.85 </td> <!--KHSD Accuracy-->
        </tr>
        <tr>
            <td>large</td>
            <td> 600K </td>
            <td> 5.145 </td> <!--NIKL CoLA Mattew's-->
            <td> 88.97 </td> <!--NSMC Accuracy-->
            <td> 89.96 </td> <!--Question-pair F1-->
            <td> 92.62 </td> <!--Question-pair Accuracy-->
            <td> 79.48 </td> <!--KorNLI Accuracy-->
            <td> 80.86 </td> <!--KorSTS Pearson-->
            <td> 81.29 </td> <!--KorSTS Spearman-->
            <td> 61.25 </td> <!--KHSD Accuracy-->
        </tr>
        <tr>
            <td rowspan=2>ko</td>
            <td>small</td>
            <td> 1M </td>
            <td> 0.7339 </td> <!--NIKL CoLA Mattew's-->
            <td> 86.2 </td> <!--NSMC Accuracy-->
            <td> 88.93 </td> <!--Question-pair F1-->
            <td> 92.08 </td> <!--Question-pair Accuracy-->
            <td> 74.38 </td> <!--KorNLI Accuracy-->
            <td> 77.24 </td> <!--KorSTS Pearson-->
            <td> 75.99 </td> <!--KorSTS Spearman-->
            <td> 59.93 </td> <!--KHSD Accuracy-->
        </tr>
        <tr>
            <td>base</td>
            <td> 600K </td>
            <td> </td> <!--NIKL CoLA Mattew's-->
            <td> </td> <!--NSMC Accuracy-->
            <td> </td> <!--Question-pair F1-->
            <td> </td> <!--Question-pair Accuracy-->
            <td> </td> <!--KorNLI Accuracy-->
            <td> </td> <!--KorSTS Pearson-->
            <td> </td> <!--KorSTS Spearman-->
            <td> </td> <!--KHSD Accuracy-->
        </tr>
    </tbody>
</table>

</details>

### English

#### GLUE

Task: `ke_t5_glue_v002_proportional`

<details>
<summary>Click</summary>

<table>
    <thead>
        <tr>
            <th rowspan=2>Dataset</th>
            <th rowspan=2>Size</th>
            <th rowspan=2>steps</th>
            <th>GLUE</th>
            <th>CoLA</th>
            <th>SST-2</th>
            <th colspan=2>MRPC</th>
            <th colspan=2>STS-B</th>
        </tr>
        <tr>
            <th>Average</th>
            <th>Mattew's</th>
            <th>Accuracy</th>
            <th>F1</th>
            <th>Accuracy</th>
            <th>Pearson</th>
            <th>Spearman</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=3>ke</td>
            <td>small</td>
            <td> 1M </td>
            <td> </td> <!--GLUE average-->
            <td> 27.31 </td> <!--CoLA Mattew's-->
            <td> 89.11 </td> <!--SST-2 Accuracy-->
            <td> 88.69 </td> <!--MRPC F1-->
            <td> 84.31 </td> <!--MRPC Accuracy-->
            <td> 81.14 </td> <!--STS-B Pearson-->
            <td> 81.38 </td> <!--STS-B Spearman-->
        </tr>
        <tr>
            <td>base</td>
            <td> 600K </td>
            <td> </td> <!--GLUE average-->
            <td> 38.26 </td> <!--CoLA Mattew's-->
            <td> 83.73 </td> <!--SST-2 Accuracy-->
            <td> 90.43 </td> <!--MRPC F1-->
            <td> 86.76 </td> <!--MRPC Accuracy-->
            <td> 85.8 </td> <!--STS-B Pearson-->
            <td> 85.82 </td> <!--STS-B Spearman-->
        </tr>
        <tr>
            <td>large</td>
            <td> 600K </td>
            <td> </td> <!--GLUE average-->
            <td> 39.85 </td> <!--CoLA Mattew's-->
            <td> 91.28 </td> <!--SST-2 Accuracy-->
            <td> 89.05 </td> <!--MRPC F1-->
            <td> 85.05 </td> <!--MRPC Accuracy-->
            <td> 88.14 </td> <!--STS-B Pearson-->
            <td> 88.14 </td> <!--STS-B Spearman-->
        </tr>
        <tr>
            <td rowspan=3>ke.newslike</td>
            <td>small</td>
            <td> 1M </td>
            <td> </td> <!--GLUE average-->
            <td> 28.04 </td> <!--CoLA Mattew's-->
            <td> 88.07 </td> <!--SST-2 Accuracy-->
            <td> 88.7 </td> <!--MRPC F1-->
            <td> 85.29 </td> <!--MRPC Accuracy-->
            <td> 84.6 </td> <!--STS-B Pearson-->
            <td> 84.33 </td> <!--STS-B Spearman-->
        </tr>
        <tr>
            <td>base</td>
            <td> 600K </td>
            <td> </td> <!--GLUE average-->
            <td> 48.18 </td> <!--CoLA Mattew's-->
            <td> 91.4  </td> <!--SST-2 Accuracy-->
            <td> 88.93 </td> <!--MRPC F1-->
            <td> 84.56 </td> <!--MRPC Accuracy-->
            <td> 86.63 </td> <!--STS-B Pearson-->
            <td> 86.32 </td> <!--STS-B Spearman-->
        </tr>
        <tr>
            <td>large</td>
            <td> 600K </td>
            <td></td> <!--GLUE average-->
            <td> 45.26 </td> <!--CoLA Mattew's-->
            <td> 92.32 </td> <!--SST-2 Accuracy-->
            <td> 89.32 </td> <!--MRPC F1-->
            <td> 85.05 </td> <!--MRPC Accuracy-->
            <td> 87.92 </td> <!--STS-B Pearson-->
            <td> 87.85 </td> <!--STS-B Spearman-->
        </tr>
        <tr>
            <td rowspan=2>ko</td>
            <td>small</td>
            <td> 1M </td>
            <td> </td> <!--GLUE average-->
            <td> 8.699 </td> <!--CoLA Mattew's-->
            <td> 82.8 </td> <!--SST-2 Accuracy-->
            <td> 85.46 </td> <!--MRPC F1-->
            <td> 79.66 </td> <!--MRPC Accuracy-->
            <td> 76.75 </td> <!--STS-B Pearson-->
            <td> 76.35 </td> <!--STS-B Spearman-->
        </tr>
        <tr>
            <td>base</td>
            <td> 600K </td>
            <td> </td> <!--GLUE average-->
            <td> 5.622 </td> <!--CoLA Mattew's-->
            <td> 82.23 </td> <!--SST-2 Accuracy-->
            <td> 83.2 </td> <!--MRPC F1-->
            <td> 84.8 </td> <!--MRPC Accuracy-->
            <td> 80.78 </td> <!--STS-B Pearson-->
            <td> 80.89 </td> <!--STS-B Spearman-->
        </tr>
    </tbody>
</table>

<table>
    <thead>
        <tr>
            <th rowspan=2>Dataset</th>
            <th rowspan=2>Size</th>
            <th rowspan=2>steps</th>
            <th colspan=2>QQP</th>
            <th>MNLI-m</th>
            <th>MNLI-mm</th>
            <th>QNLI</th>
            <th>RTE</th>
            <th>WNLI</th>
        </tr>
        <tr>
            <th>F1</th>
            <th>Accuracy</th>
            <th>Accuracy</th>
            <th>Accuracy</th>
            <th>Accuracy</th>
            <th>Accuracy</th>
            <th>Accuracy</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=3>ke</td>
            <td>small</td>
            <td> 1M </td>
            <td> 83.54 </td> <!--QQP F1-->
            <td> 89.07 </td> <!--QQP Accuracy-->
            <td> 78.06 </td> <!--MNLI-m Accuracy-->
            <td> 78.94 </td> <!--MNLI-mm Accuracy-->
            <td> 86.55 </td> <!--QNLI Accuracy-->
            <td> 64.26 </td> <!--RTE Accuracy-->
            <td> </td> <!--WNLI Accuracy-->
        </tr>
        <tr>
            <td>base</td>
            <td> 600K </td>
            <td> 90.19 </td> <!--QQP F1-->
            <td> 86.78 </td> <!--QQP Accuracy-->
            <td> 83.73 </td> <!--MNLI-m Accuracy-->
            <td> 83.86 </td> <!--MNLI-mm Accuracy-->
            <td> 89.79 </td> <!--QNLI Accuracy-->
            <td> 79.42 </td> <!--RTE Accuracy-->
            <td> </td> <!--WNLI Accuracy-->
        </tr>
        <tr>
            <td>large</td>
            <td> 600K </td>
            <td> 86.5 </td> <!--QQP F1-->
            <td> 89.86 </td> <!--QQP Accuracy-->
            <td> 83.73 </td> <!--MNLI-m Accuracy-->
            <td> 84.39 </td> <!--MNLI-mm Accuracy-->
            <td> 90.21 </td> <!--QNLI Accuracy-->
            <td> 79.42 </td> <!--RTE Accuracy-->
            <td> </td> <!--WNLI Accuracy-->
        </tr>
        <tr>
            <td rowspan=3>ke.newslike</td>
            <td>small</td>
            <td> 1M </td>
            <td> 86.06 </td> <!--QQP F1-->
            <td> 89.41 </td> <!--QQP Accuracy-->
            <td> 78.61 </td> <!--MNLI-m Accuracy-->
            <td> 79.34 </td> <!--MNLI-mm Accuracy-->
            <td> 85.85 </td> <!--QNLI Accuracy-->
            <td> 67.15 </td> <!--RTE Accuracy-->
            <td> </td> <!--WNLI Accuracy-->
        </tr>
        <tr>
            <td>base</td>
            <td> 600K </td>
            <td> 86.91 </td> <!--QQP F1-->
            <td> 90.43 </td> <!--QQP Accuracy-->
            <td> 82.79 </td> <!--MNLI-m Accuracy-->
            <td> 83.44 </td> <!--MNLI-mm Accuracy-->
            <td> 90 </td> <!--QNLI Accuracy-->
            <td> 76.9 </td> <!--RTE Accuracy-->
            <td> </td> <!--WNLI Accuracy-->
        </tr>
        <tr>
            <td>large</td>
            <td> 600K </td>
            <td> 87.67 </td> <!--QQP F1-->
            <td> 90.64 </td> <!--QQP Accuracy-->
            <td> 86.14 </td> <!--MNLI-m Accuracy-->
            <td> 85.78 </td> <!--MNLI-mm Accuracy-->
            <td> 92.04 </td> <!--QNLI Accuracy-->
            <td> 82.67 </td> <!--RTE Accuracy-->
            <td> </td> <!--WNLI Accuracy-->
        </tr>
        <tr>
            <td rowspan=3>ko</td>
            <td>small</td>
            <td> 1M </td>
            <td> 83.48 </td> <!--QQP F1-->
            <td> 87.61 </td> <!--QQP Accuracy-->
            <td> 72.68 </td> <!--MNLI-m Accuracy-->
            <td> 72.91 </td> <!--MNLI-mm Accuracy-->
            <td> 87.58 </td> <!--QNLI Accuracy-->
            <td> 62.82 </td> <!--RTE Accuracy-->
            <td> </td> <!--WNLI Accuracy-->
        </tr>
        <tr>
            <td>base</td>
            <td> 600K </td>
            <td> 85.06 </td> <!--QQP F1-->
            <td> 88.79 </td> <!--QQP Accuracy-->
            <td> 75.62 </td> <!--MNLI-m Accuracy-->
            <td> 75.97 </td> <!--MNLI-mm Accuracy-->
            <td> 82.35 </td> <!--QNLI Accuracy-->
            <td> 62.45 </td> <!--RTE Accuracy-->
            <td> </td> <!--WNLI Accuracy-->
        </tr>
    </tbody>
</table>

</details>

#### Super GLUE

Task: `ke_t5_super_glue_v102_proportional`

<details>
<summary>Click</summary>

<table>
    <thead>
        <tr>
            <th rowspan=2>Dataset</th>
            <th rowspan=2>Size</th>
            <th rowspan=2>steps</th>
            <th>SuperGLUE</th>
            <th>BoolQ</th>
            <th colspan=2>CB</th>
            <th>COPA</th>
            <th colspan=2>MultiRC</th>
            <th colspan=2>ReCoRD</th>
            <th>RTE</th>
            <th>WiC</th>
            <th>WSC</th>
        </tr>
        <tr>
            <th>Average</th>
            <th>Accuracy</th>
            <th>F1</th>
            <th>Accuracy</th>
            <th>Accuracy</th>
            <th>F1a</th>
            <th>EM</th>
            <th>F1</th>
            <th>EM</th>
            <th>Accuracy</th>
            <th>Accuracy</th>
            <th>Accuracy</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=3>ke</td>
            <td>small</td>
            <td> 1M </td>
            <td> </td> <!--Super GLUE Average-->
            <td> 70.86 </td> <!--BoolQ Accuracy-->
            <td> 70.34 </td> <!--CB F1-->
            <td> 76.79 </td> <!--CB Accuracy-->
            <td> 54 </td> <!--COPA Accuracy-->
            <td> 65.57 </td> <!--MultiRC F1a-->
            <td> 17.94 </td> <!--MultiRC EM-->
            <td> 63.86 </td> <!--ReCoRD F1-->
            <td> 61.87 </td> <!--ReCoRD Accuracy-->
            <td> 63.9 </td> <!--RTE Accuracy-->
            <td> 60.97 </td> <!--WiC Accuracy-->
            <td> 59.25 </td> <!--WSC Accuracy-->
        </tr>
        <tr>
            <td>base</td>
            <td> 600K </td>
            <td> </td> <!--Super GLUE Average-->
            <td> 77.31 </td> <!--BoolQ Accuracy-->
            <td> 73.08 </td> <!--CB F1-->
            <td> 87.5 </td> <!--CB Accuracy-->
            <td> 72 </td> <!--COPA Accuracy-->
            <td> 73.24 </td> <!--MultiRC F1a-->
            <td> 31.9 </td> <!--MultiRC EM-->
            <td> 76.9 </td> <!--ReCoRD F1-->
            <td> 76.07 </td> <!--ReCoRD Accuracy-->
            <td> 79.78 </td> <!--RTE Accuracy-->
            <td> 64.73 </td> <!--WiC Accuracy-->
            <td> 74.04 </td> <!--WSC Accuracy-->
        </tr>
        <tr>
            <td>large</td>
            <td> 600K </td>
            <td> </td> <!--Super GLUE Average-->
            <td> 76.06 </td> <!--BoolQ Accuracy-->
            <td> 61 </td> <!--CB F1-->
            <td> 87.5 </td> <!--CB Accuracy-->
            <td> 67 </td> <!--COPA Accuracy-->
            <td> 76.25 </td> <!--MultiRC F1a-->
            <td> 36.62 </td> <!--MultiRC EM-->
            <td> 81.29 </td> <!--ReCoRD F1-->
            <td> 80.31 </td> <!--ReCoRD Accuracy-->
            <td> 82.31 </td> <!--RTE Accuracy-->
            <td> 63.95 </td> <!--WiC Accuracy-->
            <td> 72.12 </td> <!--WSC Accuracy-->
        </tr>
        <tr>
            <td rowspan=3>ke.newslike</td>
            <td>small</td>
            <td> 1M </td>
            <td> </td> <!--Super GLUE Average-->
            <td> 70.8 </td> <!--BoolQ Accuracy-->
            <td> 83.93 </td> <!--CB F1-->
            <td> 85.24 </td> <!--CB Accuracy-->
            <td> 55 </td> <!--COPA Accuracy-->
            <td> 67.35 </td> <!--MultiRC F1a-->
            <td> 20.15 </td> <!--MultiRC EM-->
            <td> 63.85 </td> <!--ReCoRD F1-->
            <td> 62.06 </td> <!--ReCoRD Accuracy-->
            <td> 66.06 </td> <!--RTE Accuracy-->
            <td> 68.03 </td> <!--WiC Accuracy-->
            <td> 67.31 </td> <!--WSC Accuracy-->
        </tr>
        <tr>
            <td>base</td>
            <td> 600K </td>
            <td> </td> <!--Super GLUE Average-->
            <td> 73.77 </td> <!--BoolQ Accuracy-->
            <td> 60.29 </td> <!--CB F1-->
            <td> 85.71 </td> <!--CB Accuracy-->
            <td> 60.85 </td> <!--COPA Accuracy-->
            <td> 72.56 </td> <!--MultiRC F1a-->
            <td> 32.32 </td> <!--MultiRC EM-->
            <td> 77.35 </td> <!--ReCoRD F1-->
            <td> 74.62</td> <!--ReCoRD Accuracy-->
            <td> 76.9 </td> <!--RTE Accuracy-->
            <td> 67.24 </td> <!--WiC Accuracy-->
            <td> 73.08 </td> <!--WSC Accuracy-->
        </tr>
        <tr>
            <td>large</td>
            <td> 600K </td>
            <td> </td> <!--Super GLUE Average-->
            <td> 79.36 </td> <!--BoolQ Accuracy-->
            <td> 75.99 </td> <!--CB F1-->
            <td> 82.29 </td> <!--CB Accuracy-->
            <td> 66 </td> <!--COPA Accuracy-->
            <td> 76.54 </td> <!--MultiRC F1a-->
            <td> 39.24 </td> <!--MultiRC EM-->
            <td> 83.05 </td> <!--ReCoRD F1-->
            <td> 82.2 </td> <!--ReCoRD Accuracy-->
            <td> 79.42 </td> <!--RTE Accuracy-->
            <td> 69.44 </td> <!--WiC Accuracy-->
            <td> 77.88 </td> <!--WSC Accuracy-->
        </tr>
        <tr>
            <td rowspan=3>ko</td>
            <td>small</td>
            <td> 1M </td>
            <td> </td> <!--Super GLUE Average-->
            <td> 66.18 </td> <!--BoolQ Accuracy-->
            <td> 53.08 </td> <!--CB F1-->
            <td> 66.07 </td> <!--CB Accuracy-->
            <td> 52 </td> <!--COPA Accuracy-->
            <td> 63.31 </td> <!--MultiRC F1a-->
            <td> 18.47 </td> <!--MultiRC EM-->
            <td> 47.41 </td> <!--ReCoRD F1-->
            <td> 46.42 </td> <!--ReCoRD Accuracy-->
            <td> 58.84 </td> <!--RTE Accuracy-->
            <td> 58.31 </td> <!--WiC Accuracy-->
            <td> 60.58 </td> <!--WSC Accuracy-->
        </tr>
        <tr>
            <td>base</td>
            <td> 600K </td>
            <td> </td> <!--Super GLUE Average-->
            <td> 69.66 </td> <!--BoolQ Accuracy-->
            <td> 56.5 </td> <!--CB F1-->
            <td> 69.64 </td> <!--CB Accuracy-->
            <td> 49 </td> <!--COPA Accuracy-->
            <td> 67.19 </td> <!--MultiRC F1a-->
            <td> 18.36 </td> <!--MultiRC EM-->
            <td> 55.45 </td> <!--ReCoRD F1-->
            <td> 54.51 </td> <!--ReCoRD Accuracy-->
            <td> 66.06 </td> <!--RTE Accuracy-->
            <td> 63.95 </td> <!--WiC Accuracy-->
            <td> 61.06 </td> <!--WSC Accuracy-->
        </tr>
    </tbody>
</table>

</details>

### Korean - English
#### Extractive QA

Task: `ke_t5_ko_en_qa_equal`

<details>
<summary>Click</summary>

<table>
    <thead>
        <tr>
            <th rowspan=2>Dataset</th>
            <th rowspan=2>Size</th>
            <th rowspan=2>steps</th>
            <th colspan=2>SQuAD</th>
            <th colspan=2>KorQuAD 1.1</th>
        </tr>
        <tr>
            <th>EM</th>
            <th>F1</th>
            <th>EM</th>
            <th>F1</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=3>ke</td>
            <td>small</td>
            <td> 1M </td>
            <td> 72.88 </td> <!--SQuAD EM-->
            <td> 82.8 </td> <!--SQuAD F1-->
            <td> 82.16 </td> <!--KorQuAD 1.1 EM-->
            <td> 88.39 </td> <!--KorQuAD 1.1 F1-->
        </tr>
        <tr>
            <td>base</td>
            <td> 600K </td>
            <td> 78.43 </td> <!--SQuAD EM-->
            <td> 88.01 </td> <!--SQuAD F1-->
            <td> 85.45 </td> <!--KorQuAD 1.1 EM-->
            <td> 91.11 </td> <!--KorQuAD 1.1 F1-->
        </tr>
        <tr>
            <td>large</td>
            <td> 600K </td>
            <td> 81.33 </td> <!--SQuAD EM-->
            <td> 90.03 </td> <!--SQuAD F1-->
            <td> 86.27 </td> <!--KorQuAD 1.1 EM-->
            <td> 92.06 </td> <!--KorQuAD 1.1 F1-->
        </tr>
        <tr>
            <td rowspan=3>ke.newslike</td>
            <td>small</td>
            <td> 1M </td>
            <td> 73.75 </td> <!--SQuAD EM-->
            <td> 84.45 </td> <!--SQuAD F1-->
            <td> 83.25 </td> <!--KorQuAD 1.1 EM-->
            <td> 89.17 </td> <!--KorQuAD 1.1 F1-->
        </tr>
        <tr>
            <td>base</td>
            <td> 600K </td>
            <td> 79.21 </td> <!--SQuAD EM-->
            <td> 88.02 </td> <!--SQuAD F1-->
            <td> 85.36 </td> <!--KorQuAD 1.1 EM-->
            <td> 91.22 </td> <!--KorQuAD 1.1 F1-->
        </tr>
        <tr>
            <td>large</td>
            <td> 600K </td>
            <td> 81.06 </td> <!--SQuAD EM-->
            <td> 90.35 </td> <!--SQuAD F1-->
            <td> 86.76 </td> <!--KorQuAD 1.1 EM-->
            <td> 92.21 </td> <!--KorQuAD 1.1 F1-->
        </tr>
        <tr>
            <td rowspan=3>ko</td>
            <td>small</td>
            <td> 1M </td>
            <td> 63.89 </td> <!--SQuAD EM-->
            <td> 75.13 </td> <!--SQuAD F1-->
            <td> 82.78 </td> <!--KorQuAD 1.1 EM-->
            <td> 88.99 </td> <!--KorQuAD 1.1 F1-->
        </tr>
        <tr>
            <td>base</td>
            <td> 600K </td>
            <td> 69.52 </td> <!--SQuAD EM-->
            <td> 80.29 </td> <!--SQuAD F1-->
            <td> 85.4 </td> <!--KorQuAD 1.1 EM-->
            <td> 91.35 </td> <!--KorQuAD 1.1 F1-->
        </tr>
    </tbody>
</table>

</details>

#### Translation

Task: `ke_t5_ko_translation_proportional`

<details>
<summary>Click</summary>

<table>
    <thead>
        <tr>
            <th rowspan=3>Dataset</th>
            <th rowspan=3>Size</th>
            <th rowspan=3>steps</th>
            <th colspan=8>TED multilingual</th>
        </tr>
        <tr>
            <th colspan=3>en->ko</th>
            <th colspan=3>ko->en</th>
        </tr>
        <tr>
            <th>Rouge-1</th>
            <th>Rouge-2</th>
            <th>Rouge-L</th>
            <th>Rouge-1</th>
            <th>Rouge-2</th>
            <th>Rouge-L</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=3>ke</td>
            <td>small</td>
            <td> 1M </td>
            <td> 10.02 </td> <!--en->ko Rouge-1-->
            <td> 2.072 </td> <!--en->ko Rouge-2-->
            <td> 9.951 </td> <!--en->ko Rouge-L-->
            <td> 39.19 </td> <!--ko->en Rouge-1-->
            <td> 19.78 </td> <!--ko->en Rouge-2-->
            <td> 35.15 </td> <!--ko->en Rouge-L-->
        </tr>
        <tr>
            <td>base</td>
            <td> 600K </td>
            <td> 12.03 </td> <!--en->ko Rouge-1-->
            <td> 2.805 </td> <!--en->ko Rouge-2-->
            <td> 11.94 </td> <!--en->ko Rouge-L-->
            <td> 44.12 </td> <!--ko->en Rouge-1-->
            <td> 19.76 </td> <!--ko->en Rouge-2-->
            <td> 39.35 </td> <!--ko->en Rouge-L-->
        </tr>
        <tr>
            <td>large</td>
            <td> 600K </td>
            <td> 11.45 </td> <!--en->ko Rouge-1-->
            <td> 2.96 </td> <!--en->ko Rouge-2-->
            <td> </td> <!--en->ko Rouge-L-->
            <td> 44.52 </td> <!--ko->en Rouge-1-->
            <td> 20.21 </td> <!--ko->en Rouge-2-->
            <td> </td> <!--ko->en Rouge-L-->
        </tr>
        <tr>
            <td rowspan=3>ke.newslike</td>
            <td>small</td>
            <td> 1M </td>
            <td> 11.69 </td> <!--en->ko Rouge-1-->
            <td> 2.524 </td> <!--en->ko Rouge-2-->
            <td> 10.58 </td> <!--en->ko Rouge-L-->
            <td> 39.66 </td> <!--ko->en Rouge-1-->
            <td> 19.74 </td> <!--ko->en Rouge-2-->
            <td> 40.06 </td> <!--ko->en Rouge-L-->
        </tr>
        <tr>
            <td>base</td>
            <td> 600K </td>
            <td> 12.21 </td> <!--en->ko Rouge-1-->
            <td> 2.855 </td> <!--en->ko Rouge-2-->
            <td> 12.09 </td> <!--en->ko Rouge-L-->
            <td> 44.43 </td> <!--ko->en Rouge-1-->
            <td> 20.03 </td> <!--ko->en Rouge-2-->
            <td> 40.49 </td> <!--ko->en Rouge-L-->
        </tr>
        <tr>
            <td>large</td>
            <td> 600K </td>
            <td> 11.73 </td> <!--en->ko Rouge-1-->
            <td> 2.948 </td> <!--en->ko Rouge-2-->
            <td> 11.61 </td> <!--en->ko Rouge-L-->
            <td> 46.84 </td> <!--ko->en Rouge-1-->
            <td> 22.29 </td> <!--ko->en Rouge-2-->
            <td> 42.98 </td> <!--ko->en Rouge-L-->
        </tr>
        <tr>
            <td rowspan=3>ko</td>
            <td>small</td>
            <td> 1M </td>
            <td> 10.3 </td> <!--en->ko Rouge-1-->
            <td> 1.993 </td> <!--en->ko Rouge-2-->
            <td> 10.16 </td> <!--en->ko Rouge-L-->
            <td> 38.96 </td> <!--ko->en Rouge-1-->
            <td> 14.79 </td> <!--ko->en Rouge-2-->
            <td> 35.02 </td> <!--ko->en Rouge-L-->
        </tr>
        <tr>
            <td>base</td>
            <td> 1M </td>
            <td> 11.4 </td> <!--en->ko Rouge-1-->
            <td> 2.481 </td> <!--en->ko Rouge-2-->
            <td> 11.25 </td> <!--en->ko Rouge-L-->
            <td> 42.43 </td> <!--ko->en Rouge-1-->
            <td> 17.45 </td> <!--ko->en Rouge-2-->
            <td> 38.25 </td> <!--ko->en Rouge-L-->
        </tr>
    </tbody>
</table>

</details>

#### Summarization

Task: `ke_t5_ko_en_summary_proportional` for **CNN/DM**, `ke_t5_nikl_summary_mixture_equal` for **NIKL summarization**

<details>
<summary>Click</summary>
<table>
    <thead>
        <tr>
            <th rowspan=3>Dataset</th>
            <th rowspan=3>Size</th>
            <th rowspan=3>steps</th>
            <th colspan=3 rowspan=2>CNN/DM</th>
            <th colspan=6>NIKL summarization</th>
        </tr>
        <tr>
            <th colspan=3>summary</th>
            <th colspan=3>topic</th>
        </tr>
        <tr>
            <th>Rouge-1</th>
            <th>Rouge-2</th>
            <th>Rouge-L</th>
            <th>Rouge-1</th>
            <th>Rouge-2</th>
            <th>Rouge-L</th>
            <th>Rouge-1</th>
            <th>Rouge-2</th>
            <th>Rouge-L</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=3>ke</td>
            <td>small</td>
            <td> 1M </td>
            <td> 37.94 </td> <!--CNN/DM Rouge-1-->
            <td> 17.9 </td> <!--CNN/DM Rouge-2-->
            <td> 35.86 </td> <!--CNN/DM Rouge-L-->
            <td> 38.85 </td> <!--NIKL summary Rouge-1-->
            <td> 18.65 </td> <!--NIKL summary Rouge-2-->
            <td> 37.35 </td> <!--NIKL summary Rouge-L-->
            <td> 48.79 </td> <!--NIKL topic Rouge-1-->
            <td> 32.51 </td> <!--NIKL topic Rouge-2-->
            <td> 47.75 </td> <!--NIKL topic Rouge-L-->
        </tr>
        <tr>
            <td>base</td>
            <td> 600K </td>
            <td> 37.84 </td> <!--CNN/DM Rouge-1-->
            <td> 15.38 </td> <!--CNN/DM Rouge-2-->
            <td> 38.33 </td> <!--CNN/DM Rouge-L-->
            <td> 40.86 </td> <!--NIKL summary Rouge-1-->
            <td> 19.58 </td> <!--NIKL summary Rouge-2-->
            <td> 39.36 </td> <!--NIKL summary Rouge-L-->
            <td> 50.71 </td> <!--NIKL topic Rouge-1-->
            <td> 35.43 </td> <!--NIKL topic Rouge-2-->
            <td> 50.35 </td> <!--NIKL topic Rouge-L-->
        </tr>
        <tr>
            <td>large</td>
            <td> 600K </td>
            <td> 40.15 </td> <!--CNN/DM Rouge-1-->
            <td> 17.78 </td> <!--CNN/DM Rouge-2-->
            <td> </td> <!--CNN/DM Rouge-L-->
            <td> 40.54 </td> <!--NIKL summary Rouge-1-->
            <td> 20.04 </td> <!--NIKL summary Rouge-2-->
            <td> 39.25 </td> <!--NIKL summary Rouge-L-->
            <td> 55.52 </td> <!--NIKL topic Rouge-1-->
            <td> 37.72 </td> <!--NIKL topic Rouge-2-->
            <td> 54.78 </td> <!--NIKL topic Rouge-L-->
        </tr>
        <tr>
            <td rowspan=3>ke.newslike</td>
            <td>small</td>
            <td> 1M </td>
            <td> 38.17 </td> <!--CNN/DM Rouge-1-->
            <td> 17.59 </td> <!--CNN/DM Rouge-2-->
            <td> 36.52 </td> <!--CNN/DM Rouge-L-->
            <td> 38.28 </td> <!--NIKL summary Rouge-1-->
            <td> 18.15 </td> <!--NIKL summary Rouge-2-->
            <td> 37.18 </td> <!--NIKL summary Rouge-L-->
            <td> 50.11 </td> <!--NIKL topic Rouge-1-->
            <td> 33.09 </td> <!--NIKL topic Rouge-2-->
            <td> 49.94 </td> <!--NIKL topic Rouge-L-->
        </tr>
        <tr>
            <td>base</td>
            <td> 600K </td>
            <td> 40.64 </td> <!--CNN/DM Rouge-1-->
            <td> 18.98 </td> <!--CNN/DM Rouge-2-->
            <td> 38.03 </td> <!--CNN/DM Rouge-L-->
            <td> 40.54 </td> <!--NIKL summary Rouge-1-->
            <td> 20.44 </td> <!--NIKL summary Rouge-2-->
            <td> 39.23 </td> <!--NIKL summary Rouge-L-->
            <td> 54 </td> <!--NIKL topic Rouge-1-->
            <td> 37.63 </td> <!--NIKL topic Rouge-2-->
            <td> 53.2 </td> <!--NIKL topic Rouge-L-->
        </tr>
        <tr>
            <td>large</td>
            <td> 600K </td>
            <td> 39.48 </td> <!--CNN/DM Rouge-1-->
            <td> 17.46 </td> <!--CNN/DM Rouge-2-->
            <td> 36.79 </td> <!--CNN/DM Rouge-L-->
            <td> 39.76 </td> <!--NIKL summary Rouge-1-->
            <td> 18.12 </td> <!--NIKL summary Rouge-2-->
            <td> 38.25 </td> <!--NIKL summary Rouge-L-->
            <td> 53.58 </td> <!--NIKL topic Rouge-1-->
            <td> 38.17 </td> <!--NIKL topic Rouge-2-->
            <td> 53.15 </td> <!--NIKL topic Rouge-L-->
        </tr>
        <tr>
            <td rowspan=3>ko</td>
            <td>small</td>
            <td> 1M </td>
            <td> 37.39 </td> <!--CNN/DM Rouge-1-->
            <td> 16.49 </td> <!--CNN/DM Rouge-2-->
            <td> 35.73 </td> <!--CNN/DM Rouge-L-->
            <td> 39.2 </td> <!--NIKL summary Rouge-1-->
            <td> 17.95 </td> <!--NIKL summary Rouge-2-->
            <td> </td> <!--NIKL summary Rouge-L-->
            <td> 50.28 </td> <!--NIKL topic Rouge-1-->
            <td> 34.95 </td> <!--NIKL topic Rouge-2-->
            <td> </td> <!--NIKL topic Rouge-L-->
        </tr>
        <tr>
            <td>base</td>
            <td> 600K </td>
            <td> 39.3 </td> <!--CNN/DM Rouge-1-->
            <td> 17.94 </td> <!--CNN/DM Rouge-2-->
            <td> 36.66 </td> <!--CNN/DM Rouge-L-->
            <td> 37.93 </td> <!--NIKL summary Rouge-1-->
            <td> 17.74 </td> <!--NIKL summary Rouge-2-->
            <td> 36.36 </td> <!--NIKL summary Rouge-L-->
            <td> 48.71 </td> <!--NIKL topic Rouge-1-->
            <td> 33.85 </td> <!--NIKL topic Rouge-2-->
            <td> 48.15 </td> <!--NIKL topic Rouge-L-->
        </tr>
    </tbody>
</table>
</details>


## bibtex

KE-T5를 이용하여 연구를 진행하실 경우 아래와 같이 인용해주시길 바랍니다.
```
@inproceedings{kim-etal-2021-model-cross,
    title = "A Model of Cross-Lingual Knowledge-Grounded Response Generation for Open-Domain Dialogue Systems",
    author = "Kim, San  and
      Jang, Jin Yea  and
      Jung, Minyoung  and
      Shin, Saim",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-emnlp.33",
    doi = "10.18653/v1/2021.findings-emnlp.33",
    pages = "352--365",
    abstract = "Research on open-domain dialogue systems that allow free topics is challenging in the field of natural language processing (NLP). The performance of the dialogue system has been improved recently by the method utilizing dialogue-related knowledge; however, non-English dialogue systems suffer from reproducing the performance of English dialogue systems because securing knowledge in the same language with the dialogue system is relatively difficult. Through experiments with a Korean dialogue system, this paper proves that the performance of a non-English dialogue system can be improved by utilizing English knowledge, highlighting the system uses cross-lingual knowledge. For the experiments, we 1) constructed a Korean version of the Wizard of Wikipedia dataset, 2) built Korean-English T5 (KE-T5), a language model pre-trained with Korean and English corpus, and 3) developed a knowledge-grounded Korean dialogue model based on KE-T5. We observed the performance improvement in the open-domain Korean dialogue model even only English knowledge was given. The experimental results showed that the knowledge inherent in cross-lingual language models can be helpful for generating responses in open dialogue systems.",
} 
```
```
@misc{ke_t5,
    author       = {KETI AIRC},
    title        = {KE-T5: Korean English T5},
    month        = mar,
    year         = 2021,
    url          = {https://github.com/AIRC-KETI/ke-t5}
}
```

## Note

KE-T5는 **TFRC** 프로그램의 지원으로 학습되었습니다. \
KE-T5의 한국어 요약 학습에는 국립국어원의 **모두의 말뭉치-문서 요약 말뭉치**가 사용되었습니다. \
KE-T5의 ke, ko 사전학습에 국립국어원 **모두의 말뭉치**가 사용되었습니다.


## Acknowledgement

본 연구는 정부(과학기술정보통신부)의 재원으로 지원을 받아 수행된 연구입니다. (정보통신기획평가원, 2022-0-00320), (정보통신기획평가원, 2022-0-00608), (정보통신기획평가원, 2021-0-00537), (정보통신산업진흥원, S0316-21-1002)
