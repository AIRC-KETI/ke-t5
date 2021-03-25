# ke-t5: Korean-English T5

## Fine-tuning on downstream task

### Install pip packages
```bash
    git clone https://github.com/AIRC-KETI/ke-t5.git
    cd ke-t5
    pip3 install -r requirements.txt
```

### Run downstream tasks on your TPU
```bash
export TPU_NAME=your_tpu_name
export ZONE=your_project_zone
export TPU_SIZE=v3-8

ctpu up --name=$TPU_NAME --project=self-supervised-training --zone=$ZONE --tpu-size=$TPU_SIZE --tf-version=2.4.1 --tpu-only --noconf

export PROJECT=your_project_name
export BUCKET=gs://yourbucket/
export PRETRAINED_MODEL_DIR="${BUCKET}/your_pretrained_model_dir" # <-- put a checkpoint file of a pre-trained model in this directory.
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

### How to use a saved model (exported tensorflow model)

```python
# pip install tensorflow tensorflow_text
import numpy as np
import tensorflow as tf
import tensorflow_text

model_path = "path to exported model dir" # 'saved_model.pb','variable' should be in the directory.
loaded = tf.saved_model.load(model_path)
infer = loaded.signatures["serving_default"]

# We assume that the task of the model is 'ke_t5_nikl_summary_mixture_equal'.
# There are two types of summary actions in 'ke_t5_nikl_summary_mixture_equal'.
# These are "summarize_topic" and "summarize_summary".
# 'summarize_topic' selects a topic sentence from the input text.
# 'summarize_summary' generates a summary of the input text.

# source of input_str : https://news.naver.com/main/read.nhn?mode=LSD&mid=shm&sid1=102&oid=081&aid=0003173411
# press: 서울신문(www.seoul.co.kr)
# author: 이주원 기자 starjuwon@seoul.co.kr
input_str = """“처음에는 ‘금방 끝나겠지’라고 생각했는데 어느덧 100일이 됐네요. 그동안 춥고 아프고 힘들었지만 인간으로서 대우를 받을 수만 있다면 끝까지 버틸 수 있습니다.” LG트윈타워 청소 노동자들이 고용승계를 주장하며 파업에 나선지 100일째를 하루 앞둔 24일 서울 여의도 LG트윈타워 앞 ‘행복한 고용승계 텐트촌’에서 만난 박상설(63)씨는 힘들었던 투쟁 과정을 회상하며 눈시울을 붉혔다. 박씨는 2017년부터 LG트윈타워에서 청소 노동을 했지만 지난 1월 1일부로 계약이 종료돼 직장을 떠났다. 자동차 소음과 불편한 잠자리로 텐트에서 매일 밤잠을 설치지만 투쟁을 포기할 수 없다고 한다. 그는 “LG가 그동안 사회적 책임과 정도경영을 강조해 왔기에 파업이 금방 끝날 줄 알았다”며 “버티지 못하고 점점 떠나는 동지들을 바라볼 때마다 마음이 아프지만 정당한 노동 권리를 인정받기 위해 끝까지 투쟁할 것”이라고 강조했다. 지난해 11월 26일부터 파업에 돌입한 청소 노동자들은 25일 파업 100일째를 맞는다. 건물 1층 로비에서 시위를 하던 25명의 청소 노동자들은 지난 22일부터 정문 앞 도보에 텐트촌을 설치하고 장소를 옮겼다. 파업 100일에 맞춰 25일까지 시민연대와 함께 텐트 100개를 설치하고 주·야간 연대 시위를 이어가겠다는 뜻에서다. 노동자들은 한 명이 간신히 누울 수 있는 크기의 텐트 안에서 딱딱한 시멘트 바닥에 몸을 기대 쪽잠을 청하고 있다. LG트윈타워를 관리하는 LG그룹 계열사 ‘에스엔아이코퍼레이션’은 지난해 말 ‘지수아이앤씨’와 청소 용역 계약을 끝내고 다른 업체와 새로 계약했다. 사측은 ‘품질 저하’를 이유로 들었다. 반면 노동자들은 2019년 노조를 결성하고 권리를 주장하기 시작하면서 사측 눈 밖에 났다고 주장한다. 그동안 업체가 변경되더라도 기존 업체 노동자들이 새 업체에 고용승계가 되는 게 관례였지만 새 업체는 고용승계를 보장할 수 없다고 밝혔다. 지난달까지 고용노동부 중재로 수차례 노사 교섭이 있었지만 상황은 달라지지 않았다. 사측은 대신 노동자들에게 다른 사업장에서 일을 하게 해주겠다고 권유했다. 하지만 노동자들은 노조를 인정하지 않는 대기업의 행태를 묵인한 채 사측의 권유에 따른다면 어느 사업장에서 일을 하던 똑같은 행태가 반복될 수밖에 없다고 목소리를 높인다. 때문에 반드시 LG트윈타워에서 정당한 권리를 인정받고 노동을 이어가야만 한다고 말한다. 이들은 구광모 LG그룹 회장이 나서 문제를 해결해야 한다고 주장한다. 이혜정 LG트윈타워 공동대책위원회 집행위원은 “구 회장이 책임있는 답변을 내놓을 때까지 시민사회 단위와 함께 결의를 담아 끝까지 텐트촌을 유지할 것”이라고 강조했다."""

input_str_topic = "summarize_topic: " + input_str
input_str_summary = "summarize_summary: " + input_str

x = tf.constant([input_str_topic])

result = infer(x)
print([out.decode('utf-8') for out in result['inputs'].numpy()])
print([out.decode('utf-8') for out in result['outputs'].numpy()])

# summarize_topic
# 'LG트윈타워 청소 노동자가 고용승계를 주장하며 파업에 나선지 100일째를 하루 앞둔 24일 서울 여의도 LG트윈타워 앞 ‘행복한 고용승계 텐트촌’에서 만난 박상설(63)씨는 힘들었던 투쟁 과정을 회상하며 눈시울을 붉혔다. 반면 노동자들은 2019년 노조를 결성하고 권리를 주장하기 시작하면서 사측 눈 밖에 났다고 주장한다. 때문에 반드시 LG트윈타워에서 정당한 권리를 인정받고 노동을 이어가야 한다고 말한다.

# summarize_summary
# 'LG트윈타워 청소 노동자가 고용승계를 주장하며 파업에 나선지 100일째를 맞았다. LG트윈타워를 관리하는 LG그룹 계열사 ‘에스엔아이코퍼레이션’은 지난해 말 ‘지수아이앤씨’와 청소 용역 계약을 끝내고 다른 업체와 새로 계약했다. 그러나 노동자들은 노조를 인정하지 않는 대기업의 행태를 묵인한 채 사측의 권유에 따라 노동을 이어가라고 주장한다.'

```


## Datasets

### `ke`
**Dataset size**: `92.02GiB`\
**Corpus type**: `ko` (Newspaper, Written, Web text, Messenger, Spoken) `en` (Newspaper)\
**Split**\
'train': total **36,534,568** examples (ko: 22,734,730, en: 13,799,838)\
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


## models

### Pretrained models

<table>
    <thead>
        <tr>
            <th>Dataset</th>
            <th>Size</th>
            <th>steps</th>
            <th>Download URL(Tensorflow)</th>
            <th>Download URL(Pytorch)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=6>ke</td>
            <td rowspan=2>small</td>
            <td> 1M</td>
            <td>  <a href='https://drive.google.com/file/d/1RPq7zZWH0JfkA5Qq79KxEHgQ1-OmhaKD/view?usp=sharing'> Download </a>   </td>
            <td> - </td>
        </tr>
        <tr>
            <td> 3M </td>
            <td> TODO </td>
            <td> - </td>
        </tr>
        <tr>
            <td rowspan=2>base</td>
            <td> 600K </td>
            <td>  <a href='https://drive.google.com/file/d/1bkHRcf7gsEUbE0LeayBkxfE2o17wGn76/view?usp=sharing'> Download </a>  </td>
            <td> - </td>
        </tr>
        <tr>
            <td> 1M </td>
            <td> TODO </td>
            <td> - </td>
        </tr>
        <tr>
            <td rowspan=2>large</td>
            <td> 600K </td>
            <td>  <a href='https://drive.google.com/file/d/1lCoYijBQx3fh0gj1rKzvjpIU-To-SasQ/view?usp=sharing'> Download </a>  </td>
            <td> - </td>
        </tr>
        <tr>
            <td> 1M </td>
            <td> TODO </td>
            <td> - </td>
        </tr>
        <tr>
            <td rowspan=6>ke.newslike</td>
            <td rowspan=2>small</td>
            <td> 1M</td>
            <td>  <a href='https://drive.google.com/file/d/1OWvbRlTctQtrzk5iQpUJMprTWsObqtgD/view?usp=sharing'> Download </a>   </td>
            <td> - </td>
        </tr>
        <tr>
            <td> 3M </td>
            <td> TODO </td>
            <td> - </td>
        </tr>
        <tr>
            <td rowspan=2>base</td>
            <td> 600K </td>
            <td>  <a href='https://drive.google.com/file/d/1IqBOqNWYLkyPep84Aw7xBOL4X9W1x0bY/view?usp=sharing'> Download </a>  </td>
            <td> - </td>
        </tr>
        <tr>
            <td> 1M </td>
            <td> TODO </td>
            <td> - </td>
        </tr>
        <tr>
            <td rowspan=2>large</td>
            <td> 600K </td>
            <td>  <a href='https://drive.google.com/file/d/1_R9cQjUJkIC6GVLy1hOTUJQyldJTxlWo/view?usp=sharing'> Download </a>  </td>
            <td> - </td>
        </tr>
        <tr>
            <td> 1M </td>
            <td> TODO </td>
            <td> - </td>
        </tr>
        <tr>
            <td rowspan=6>ko</td>
            <td rowspan=2>small</td>
            <td> 1M</td>
            <td>  <a href='https://drive.google.com/file/d/1V3usXySS7JUnFSACvvR8kN-XBN5C6cWF/view?usp=sharing'> Download </a>   </td>
            <td> - </td>
        </tr>
        <tr>
            <td> 3M </td>
            <td> TODO </td>
            <td> - </td>
        </tr>
        <tr>
            <td rowspan=2>base</td>
            <td> 600K </td>
            <td>  <a href='https://drive.google.com/file/d/1LCvR_rC_cmouTy1720z48BC9Gkmg3EyK/view?usp=sharing'> Download </a>  </td>
            <td> - </td>
        </tr>
        <tr>
            <td> 1M </td>
            <td> TODO </td>
            <td> - </td>
        </tr>
    </tbody>
</table>



### Downstream models (Tensorflow saved model)

<table>
    <thead>
        <tr>
            <th>Dataset</th>
            <th>Size</th>
            <th>steps</th>
            <th>ke_t5_nikl_summary_mixture_equal</th>
            <th>ke_t5_ko_en_qa_equal</th>
            <th>ke_t5_ko_translation_equal</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=3>ke</td>
            <td>small</td>
            <td> 1M </td>
            <td> - </td>
            <td> - </td>
            <td> - </td>
        </tr>
        <tr>
            <td>base</td>
            <td> 600K </td>
            <td> - </td>
            <td> - </td>
            <td> - </td>
        </tr>
        <tr>
            <td>large</td>
            <td> 600K </td>
            <td> - </td>
            <td> - </td>
            <td> - </td>
        </tr>
        <tr>
            <td rowspan=3>ke.newslike</td>
            <td>small</td>
            <td> 1M </td>
            <td> - </td>
            <td> - </td>
            <td> - </td>
        </tr>
        <tr>
            <td>base</td>
            <td> 600K </td>
            <td> - </td>
            <td> - </td>
            <td> - </td>
        </tr>
        <tr>
            <td>large</td>
            <td> 600K </td>
            <td> - </td>
            <td> - </td>
            <td> - </td>
        </tr>
        <tr>
            <td rowspan=3>ko</td>
            <td>small</td>
            <td> 1M </td>
            <td> - </td>
            <td> - </td>
            <td> - </td>
        </tr>
        <tr>
            <td>base</td>
            <td> 600K </td>
            <td> <a href='https://drive.google.com/file/d/1OpjLxr2tyLi_B_iC2hDyw8PjbB-KxNlI/view?usp=sharing'> Download </a> </td>
            <td> - </td>
            <td> - </td>
        </tr>
    </tbody>
</table>


## Downstream Tasks

### Tasks

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
| ke_t5_ted_multi_en_ko  | `translate en to ko: {text}`  | `{text}`  | Translation  | ko, en |
| ke_t5_ted_multi_ko_en  | `translate ko to en: {text}`  | `{text}`  | Translation  | ko, en |

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



### Mixture Tasks


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


## Samples
### Model Spec
Dataset: `ke.newslike`\
Model Size: `small`\
steps: `1M`

### Summarization

#### NIKL Topic sentences

<details>
<summary>Summarization (NIKL topic) (click)</summary>

`Input`
```
summarize_topic: "온난화가 파미르 고원까지 덮쳤어요" 첫 주한 타지키스탄 대사 살로히딘, 산사태 상영회... "모두 각성해야" 지난 12일 서울 광희동에서 열린 타지키스탄 독립 24주년 기념행사에서 파미르 고원의 산사태 장면을 담은 동영상이 상영됐다. 이 행사를 마련한 키로모프 살로히딘 주한 타지키스탄 대사는 "한국인에게는 아직 낯선 타지키스탄이 '세계의 지붕' 파미르 고원을 품고 있는 나라라는 것, 그리고 지구온난화로 인해 파미르 고원이 아픔을 겪고 있다는 것을 알리고 싶었다"고 했다. 영상에는 협곡에서 시커먼 흙더미가 떠밀려오는 모습, 산 아래 밭과 가옥들이 순식간에 묻혀버리는 모습 등이 담겼다. 지구온난화의 영향으로 산 정상 부근의 얼음이 급속히 녹아든 것이 원인이다. 평균 고도 5000m에 이르는 파미르 고원은 히말라야와 힌두쿠시·톈산 산맥 등을 품고 있다. 지난 18일 서울 충무로 한국·중앙아시아 친선협회 사무실에서 만난 살로히딘 대사는 "국토의 93%가 산지인 우리나라는 잠재력이 무궁무진하다"며 '국가 세일즈'에 나섰다. "국민 대다수가 '산사람'이죠. 산에서 보석과 광물을 캐고, 수력으로 발전기를 돌립니다. 자원은 부족해도 선진 경제를 이룩한 한국이 우리의 성장 모델입니다. 앞으로 윈·윈 관계를 이룰 수 있을 겁니다." 파미르 고원 지역을 관할하는 바다흐 ⁇ 주(州)의 리요에프 누랄리 부주지사도 살로히딘 대사와 함께 만났다. 누랄리 부주지사는 새마을중앙회 초청 외국 공무원 행정 연수에 참여하기 위해 한국에 왔다. 두 사람은 "타지키스탄에서 산은 삶 그 자체"라고 했다. 누랄리 부주지사는 "이번 산사태로 학교와 병원, 도로 같은 기반 시설이 크게 파괴됐다"며 "아픔을 겪고 일어날 수 있도록 응원을 부탁한다. 그리고 지구온난화의 결과가 얼마나 무서운지 모두가 새롭게 경각심을 가졌으면 좋겠다"고 했다. 1991년 구소련의 붕괴로 독립한 타지키스탄은 다른 중앙아시아 국가처럼 한국을 '경제 발전의 교과서'로 삼고 있다. 지난 4월에는 에모말리 라흐몬 타지키스탄 대통령의 방한에 맞춰 서울에 대사관도 열었다.
```

\
`Target`
```
지난 12일 서울 광희동에서 열린 타지키스탄 독립 24주년 기념행사에서 파미르 고원의 산사태 장면을 담은 동영상이 상영됐다. 지난 18일 서울 충무로 한국·중앙아시아 친선협회 사무실에서 만난 살로히딘 대사는 "국토의 93%가 산지인 우리나라는 잠재력이 무궁무진하다"며 '국가 세일즈'에 나섰다. 1991년 구소련의 붕괴로 독립한 타지키스탄은 다른 중앙아시아 국가처럼 한국을 '경제 발전의 교과서'로 삼고 있다.
```

\
`Prediction`
```
지난 12일 서울 광희동에서 열린 타지키스탄 독립 24주년 기념행사에서 파미르 고원의 산사태 장면을 담은 동영상이 상영됐다. 이 행사를 마련한 키로모프 살로히딘 타지키스탄 대사는 "국토의 93%가 산지인 우리나라는 잠재력이 무궁무진하다"며 '국가 세일즈'에 나섰다. 1991년 구소련의 붕괴로 독립한 타지키스탄은 다른 중앙아시아 국가처럼 한국을 '경제 발전의 교과서'로 삼고 있다.
```

</details>


#### NIKL Summary

<details>
<summary>Summarization (NIKL summary) (click)</summary>

`Input`

```
summarize_summary: ‘스페셜 원’, 그도 레알에선 평범했다. 모리뉴 감독 ‘시즌 뒤 사퇴’ 발표. 바르사에 밀려 라리가서 부진. 공언했던 챔스리그 우승 못해. 카시야스 등 선수들과 불화도. 새 감독엔 안첼로티·지단 거론. 그의 사전에 실패란 없는 듯했다. 가는 팀마다 승승장구했고, 우승트로피가 쌓여만 갔다. 2008년부터 2010년까지 이탈리아 세리에A 명문 인터밀란 사령탑으로 있으면서 ‘트레블’ 위업까지 달성했다. 스스로 ‘스페셜 원’이라고 했고, 사람들은 그를 앨릭스 퍼거슨 맨체스터 유나이티드 감독에 버금가는 명장 반열에 올려놨다. 그런 명성을 발판으로 스페인의 명문 레알 마드리드로 화려하게 이적했다. 그러나 레알 지휘봉을 잡고서는 그런 명성에 금이 가기 시작했다. 크리스티아누 호날두라는 걸출한 스타가 팀에 버티고 있었지만, 그보다 더 센 리오넬 메시가 포진한 FC바르셀로나의 기세에 밀려 라 리가와 코파 델 레이(스페인국왕컵), 스페인 수페르코파(슈퍼컵)에서 한번씩 우승하는 것에 그쳤다. 그가 취임하면서 공언했던 챔피언스리그 타이틀은 한번도 차지하지 못했다. 바르사와의 엘 클라시코에서도 열세를 면치 못했다. 조제 모리뉴(50) 레알 마드리드 감독이 결국 2012~2013 시즌 뒤 팀을 떠나게 됐다. 자신이 설정한 목표를 달성하지 못한 불명예 퇴진이나 마찬가지인 셈이다. 미국 <이에스피엔>(ESPN)은 이에 대해 그레이엄 헌터의 기고를 통해 “캡틴 카시야스와의 갈등을 불필요하게 피하지 못한 것은 모리뉴의 레알 통치 기간 볼썽사나운 상처로 남게 됐다. 카시야스의 32번째 생일에 이뤄진 모리뉴의 감독 사퇴는 카시야스에게 승리를 안겨준 것”이라고 평가했다. 플로렌티노 페레스 레알 회장은 20일 저녁 무렵(현지시각) 홈경기장인 산티아고 베르나베우에서 연 기자회견을 통해 “모리뉴 감독이 이번 시즌을 끝으로 팀을 떠나기로 했다. 구단과 모리뉴 감독의 합의에 의한 것”이라고 공식 발표했다. 페레스 회장은 “모리뉴 감독이 계속 팀을 이끌길 바랐지만 모리뉴 감독 스스로 레알을 최고의 팀으로 만들고자 하는 압박감이 심했다”고 설명했다. 그는 “모리뉴 감독이 ‘영국에선 축구 경기가 90분으로 끝나지만 스페인에선 한 경기가 일주일, 매일 24시간으로 연장된다’고 얘기했다”며 모리뉴 감독의 스트레스가 적지 않았다고 덧붙였다. 모리뉴 감독은 지난해 계약 기간을 2016년까지로 연장했지만 결국 성적 부진에다 카시야스 등 일부 선수들과의 불화설까지 겹쳐 중도하차하게 됐다. 페레스 회장은 “레알이 이전엔
```

\
`Target`
```
레알 마드리드의 조제 모리뉴 감독이 2012-2013 시즌을 끝으로 팀을 떠난다. 모리뉴 감독은 성적 부진에 일부 선수들과의 불화설이 겹치면서 중도하차하게 됐다. 팀의 차기 사령탑으로는 카를로 안첼로티 감독이 거론되고 있다.
```

\
`Prediction`
```
조제 모리뉴 레알 마드리드 감독이 시즌 뒤 팀을 떠나게 돼 자신의 목표를 달성하지 못한 불명예 퇴진이나 마찬가지다. 모리뉴 감독은 지난해 계약 기간을 2016년까지로 연장했지만 결국 성적 부진에다 일부 선수들과의 불화설까지 겹쳐 중도하차하게 됐다. 레알 차기 사령탑으로는 카를로 안첼로티(54) 파리 생제르맹(프랑스) 감독이 거론되고 있다.
```

</details>


#### CNN/DM

<details>
<summary>Summarization (CNN/DM) (click)</summary>

`Input`
```
summarize: Dougie Freedman is on the verge of agreeing a new two-year deal to remain at Nottingham Forest. Freedman has stabilised Forest since he replaced cult hero Stuart Pearce and the club's owners are pleased with the job he has done at the City Ground. Dougie Freedman is set to sign a new deal at Nottingham Forest. Freedman has impressed at the City Ground since replacing Stuart Pearce in February. They made an audacious attempt on the play-off places when Freedman replaced Pearce but have tailed off in recent weeks. That has not prevented Forest's ownership making moves to secure Freedman on a contract for the next two seasons.
```

\
`Target`
```
Nottingham Forest are close to extending Dougie Freedman's contract . The Forest boss took over from former manager Stuart Pearce in February . Freedman has since lead the club to ninth in the Championship .
```

\
`Prediction`
```
Dougie Freedman set to sign a two-year deal at Nottingham Forest . Freedman replaced Stuart Pearce in February . Forest have made an audacious bid on the play-off places .
```

</details>


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


## Performance

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
            <td></td>
            <td> </td> <!--NIKL CoLA Mattew's-->
            <td> </td> <!--NSMC Accuracy-->
            <td> </td> <!--Question-pair F1-->
            <td> </td> <!--Question-pair Accuracy-->
            <td> </td> <!--KorNLI Accuracy-->
            <td> </td> <!--KorSTS Pearson-->
            <td> </td> <!--KorSTS Spearman-->
            <td> </td> <!--KHSD Accuracy-->
        </tr>
        <tr>
            <td>base</td>
            <td></td>
            <td> </td> <!--NIKL CoLA Mattew's-->
            <td> </td> <!--NSMC Accuracy-->
            <td> </td> <!--Question-pair F1-->
            <td> </td> <!--Question-pair Accuracy-->
            <td> </td> <!--KorNLI Accuracy-->
            <td> </td> <!--KorSTS Pearson-->
            <td> </td> <!--KorSTS Spearman-->
            <td> </td> <!--KHSD Accuracy-->
        </tr>
        <tr>
            <td>large</td>
            <td></td>
            <td> </td> <!--NIKL CoLA Mattew's-->
            <td> </td> <!--NSMC Accuracy-->
            <td> </td> <!--Question-pair F1-->
            <td> </td> <!--Question-pair Accuracy-->
            <td> </td> <!--KorNLI Accuracy-->
            <td> </td> <!--KorSTS Pearson-->
            <td> </td> <!--KorSTS Spearman-->
            <td> </td> <!--KHSD Accuracy-->
        </tr>
        <tr>
            <td rowspan=3>ke.newslike</td>
            <td>small</td>
            <td> 1M </td>
            <td> 18.78 </td> <!--NIKL CoLA Mattew's-->
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
            <td rowspan=3>ko</td>
            <td>small</td>
            <td></td>
            <td> </td> <!--NIKL CoLA Mattew's-->
            <td> </td> <!--NSMC Accuracy-->
            <td> </td> <!--Question-pair F1-->
            <td> </td> <!--Question-pair Accuracy-->
            <td> </td> <!--KorNLI Accuracy-->
            <td> </td> <!--KorSTS Pearson-->
            <td> </td> <!--KorSTS Spearman-->
            <td> </td> <!--KHSD Accuracy-->
        </tr>
        <tr>
            <td>base</td>
            <td></td>
            <td> </td> <!--NIKL CoLA Mattew's-->
            <td> </td> <!--NSMC Accuracy-->
            <td> </td> <!--Question-pair F1-->
            <td> </td> <!--Question-pair Accuracy-->
            <td> </td> <!--KorNLI Accuracy-->
            <td> </td> <!--KorSTS Pearson-->
            <td> </td> <!--KorSTS Spearman-->
            <td> </td> <!--KHSD Accuracy-->
        </tr>
        <tr>
            <td>large</td>
            <td></td>
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
            <td></td>
            <td> </td> <!--GLUE average-->
            <td> </td> <!--CoLA Mattew's-->
            <td> </td> <!--SST-2 Accuracy-->
            <td> </td> <!--MRPC F1-->
            <td> </td> <!--MRPC Accuracy-->
            <td> </td> <!--STS-B Pearson-->
            <td> </td> <!--STS-B Spearman-->
        </tr>
        <tr>
            <td>base</td>
            <td></td>
            <td> </td> <!--GLUE average-->
            <td> </td> <!--CoLA Mattew's-->
            <td> </td> <!--SST-2 Accuracy-->
            <td> </td> <!--MRPC F1-->
            <td> </td> <!--MRPC Accuracy-->
            <td> </td> <!--STS-B Pearson-->
            <td> </td> <!--STS-B Spearman-->
        </tr>
        <tr>
            <td>large</td>
            <td></td>
            <td> </td> <!--GLUE average-->
            <td> </td> <!--CoLA Mattew's-->
            <td> </td> <!--SST-2 Accuracy-->
            <td> </td> <!--MRPC F1-->
            <td> </td> <!--MRPC Accuracy-->
            <td> </td> <!--STS-B Pearson-->
            <td> </td> <!--STS-B Spearman-->
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
            <td rowspan=3>ko</td>
            <td>small</td>
            <td></td>
            <td> </td> <!--GLUE average-->
            <td> </td> <!--CoLA Mattew's-->
            <td> </td> <!--SST-2 Accuracy-->
            <td> </td> <!--MRPC F1-->
            <td> </td> <!--MRPC Accuracy-->
            <td> </td> <!--STS-B Pearson-->
            <td> </td> <!--STS-B Spearman-->
        </tr>
        <tr>
            <td>base</td>
            <td></td>
            <td> </td> <!--GLUE average-->
            <td> </td> <!--CoLA Mattew's-->
            <td> </td> <!--SST-2 Accuracy-->
            <td> </td> <!--MRPC F1-->
            <td> </td> <!--MRPC Accuracy-->
            <td> </td> <!--STS-B Pearson-->
            <td> </td> <!--STS-B Spearman-->
        </tr>
        <tr>
            <td>large</td>
            <td></td>
            <td> </td> <!--GLUE average-->
            <td> </td> <!--CoLA Mattew's-->
            <td> </td> <!--SST-2 Accuracy-->
            <td> </td> <!--MRPC F1-->
            <td> </td> <!--MRPC Accuracy-->
            <td> </td> <!--STS-B Pearson-->
            <td> </td> <!--STS-B Spearman-->
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
            <td></td>
            <td> </td> <!--QQP F1-->
            <td> </td> <!--QQP Accuracy-->
            <td> </td> <!--MNLI-m Accuracy-->
            <td> </td> <!--MNLI-mm Accuracy-->
            <td> </td> <!--QNLI Accuracy-->
            <td> </td> <!--RTE Accuracy-->
            <td> </td> <!--WNLI Accuracy-->
        </tr>
        <tr>
            <td>base</td>
            <td></td>
            <td> </td> <!--QQP F1-->
            <td> </td> <!--QQP Accuracy-->
            <td> </td> <!--MNLI-m Accuracy-->
            <td> </td> <!--MNLI-mm Accuracy-->
            <td> </td> <!--QNLI Accuracy-->
            <td> </td> <!--RTE Accuracy-->
            <td> </td> <!--WNLI Accuracy-->
        </tr>
        <tr>
            <td>large</td>
            <td></td>
            <td> </td> <!--QQP F1-->
            <td> </td> <!--QQP Accuracy-->
            <td> </td> <!--MNLI-m Accuracy-->
            <td> </td> <!--MNLI-mm Accuracy-->
            <td> </td> <!--QNLI Accuracy-->
            <td> </td> <!--RTE Accuracy-->
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
            <td></td>
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
            <td></td>
            <td> </td> <!--QQP F1-->
            <td> </td> <!--QQP Accuracy-->
            <td> </td> <!--MNLI-m Accuracy-->
            <td> </td> <!--MNLI-mm Accuracy-->
            <td> </td> <!--QNLI Accuracy-->
            <td> </td> <!--RTE Accuracy-->
            <td> </td> <!--WNLI Accuracy-->
        </tr>
        <tr>
            <td>small</td>
            <td></td>
            <td> </td> <!--QQP F1-->
            <td> </td> <!--QQP Accuracy-->
            <td> </td> <!--MNLI-m Accuracy-->
            <td> </td> <!--MNLI-mm Accuracy-->
            <td> </td> <!--QNLI Accuracy-->
            <td> </td> <!--RTE Accuracy-->
            <td> </td> <!--WNLI Accuracy-->
        </tr>
        <tr>
            <td>large</td>
            <td></td>
            <td> </td> <!--QQP F1-->
            <td> </td> <!--QQP Accuracy-->
            <td> </td> <!--MNLI-m Accuracy-->
            <td> </td> <!--MNLI-mm Accuracy-->
            <td> </td> <!--QNLI Accuracy-->
            <td> </td> <!--RTE Accuracy-->
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
            <td></td>
            <td> </td> <!--Super GLUE Average-->
            <td> </td> <!--BoolQ Accuracy-->
            <td> </td> <!--CB F1-->
            <td> </td> <!--CB Accuracy-->
            <td> </td> <!--COPA Accuracy-->
            <td> </td> <!--MultiRC F1a-->
            <td> </td> <!--MultiRC EM-->
            <td> </td> <!--ReCoRD F1-->
            <td> </td> <!--ReCoRD Accuracy-->
            <td> </td> <!--RTE Accuracy-->
            <td> </td> <!--WiC Accuracy-->
            <td> </td> <!--WSC Accuracy-->
        </tr>
        <tr>
            <td>base</td>
            <td></td>
            <td> </td> <!--Super GLUE Average-->
            <td> </td> <!--BoolQ Accuracy-->
            <td> </td> <!--CB F1-->
            <td> </td> <!--CB Accuracy-->
            <td> </td> <!--COPA Accuracy-->
            <td> </td> <!--MultiRC F1a-->
            <td> </td> <!--MultiRC EM-->
            <td> </td> <!--ReCoRD F1-->
            <td> </td> <!--ReCoRD Accuracy-->
            <td> </td> <!--RTE Accuracy-->
            <td> </td> <!--WiC Accuracy-->
            <td> </td> <!--WSC Accuracy-->
        </tr>
        <tr>
            <td>large</td>
            <td></td>
            <td> </td> <!--Super GLUE Average-->
            <td> </td> <!--BoolQ Accuracy-->
            <td> </td> <!--CB F1-->
            <td> </td> <!--CB Accuracy-->
            <td> </td> <!--COPA Accuracy-->
            <td> </td> <!--MultiRC F1a-->
            <td> </td> <!--MultiRC EM-->
            <td> </td> <!--ReCoRD F1-->
            <td> </td> <!--ReCoRD Accuracy-->
            <td> </td> <!--RTE Accuracy-->
            <td> </td> <!--WiC Accuracy-->
            <td> </td> <!--WSC Accuracy-->
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
            <td></td>
            <td> </td> <!--Super GLUE Average-->
            <td> </td> <!--BoolQ Accuracy-->
            <td> </td> <!--CB F1-->
            <td> </td> <!--CB Accuracy-->
            <td> </td> <!--COPA Accuracy-->
            <td> </td> <!--MultiRC F1a-->
            <td> </td> <!--MultiRC EM-->
            <td> </td> <!--ReCoRD F1-->
            <td> </td> <!--ReCoRD Accuracy-->
            <td> </td> <!--RTE Accuracy-->
            <td> </td> <!--WiC Accuracy-->
            <td> </td> <!--WSC Accuracy-->
        </tr>
        <tr>
            <td>base</td>
            <td></td>
            <td> </td> <!--Super GLUE Average-->
            <td> </td> <!--BoolQ Accuracy-->
            <td> </td> <!--CB F1-->
            <td> </td> <!--CB Accuracy-->
            <td> </td> <!--COPA Accuracy-->
            <td> </td> <!--MultiRC F1a-->
            <td> </td> <!--MultiRC EM-->
            <td> </td> <!--ReCoRD F1-->
            <td> </td> <!--ReCoRD Accuracy-->
            <td> </td> <!--RTE Accuracy-->
            <td> </td> <!--WiC Accuracy-->
            <td> </td> <!--WSC Accuracy-->
        </tr>
        <tr>
            <td>large</td>
            <td></td>
            <td> </td> <!--Super GLUE Average-->
            <td> </td> <!--BoolQ Accuracy-->
            <td> </td> <!--CB F1-->
            <td> </td> <!--CB Accuracy-->
            <td> </td> <!--COPA Accuracy-->
            <td> </td> <!--MultiRC F1a-->
            <td> </td> <!--MultiRC EM-->
            <td> </td> <!--ReCoRD F1-->
            <td> </td> <!--ReCoRD Accuracy-->
            <td> </td> <!--RTE Accuracy-->
            <td> </td> <!--WiC Accuracy-->
            <td> </td> <!--WSC Accuracy-->
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
            <td></td>
            <td> </td> <!--SQuAD EM-->
            <td> </td> <!--SQuAD F1-->
            <td> </td> <!--KorQuAD 1.1 EM-->
            <td> </td> <!--KorQuAD 1.1 F1-->
        </tr>
        <tr>
            <td>base</td>
            <td></td>
            <td> </td> <!--SQuAD EM-->
            <td> </td> <!--SQuAD F1-->
            <td> </td> <!--KorQuAD 1.1 EM-->
            <td> </td> <!--KorQuAD 1.1 F1-->
        </tr>
        <tr>
            <td>large</td>
            <td></td>
            <td> </td> <!--SQuAD EM-->
            <td> </td> <!--SQuAD F1-->
            <td> </td> <!--KorQuAD 1.1 EM-->
            <td> </td> <!--KorQuAD 1.1 F1-->
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
            <td></td>
            <td> </td> <!--SQuAD EM-->
            <td> </td> <!--SQuAD F1-->
            <td> </td> <!--KorQuAD 1.1 EM-->
            <td> </td> <!--KorQuAD 1.1 F1-->
        </tr>
        <tr>
            <td>base</td>
            <td></td>
            <td> </td> <!--SQuAD EM-->
            <td> </td> <!--SQuAD F1-->
            <td> </td> <!--KorQuAD 1.1 EM-->
            <td> </td> <!--KorQuAD 1.1 F1-->
        </tr>
        <tr>
            <td>large</td>
            <td></td>
            <td> </td> <!--SQuAD EM-->
            <td> </td> <!--SQuAD F1-->
            <td> </td> <!--KorQuAD 1.1 EM-->
            <td> </td> <!--KorQuAD 1.1 F1-->
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
            <td></td>
            <td> </td> <!--en->ko Rouge-1-->
            <td> </td> <!--en->ko Rouge-2-->
            <td> </td> <!--en->ko Rouge-L-->
            <td> </td> <!--ko->en Rouge-1-->
            <td> </td> <!--ko->en Rouge-2-->
            <td> </td> <!--ko->en Rouge-L-->
        </tr>
        <tr>
            <td>base</td>
            <td></td>
            <td> </td> <!--en->ko Rouge-1-->
            <td> </td> <!--en->ko Rouge-2-->
            <td> </td> <!--en->ko Rouge-L-->
            <td> </td> <!--ko->en Rouge-1-->
            <td> </td> <!--ko->en Rouge-2-->
            <td> </td> <!--ko->en Rouge-L-->
        </tr>
        <tr>
            <td>large</td>
            <td></td>
            <td> </td> <!--en->ko Rouge-1-->
            <td> </td> <!--en->ko Rouge-2-->
            <td> </td> <!--en->ko Rouge-L-->
            <td> </td> <!--ko->en Rouge-1-->
            <td> </td> <!--ko->en Rouge-2-->
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
            <td></td>
            <td> </td> <!--en->ko Rouge-1-->
            <td> </td> <!--en->ko Rouge-2-->
            <td> </td> <!--en->ko Rouge-L-->
            <td> </td> <!--ko->en Rouge-1-->
            <td> </td> <!--ko->en Rouge-2-->
            <td> </td> <!--ko->en Rouge-L-->
        </tr>
        <tr>
            <td rowspan=3>ko</td>
            <td>small</td>
            <td></td>
            <td> </td> <!--en->ko Rouge-1-->
            <td> </td> <!--en->ko Rouge-2-->
            <td> </td> <!--en->ko Rouge-L-->
            <td> </td> <!--ko->en Rouge-1-->
            <td> </td> <!--ko->en Rouge-2-->
            <td> </td> <!--ko->en Rouge-L-->
        </tr>
        <tr>
            <td>base</td>
            <td></td>
            <td> </td> <!--en->ko Rouge-1-->
            <td> </td> <!--en->ko Rouge-2-->
            <td> </td> <!--en->ko Rouge-L-->
            <td> </td> <!--ko->en Rouge-1-->
            <td> </td> <!--ko->en Rouge-2-->
            <td> </td> <!--ko->en Rouge-L-->
        </tr>
        <tr>
            <td>large</td>
            <td></td>
            <td> </td> <!--en->ko Rouge-1-->
            <td> </td> <!--en->ko Rouge-2-->
            <td> </td> <!--en->ko Rouge-L-->
            <td> </td> <!--ko->en Rouge-1-->
            <td> </td> <!--ko->en Rouge-2-->
            <td> </td> <!--ko->en Rouge-L--> 
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
            <td></td>
            <td> </td> <!--CNN/DM Rouge-1-->
            <td> </td> <!--CNN/DM Rouge-2-->
            <td> </td> <!--CNN/DM Rouge-L-->
            <td> </td> <!--NIKL summary Rouge-1-->
            <td> </td> <!--NIKL summary Rouge-2-->
            <td> </td> <!--NIKL summary Rouge-L-->
            <td> </td> <!--NIKL topic Rouge-1-->
            <td> </td> <!--NIKL topic Rouge-2-->
            <td> </td> <!--NIKL topic Rouge-L-->
        </tr>
        <tr>
            <td>base</td>
            <td></td>
            <td> </td> <!--CNN/DM Rouge-1-->
            <td> </td> <!--CNN/DM Rouge-2-->
            <td> </td> <!--CNN/DM Rouge-L-->
            <td> </td> <!--NIKL summary Rouge-1-->
            <td> </td> <!--NIKL summary Rouge-2-->
            <td> </td> <!--NIKL summary Rouge-L-->
            <td> </td> <!--NIKL topic Rouge-1-->
            <td> </td> <!--NIKL topic Rouge-2-->
            <td> </td> <!--NIKL topic Rouge-L-->
        </tr>
        <tr>
            <td>large</td>
            <td></td>
            <td> </td> <!--CNN/DM Rouge-1-->
            <td> </td> <!--CNN/DM Rouge-2-->
            <td> </td> <!--CNN/DM Rouge-L-->
            <td> </td> <!--NIKL summary Rouge-1-->
            <td> </td> <!--NIKL summary Rouge-2-->
            <td> </td> <!--NIKL summary Rouge-L-->
            <td> </td> <!--NIKL topic Rouge-1-->
            <td> </td> <!--NIKL topic Rouge-2-->
            <td> </td> <!--NIKL topic Rouge-L-->
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
            <td> </td> <!--CNN/DM Rouge-1-->
            <td> </td> <!--CNN/DM Rouge-2-->
            <td> </td> <!--CNN/DM Rouge-L-->
            <td> </td> <!--NIKL summary Rouge-1-->
            <td> </td> <!--NIKL summary Rouge-2-->
            <td> </td> <!--NIKL summary Rouge-L-->
            <td> </td> <!--NIKL topic Rouge-1-->
            <td> </td> <!--NIKL topic Rouge-2-->
            <td> </td> <!--NIKL topic Rouge-L-->
        </tr>
        <tr>
            <td>large</td>
            <td> 600K </td>
            <td> </td> <!--CNN/DM Rouge-1-->
            <td> </td> <!--CNN/DM Rouge-2-->
            <td> </td> <!--CNN/DM Rouge-L-->
            <td> </td> <!--NIKL summary Rouge-1-->
            <td> </td> <!--NIKL summary Rouge-2-->
            <td> </td> <!--NIKL summary Rouge-L-->
            <td> </td> <!--NIKL topic Rouge-1-->
            <td> </td> <!--NIKL topic Rouge-2-->
            <td> </td> <!--NIKL topic Rouge-L-->
        </tr>
        <tr>
            <td rowspan=3>ko</td>
            <td>small</td>
            <td></td>
            <td> </td> <!--CNN/DM Rouge-1-->
            <td> </td> <!--CNN/DM Rouge-2-->
            <td> </td> <!--CNN/DM Rouge-L-->
            <td> </td> <!--NIKL summary Rouge-1-->
            <td> </td> <!--NIKL summary Rouge-2-->
            <td> </td> <!--NIKL summary Rouge-L-->
            <td> </td> <!--NIKL topic Rouge-1-->
            <td> </td> <!--NIKL topic Rouge-2-->
            <td> </td> <!--NIKL topic Rouge-L-->
        </tr>
        <tr>
            <td>base</td>
            <td></td>
            <td> </td> <!--CNN/DM Rouge-1-->
            <td> </td> <!--CNN/DM Rouge-2-->
            <td> </td> <!--CNN/DM Rouge-L-->
            <td> </td> <!--NIKL summary Rouge-1-->
            <td> </td> <!--NIKL summary Rouge-2-->
            <td> </td> <!--NIKL summary Rouge-L-->
            <td> </td> <!--NIKL topic Rouge-1-->
            <td> </td> <!--NIKL topic Rouge-2-->
            <td> </td> <!--NIKL topic Rouge-L-->
        </tr>
        <tr>
            <td>large</td>
            <td></td>
            <td> </td> <!--CNN/DM Rouge-1-->
            <td> </td> <!--CNN/DM Rouge-2-->
            <td> </td> <!--CNN/DM Rouge-L-->
            <td> </td> <!--NIKL summary Rouge-1-->
            <td> </td> <!--NIKL summary Rouge-2-->
            <td> </td> <!--NIKL summary Rouge-L-->
            <td> </td> <!--NIKL topic Rouge-1-->
            <td> </td> <!--NIKL topic Rouge-2-->
            <td> </td> <!--NIKL topic Rouge-L-->
        </tr>
    </tbody>
</table>
</details>


## bibtex
```
@misc{ke_t5,
    author       = {KETI AIRC},
    title        = {{Korean English T5}},
    month        = mar,
    year         = 2021,
    version      = {1.0},
    url          = {https://github.com/AIRC-KETI/ke-t5}
}
```

## Note

**TFRC-supported**
