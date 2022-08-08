
## 필요 패키시 설치

```bash
pip install -r requirements.txt
# pip3 install -r requirements.txt
```

## Manual Data Download

AI Hub에서 데이터를 다운로드합니다.

[요약문 및 레포트 생성데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=582)

[일상생활 및 구어체 한-영 번역 병렬 말뭉치 데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=582)

여기서 다운받은 데이터들이 위치해 있는 폴더가 `--hf_data_dir`의 위치가 됩니다.
`--hf_cache_dir`은 데이터 cache가 저장될 위치입니다.


## Huggingface Accelerate Configuration

학습하시는 형태에 따라 알맞게 설정하시면 될 것 같습니다.
저 같은 경우는 single node, 8-GPU로 세팅했습니다.

```bash
$ accelerate config
In which compute environment are you running? ([0] This machine, [1] AWS (Amazon SageMaker)): 0
Which type of machine are you using? ([0] No distributed training, [1] multi-CPU, [2] multi-GPU, [3] TPU): 2
How many different machines will you use (use more than 1 for multi-node training)? [1]: 1
Do you want to use DeepSpeed? [yes/NO]: NO
Do you want to use FullyShardedDataParallel? [yes/NO]: NO
How many GPU(s) should be used for distributed training? [1]:8
Do you wish to use FP16 or BF16 (mixed precision)? [NO/fp16/bf16]: NO
```

## Huggingface Generate keyword arguments

저같은 경우는 `early_stopping`, `length_penalty`, `no_repeat_ngram_size`를 조정합니다.

## Generative Summarization 학습

```bash
epochs=10
scheduler_type=constant
# generative summary
accelerate launch run_summarization_no_trainer.py \
--dataset_name summary_and_report.py \
--dataset_config_name base \
--hf_cache_dir huggingface_datasets \
--hf_data_dir data \
--ignore_verifications \
--text_column 'passage' \
--summary_column 'generative_summary' \
--model_name_or_path 'KETI-AIR/ke-t5-small' \
--source_prefix 'summarize: ' \
--max_source_length 512 \
--preprocessing_num_workers 64 \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 32 \
--lr_scheduler_type $scheduler_type \
--val_max_target_length 200 \
--learning_rate 0.001 \
--num_train_epochs $epochs \
--num_beams 4 \
--eval_on_last_epoch \
--output_dir outputs/ket5_generative_summary_beam_$scheduler_type-e$epochs
```

### Quantitive Performance

| Model | R-1  | R-2 | R-L (LCS:Longest Common Subsequence) | R-LSUM (LCS summary-level) |
| -------------| ------------- | ------------- | ------------- | ------------- |
| ke-t5-small | 12.8004 | 2.617 | 12.6944 | 12.6726 |


### Qualitive Performance

```python
{
'inputs': 'summarize: 문재인 대통령은 2일 “새해에는 더욱 ‘확실한 변화’를 만들어내겠다”며 “권력기관 개혁과 공정사회 개혁이 그 시작”이라고 말했다. 문 대통령은 이날 오전 11시 서울 대한상공회의소에서 열린 신년인사회의 인사말에서 “어떠한 권력기관도 국민 위에 존재할 수 없다”며 이렇게 밝혔다. 문 대통령의 발언은 연말 고위공직자범죄수사처 관련 법안 통과, 이날 오전 추미애 법무부 장관의 임명 재가 등으로 이어지는 검찰개혁 속도전의 연장선으로 해석된다. 문 대통령은 “권력기관이 국민의 신뢰를 받을 수 있을 때까지 법적ᆞ제도적 개혁을 멈추지 않겠다”며 “권력기관 스스로 개혁에 앞장서 주길 기대한다”고 말했다. 특히 문 대통령은 “국민이 선출한 대통령으로서 헌법에 따라 권한을 다 하겠다”고 강조했다. 현직 대통령이 헌법에 따른 ‘의무’가 아니라 ‘권한’을 언급한 것은 드문 일로, 문 대통령이 직접 검찰개혁을 비롯한 권력기관 개편을 진두지휘하겠다는 의지를 드러낸 셈이다. 문 대통령이 새해 첫 대국민 메시지인 신년인사에서 권력기관 개혁을 부각하면서 조만간 검찰에 한바탕 소용돌이가 불어닥칠 가능성이 커졌다. 그 첫 번째 수순은 검사장급 이상 검찰 고위직에 대한 인사권 행사가 될 것으로 보인다. 신년인사회에는 이날 오전 임명이 재가된 추미애 법무부 장관과 윤석열 검찰총장도 참석했기 때문에 문 대통령을 발언을 두고 미묘한 장면이 연출됐다. 공정사회와 관련해 문 대통령은 “우리 정부 출범 이후 대기업집단의 순환출자가 대부분 해소되고 불공정거래 관행이 크게 개선되는 등 공정경제에서 일부 성과가 나타나고 있다”면서도 “교육ᆞ사회ᆞ문화 전반에서 국민 눈높이에 맞는 ‘공정사회 개혁’은 아직 갈 길이 멀다”고 말했다. 문 대통령은 이어 “정부는 같은 기회와 공정한 경쟁을 바라는 국민들, 특히 청년들의 높은 요구를 절감했고 반드시 이에 부응할 것”이라고 말했다. 그러면서 “공정사회 없이는 상생 도약도 없다는 각오로 교육과 채용에서 탈세ᆞ병역ᆞ직장에 이르기까지 우리 삶의 모든 영역에 존재하는 불공정을 개선하겠다”고 강조했다. 그간 문재인 정부가 최대 업적으로 자랑해 온 북한과의 관계 재설정에 대해서는 “한반도 평화를 위한 우리 국민의 열망으로 반드시 ‘상생 번영의 평화공동체’를 이뤄낼 것”이라며 “지난해에도 우리는 국제사회와 보조를 맞추며, 한반도 평화를 향해 조금씩 앞으로 나아갔고, 북미 정상 간의 대화 의지도 지속되고 있다”고만 말했다. ', 
'preds': '문 대통령은 권력기관이 국민의 신뢰를 받을 수 있을 때까지 법적ᆞ제도적 개혁을 멈추지 않겠다며 권력기관 스스로 개혁에 앞장서 주길 기대한다고 말했다.', 
'labels': '어떠한 권력기관도 국민 위에 존재할 수 없다는 문 대통령의 발언은 검찰개혁 속도전의 연장선으로 해석된다.'
}
```

```bash
epochs=10
scheduler_type=constant
# extractive summary
accelerate launch run_summarization_no_trainer.py \
--dataset_name summary_and_report.py \
--dataset_config_name base \
--hf_cache_dir huggingface_datasets \
--hf_data_dir data \
--ignore_verifications \
--text_column 'passage' \
--summary_column 'extractive_summary' \
--model_name_or_path 'KETI-AIR/ke-t5-small' \
--source_prefix 'summarize: ' \
--max_source_length 512 \
--preprocessing_num_workers 64 \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 32 \
--lr_scheduler_type $scheduler_type \
--val_max_target_length 200 \
--learning_rate 0.001 \
--num_train_epochs $epochs \
--num_beams 4 \
--eval_on_last_epoch \
--output_dir outputs/ket5_extractive_summary_beam_$scheduler_type-e$epochs
```

### Quantitive Performance

| Model | R-1  | R-2 | R-L (LCS:Longest Common Subsequence) | R-LSUM (LCS summary-level) |
| -------------| ------------- | ------------- | ------------- | ------------- |
| ke-t5-small| 28.4275 | 17.641 | 28.0147 | 28.3445 |


### Qualitive Performance

```python
{'inputs': 'summarize: 문재인 대통령은 2일 “새해에는 더욱 ‘확실한 변화’를 만들어내겠다”며 “권력기관 개혁과 공정사회 개혁이 그 시작”이라고 말했다. 문 대통령은 이날 오전 11시 서울 대한상공회의소에서 열린 신년인사회의 인사말에서 “어떠한 권력기관도 국민 위에 존재할 수 없다”며  이렇게 밝혔다. 문 대통령의 발언은 연말 고위공직자범죄수사처 관련 법안 통과, 이날 오전 추미애 법무부 장관의 임명 재가 등으로 이어지는 검찰개혁 속도전의 연장선으로 해석된다. 문 대통령은 “권력기관이 국민의 신뢰를 받을 수 있을 때까지 법적ᆞ제도적 개혁을 멈추지 않겠다”며 “권력기관 스스로 개혁에 앞장서 주길 기대한다”고 말했다. 특히 문 대통령은 “국민이 선출한 대통령으로서 헌법에 따라 권한을 다 하겠다”고 강조했다. 현 직 대통령이 헌법에 따른 ‘의무’가 아니라 ‘권한’을 언급한 것은 드문 일로, 문 대통령이 직접 검찰개혁을 비롯한 권력기관 개편을 진두지휘하겠다는 의지를 드러낸 셈이다. 문 대통령이 새해 첫 대국민 메시지인 신년인사에서 권력기관 개혁을 부각하면서 조만간 검찰에 한바탕 소용돌이가 불어닥칠 가능성이 커졌다. 그 첫 번째 수순은 검사장급 이상 검찰 고위직에 대한 인사권 행사가 될 것으로 보인다. 신년인사회에는 이날 오전 임명이 재가된 추미애 법무부 장관과 윤석열 검찰총장도 참석했기 때문에 문 대통령을 발언을 두고 미묘한 장면이 연출됐다. 공정사회와 관련해 문 대통령은 “우리 정부 출범 이후 대기업집단의 순환출자가 대부분 해소되고 불공정거래 관행이 크게 개선되는 등 공정경제에서 일부 성과가 나타나고 있다”면서도 “교육ᆞ사회ᆞ문화 전반에서 국민 눈높이에 맞는 ‘공정사회 개혁’은 아직 갈 길이 멀다”고 말했다. 문 대통령은 이어 “정부는 같은 기회와 공 정한 경쟁을 바라는 국민들, 특히 청년들의 높은 요구를 절감했고 반드시 이에 부응할 것”이라고 말했다. 그러면서 “공정사회 없이는 상생 도약도 없다는 각오로 교육과 채용에서 탈세ᆞ병역ᆞ직장에 이르기까지 우리 삶의 모든 영역에 존재하는 불공정을 개선하겠다”고 강조했다. 그간 문재인 정부 가 최대 업적으로 자랑해 온 북한과의 관계 재설정에 대해서는 “한반도 평화를 위한 우리 국민의 열망으로 반드시 ‘상생 번영의 평화공동체’를 이 뤄낼 것”이라며 “지난해에도 우리는 국제사회와 보조를 맞추며, 한반도 평화를 향해 조금씩 앞으로 나아갔고, 북미 정상 간의 대화 의지도 지속되 고 있다”고만 말했다. ', 
'preds': '문재인 대통령은 2일 “새해에는 더욱 ‘확실한 변화’를 만들어내겠다”며 “권력기관 개혁과 공정사회 개혁이 그 시작”이라고 말했다.\n문 대통령은 이날 오전 11시 서울 대한상공회의소에서 열린 신년인사회의 인사말에서 “어떠한 권력기관도 국민 위에 존재할 수 없다”며 이렇게 밝혔다.\n문 대통령은 이날 오전 11시 서울 대한상공회의소에서 열린 신년인사회의 인사말에서 “어떠한 권력기관도 국민 위에  존재할 수 없다”며 이렇게 밝혔다.', 
'labels': '문 대통령은 이날 오전 11시 서울 대한상공회의소에서 열린 신년인사회의 인사말에서 “어떠한 권 력기관도 국민 위에 존재할 수 없다”며 이렇게 밝혔다.\n문 대통령의 발언은 연말 고위공직자범죄수사처 관련 법안 통과, 이날 오전 추미애 법무부 장관의 임명 재가 등으로 이어지는 검찰개혁 속도전의 연장선으로 해석된다.'}
```


```bash
epochs=10
scheduler_type=constant
# ko2en
accelerate launch run_translation_no_trainer.py \
--dataset_name spoken_language_translation.py \
--dataset_config_name base \
--hf_cache_dir huggingface_datasets \
--hf_data_dir data \
--ignore_verifications \
--source_lang 'ko' \
--target_lang 'en' \
--model_name_or_path 'KETI-AIR/ke-t5-small' \
--source_prefix 'translate_ko2en: ' \
--max_source_length 512 \
--preprocessing_num_workers 64 \
--per_device_train_batch_size 64 \
--per_device_eval_batch_size 64 \
--lr_scheduler_type $scheduler_type \
--val_max_target_length 128 \
--learning_rate 0.001 \
--num_train_epochs $epochs \
--num_beams 4 \
--eval_on_last_epoch \
--output_dir outputs/ket5_translate_ko2en_beam_$scheduler_type-e$epochs
```
### Quantitive Performance

| Model | bleu  | 
| -------------| ------------- | 
| ke-t5-small| 48.869 | 


### Qualitive Performance

```python
{'inputs': 'translate_ko2en: 심박수 모니터링, 14일 배터리 수명, 음악 플레이 제어 디스플레이 수면 및 수영 트래킹 방수 스마트 알림 기능이 있습니다.', 'preds': 'Heart rate monitoring, 14 days battery life, music play life display sleep and swimming waterproof smart notifications.', 'labels': ['Features include heart rate monitoring, 14-day battery life, music play control display sleep and swim tracking waterproof smart notifications.']}
{'inputs': 'translate_ko2en: 터치스크린에는 3세대 강화유리와 지문 방지 코팅이 적용되었습니다.', 'preds': 'The touchscreen has 3rd generation tempered glass and anti-fingerprint coating.', 'labels': ['The touch screen has 3rd generation tempered glass and anti-fingerprint coating applied.']}
{'inputs': 'translate_ko2en: 제품 문의는 이메일로 주시면 감사하겠습니다.', 'preds': 'For product inquiries, please send us an email.', 'labels': ['For product inquiries, please send us an email.']}
{'inputs': 'translate_ko2en: 안녕하세요, BBB회사 AAA입니다.', 'preds': 'Hello, BBB company AAA.', 'labels': ['Hello, BBB company AAA.']}
{'inputs': 'translate_ko2en: 저는 얼티밋 래디언스 컬렉션 스킨케어 선물 세트 제품을 소개하려고 합니다.', 'preds': 'I would like to introduce the Ultimate Radiance Collection skincare gift set product.', 'labels': ['I would like to introduce the Ultimate Radiance Collection Skincare Gift Set.']}
{'inputs': 'translate_ko2en: 더 부드럽고, 밝고, 고른 톤의 피부를 선사합니다.', 'preds': 'Provides smoother, brighter, and even skin.', 'labels': ['Provides smoother, brighter, and even-toned skin.']}
{'inputs': 'translate_ko2en: 표면의 불순물을 부드럽게 제거하는 천연 유래 과립 성분이 함유되어 있습니다.', 'preds': 'Contains naturally derived granual ingredients that gently remove surface impurities.', 'labels': ['Contains naturally derived granular ingredients that gently remove surface impurities.']}
```

```bash
epochs=10
scheduler_type=constant
# en2ko
accelerate launch run_translation_no_trainer.py \
--dataset_name spoken_language_translation.py \
--dataset_config_name base \
--hf_cache_dir huggingface_datasets \
--hf_data_dir data \
--ignore_verifications \
--source_lang 'en' \
--target_lang 'ko' \
--model_name_or_path 'KETI-AIR/ke-t5-small' \
--source_prefix 'translate_en2ko: ' \
--max_source_length 512 \
--preprocessing_num_workers 64 \
--per_device_train_batch_size 64 \
--per_device_eval_batch_size 64 \
--lr_scheduler_type $scheduler_type \
--val_max_target_length 128 \
--learning_rate 0.001 \
--num_train_epochs $epochs \
--num_beams 4 \
--eval_on_last_epoch \
--output_dir outputs/ket5_translate_en2ko_beam_$scheduler_type-e$epochs
```

### Quantitive Performance

| Model | bleu  | 
| -------------| ------------- | 
| ke-t5-small| 32.142 | 


### Qualitive Performance

```python
{'inputs': 'translate_en2ko: Features include heart rate monitoring, 14-day battery life, music play control display sleep and swim tracking waterproof smart notifications.', 'preds': '기능에는 심박수 모니터링, 14일간의 배터리 수명, 음악 재생 조절 조절 기능, 음악 플레이 컨트롤 디스플레이 수면 및 수영 안전 스마트 알림이 포함됩니다.', 'labels': ['심박수 모니터링, 14일 배터리 수명, 음악 플레이 제어 디스플레이 수면 및 수영 트래킹 방수 스마트 알림 기능이 있습니다.']}
{'inputs': 'translate_en2ko: The touch screen has 3rd generation tempered glass and anti-fingerprint coating applied.', 'preds': '터치스크린에는 3세대 강화유리와 지문 방지 코팅이 적용되었습니다.', 'labels': ['터치스크린에는 3세대 강화유리와 지문 방지 코팅이 적용되었습니다.']}
{'inputs': 'translate_en2ko: For product inquiries, please send us an email.', 'preds': '제품 문의는 이메일로 주시면 감사하겠습니다.', 'labels': ['제품 문의는 이메일로 주시면 감사하겠습니다.']}
{'inputs': 'translate_en2ko: Hello, BBB company AAA.', 'preds': '안녕하세요, BBB회사 AAA입니다.', 'labels': ['안녕하세요, BBB회사 AAA입니다.']}
{'inputs': 'translate_en2ko: I would like to introduce the Ultimate Radiance Collection Skincare Gift Set.', 'preds': '얼티미트 래디언스 컬렉션 스킨케어 선물세트 세트를 소개하려고 합니다.', 'labels': ['저는 얼티밋 래디언스 컬렉션 스킨케어 선물 세트 제품을 소개하려고 합니다.']}
{'inputs': 'translate_en2ko: Provides smoother, brighter, and even-toned skin.', 'preds': '더 부드럽고, 밝고, 균일한 피부를 선사합니다.', 'labels': ['더 부드럽고, 밝고, 고른 톤의 피부를 선사합니다.']}
{'inputs': 'translate_en2ko: Contains naturally derived granular ingredients that gently remove surface impurities.', 'preds': '표면 불순물을 부드럽게 제거하는 천연 유래 과립 성분이 함유되어 있습니다.', 'labels': ['표면의 불순물을 부드럽게 제거하는 천연 유래 과립 성분이 함유되어 있습니다.']}
```

