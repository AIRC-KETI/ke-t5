# Copyright 2022 san kim
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

import os
import json
import zipfile

import datasets

_VERSION = datasets.Version("1.0.0", "")

_URL = "https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=582"

_CITATION = """\
There is no citation information
"""

_DESCRIPTION = """\
# 일상생활 및 구어체 한-영 번역 병렬 말뭉치 데이터

## 소개
- 신경망 기반 기계 번역기 학습 데이터로 활용하기 위한 한영, 영한 말뭉치
- 일상생활 및 구어체 번역기의 성능 향상을 위한 학습용 데이터
## 구축목적
- 상황별 신조어, 약어, 은어, 관용적 의미와 어투까지 효과적으로 전달할 수 있는 인공 신경망 기계 번역기 학습용 말뭉치 데이터 구축

## Usage
```python
from datasets import load_dataset

raw_datasets = load_dataset(
                "spoken_language_translation.py", 
                "base",
                cache_dir="huggingface_datasets", 
                data_dir="data",
                ignore_verifications=True,
            )

dataset_train = raw_datasets["train"]

for item in dataset_train:
    print(item)
    exit()
```

## 데이터 관련 문의처
| 담당자명 | 전화번호 | 이메일 |
| ------------- | ------------- | ------------- |
| 최규동 | 1833-5926 | ken.choi@twigfarm.net |


## Copyright

### 데이터 소개
AI 허브에서 제공되는 인공지능 학습용 데이터(이하 ‘AI데이터’라고 함)는 과학기술정보통신부와 한국지능정보사회진흥원의 「지능정보산업 인프라 조성」 사업의 일환으로 구축되었으며, 본 사업의 유‧무형적 결과물인 데이터, AI 응용모델 및 데이터 저작도구의 소스, 각종 매뉴얼 등(이하 ‘AI데이터 등’)에 대한 일체의 권리는 AI데이터 등의 구축 수행기관 및 참여기관(이하 ‘수행기관 등’)과 한국지능정보사회진흥원에 있습니다.
본 AI데이터 등은 인공지능 기술 및 제품·서비스 발전을 위하여 구축하였으며, 지능형 제품・서비스, 챗봇 등 다양한 분야에서 영리적・비영리적 연구・개발 목적으로 활용할 수 있습니다.

### 데이터 이용정책
- 본 AI데이터 등을 이용하기 위해서 다음 사항에 동의하며 준수해야 함을 고지합니다.

1. 본 AI데이터 등을 이용할 때에는 반드시 한국지능정보사회진흥원의 사업결과임을 밝혀야 하며, 본 AI데이터 등을 이용한 2차적 저작물에도 동일하게 밝혀야 합니다.
2. 국외에 소재하는 법인, 단체 또는 개인이 AI데이터 등을 이용하기 위해서는 수행기관 등 및 한국지능정보사회진흥원과 별도로 합의가 필요합니다.
3. 본 AI데이터 등의 국외 반출을 위해서는 수행기관 등 및 한국지능정보사회진흥원과 별도로 합의가 필요합니다.
4. 본 AI데이터는 인공지능 학습모델의 학습용으로만 사용할 수 있습니다. 한국지능정보사회진흥원은 AI데이터 등의 이용의 목적이나 방법, 내용 등이 위법하거나 부적합하다고 판단될 경우 제공을 거부할 수 있으며, 이미 제공한 경우 이용의 중지와 AI 데이터 등의 환수, 폐기 등을 요구할 수 있습니다.
5. 제공 받은 AI데이터 등을 수행기관 등과 한국지능정보사회진흥원의 승인을 받지 않은 다른 법인, 단체 또는 개인에게 열람하게 하거나 제공, 양도, 대여, 판매하여서는 안됩니다.
6. AI데이터 등에 대해서 제 4항에 따른 목적 외 이용, 제5항에 따른 무단 열람, 제공, 양도, 대여, 판매 등의 결과로 인하여 발생하는 모든 민・형사 상의 책임은 AI데이터 등을 이용한 법인, 단체 또는 개인에게 있습니다.
7. 이용자는 AI 허브 제공 데이터셋 내에 개인정보 등이 포함된 것이 발견된 경우, 즉시 AI 허브에 해당 사실을 신고하고 다운로드 받은 데이터셋을 삭제하여야 합니다.
8. AI 허브로부터 제공받은 비식별 정보(재현정보 포함)를 인공지능 서비스 개발 등의 목적으로 안전하게 이용하여야 하며, 이를 이용해서 개인을 재식별하기 위한 어떠한 행위도 하여서는 안됩니다.
9. 향후 한국지능정보사회진흥원에서 활용사례・성과 등에 관한 실태조사를 수행 할 경우 이에 성실하게 임하여야 합니다.

### 데이터 다운로드 신청방법
1. AI 허브를 통해 제공 중인 AI데이터 등을 다운로드 받기 위해서는 별도의 신청자 본인 확인과 정보 제공, 목적을 밝히는 절차가 필요합니다.
2. AI데이터를 제외한 데이터 설명, 저작 도구 등은 별도의 신청 절차나 로그인 없이 이용이 가능합니다.
3. 한국지능정보사회진흥원이 권리자가 아닌 AI데이터 등은 해당 기관의 이용정책과 다운로드 절차를 따라야 하며 이는 AI 허브와 관련이 없음을 알려 드립니다.

"""

TRAINING_ENKO_FPATH_REL = "025.일상생활 및 구어체 한-영 번역 병렬 말뭉치 데이터/01.데이터/1.Training/라벨링데이터/TL1.zip"
TRAINING_KOEN_FPATH_REL = "025.일상생활 및 구어체 한-영 번역 병렬 말뭉치 데이터/01.데이터/1.Training/라벨링데이터/TL2.zip"

VALIDATION_FPATH_REL = "025.일상생활 및 구어체 한-영 번역 병렬 말뭉치 데이터/01.데이터/2.Validation/라벨링데이터/VL1.zip"


def generator(fpath_list):
    for fpath in fpath_list:
        with zipfile.ZipFile(fpath, "r") as fp:
            file_list = fp.namelist()
            for fname in file_list:
                item_list = json.load(fp.open(fname, "r"))
                for item in item_list["data"]:
                    sn = item["sn"]
                    domain = item["domain"]
                    subdomain = item["subdomain"]
                    ko_script = item["ko"]
                    en_script = item["en"]
                    yield {
                        "sn": sn,
                        "domain": domain,
                        "subdomain": subdomain,
                        "translation":{
                            "ko": ko_script,
                            "en": en_script,
                        },
                    }


        
class SpokenLanguageTranslation(datasets.GeneratorBasedBuilder):
    """SpokenLanguageTranslation Dataset"""

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="base",
            version=_VERSION,
            description="SpokenLanguageTranslation Dataset",
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "sn": datasets.Value("string"),
                    "domain": datasets.Value("string"),
                    "subdomain": datasets.Value("string"),
                    "translation": datasets.Translation(languages=['ko', 'en'])
                }
            ),
            supervised_keys=None,  # Probably needs to be fixed.
            homepage=_URL,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):

        path_kv = {
            datasets.Split.TRAIN: [
                os.path.join(dl_manager.manual_dir, TRAINING_ENKO_FPATH_REL), 
                os.path.join(dl_manager.manual_dir, TRAINING_KOEN_FPATH_REL)
                ],
            datasets.Split.VALIDATION: [
                os.path.join(dl_manager.manual_dir, VALIDATION_FPATH_REL)
                ],
        }

        return [
                datasets.SplitGenerator(name=k, gen_kwargs={'fpath_list': v}) for k, v in path_kv.items()
        ]

    def _generate_examples(self, fpath_list):
        """Yields examples."""
        for idx, item in enumerate(generator(fpath_list)):
            yield idx, item


