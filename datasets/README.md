# 데이터셋 빌드 방법

데이터셋을 생성하기 위해서는 `tensorflow_datasets` package가 설치되어 있어야 합니다. `pip install tensorflow_datasets`를 이용하여 패키지를 설치하고 생성하고자하는 dataset의 root에서 `tfds build`를 통하여 데이터셋을 빌드합니다.

```bash
pushd kor_corpora
tfds build --data_dir ../../tmp/tensorflow_datasets --config nsmc
popd
```

위 스크립트와 같이 빌드하고자 하는 configuration만 빌드할 수 있으며, 데이터셋이 저장될 위치를 지정할 수 있습니다.
위 예제에서는 [NSMC](https://github.com/e9t/nsmc) 데이터만 빌드하며, `../../tmp/tensorflow_datasets`에 데이터셋을 생성합니다.
지정하지 않을 경우 `~/tensorflow_datasets`에 저장됩니다.

[모두의 말뭉치(NIKL)](https://corpus.korean.go.kr/)의 경우 라이센스 문제로 인하여 데이터셋을 직접 다운받으셔야 합니다.
데이터를 다운 받은 뒤 아래와 같은 형식으로 포맷팅합니다.

```
manual_dir/
    └── NIKL
        └── v1.0
            ├── CoLA
            │   ├── NIKL_CoLA_in_domain_dev.tsv
            │   ├── NIKL_CoLA_in_domain_train.tsv
            │   ├── NIKL_CoLA_out_of_domain_dev.tsv
            │   └── References.tsv
            ├── DP
            │   └── NXDP1902008051.json
            ├── LS
            │   ├── NXLS1902008050.json
            │   └── SXLS1902008030.json
            ├── MESSENGER
            │   ├── MDRW1900000002.json
            │   ├── MDRW1900000003.json
            │   ├── MDRW1900000008.json
            │   ├── MDRW1900000010.json
            │   ├── MDRW1900000011.json
            │   ├── MDRW1900000012.json
                        .
                        .
                        .
```

`manual_dir`에 데이터를 위와 같은 방식으로 저장하였으면, 아래 명령어를 통하여 데이터셋을 빌드합니다.

```bash
pushd nikl
tfds build --data_dir ../../tmp/tensorflow_datasets --manual_dir ../../manual_dir --config summarization.v1.0.summary.split
popd
```

더 많은 tfds build 옵션을 알고 싶으시면 `tfds build -h`로 확인하시기 바랍니다.
