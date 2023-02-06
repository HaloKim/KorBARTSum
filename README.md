# KoBART

Huggingface🤗 Trainer를 활용하여 BART 한글 문서 생성 요약 Finetune 해보기

최대한 default 값 없이 재현하고 싶어서 작성한 BART Huggingface Fine tuning 코드

# Data

![image](https://user-images.githubusercontent.com/44603549/216853537-442e05eb-1ad5-4cff-92da-5e1f36cebf58.png)

# Score

2 rows val_dataset으로 측정했기 때문에 완전한 score가 아닙니다 !

| Score     | Rouge-1 | Rouge-2 | Rouge-L |
|---------- | ------- | ------- | ------- |
| Recall    | 1.0     | 1.0     | 1.0     |
| Precision | 1.0     | 1.0     | 1.0     |
| F1_score  | 0.99    | 0.99    | 0.99    |

# 깨달은점

* 초기 lr 1e-3 no scheduler 환경에서 학습이 전혀 진행되지 않아 모델 자체의 문제라고 생각했지만, lr과 scheduler warmup을 작성해주니 학습이 잘된점에서 깨달음을 얻음

초기

![캡처](https://user-images.githubusercontent.com/44603549/216854384-ab1d76e8-ecf8-4561-bcd9-1b535e2ad80b.PNG)

lr, scheduler 적용 후

![image](https://user-images.githubusercontent.com/44603549/216854421-b6cf89eb-53c0-4ff2-8f43-0d5bfbbed854.png)

* 당연하지만 인풋의 길이가 모델의 길이와 다르면 이상한 결과가 나옴

![image](https://user-images.githubusercontent.com/44603549/216854469-09498c67-6328-4c26-93ef-ddcd81dd2883.png)


# Reference

[학습에 사용한 데이터](https://github.com/seujung/KoBART-summarization/tree/main/data)

[gogamza/kobart-base-v1](https://huggingface.co/gogamza/kobart-base-v1)
