# KoBART

Huggingface🤗 Trainer를 활용하여 BART 한글 문서 생성 요약 Finetune 해보기

최대한 default 값 없이 재현하고 싶어서 작성한 BART Huggingface Fine tuning 코드

# Data

![image](https://user-images.githubusercontent.com/44603549/216853537-442e05eb-1ad5-4cff-92da-5e1f36cebf58.png)

# 결과물

```python
def infer(text):
    if type(text) == int:
        out = model.generate(input_ids = torch.tensor(train_dataset[text]['input_ids'])[None,:].to(device)).logits.argmax(2)
        result = tokenizer.decode(out[0])
    else:
        tmp = [tokenizer.bos_token_id] + tokenizer.encode(text) + [tokenizer.eos_token_id]
        out = model.generate(input_ids = torch.tensor(tmp)[None,:].to(device))
        result = tokenizer.decode(out[0])
    return print(result)
text = """1일 오후 9시까지 최소 20만3220명이 코로나19에 신규 확진됐다. 
또다시 동시간대 최다 기록으로, 사상 처음 20만명대에 진입했다. 
방역 당국과 서울시 등 각 지방자치단체에 따르면 이날 0시부터 오후 9시까지 전국 신규 확진자는 총 20만3220명으로 집계됐다. 
국내 신규 확진자 수가 20만명대를 넘어선 것은 이번이 처음이다. 
동시간대 최다 기록은 지난 23일 오후 9시 기준 16만1389명이었는데, 이를 무려 4만1831명이나 웃돌았다. 
전날 같은 시간 기록한 13만3481명보다도 6만9739명 많다. 
확진자 폭증은 3시간 전인 오후 6시 집계에서도 예견됐다. 
오후 6시까지 최소 17만8603명이 신규 확진돼 동시간대 최다 기록(24일 13만8419명)을 갈아치운 데 이어 이미 직전 0시 기준 역대 최다 기록도 넘어섰다. 
역대 최다 기록은 지난 23일 0시 기준 17만1451명이었다. 
17개 지자체별로 보면 서울 4만6938명, 경기 6만7322명, 인천 1만985명 등 수도권이 12만5245명으로 전체의 61.6%를 차지했다. 
서울과 경기는 모두 동시간대 기준 최다로, 처음으로 각각 4만명과 6만명을 넘어섰다. 
비수도권에서는 7만7975명(38.3%)이 발생했다. 
제주를 제외한 나머지 지역에서 모두 동시간대 최다를 새로 썼다. 
부산 1만890명, 경남 9909명, 대구 6900명, 경북 6977명, 충남 5900명, 대전 5292명, 전북 5150명, 울산 5141명, 광주 5130명, 전남 4996명, 강원 4932명, 충북 3845명, 제주 1513명, 세종 1400명이다. 
집계를 마감하는 자정까지 시간이 남아있는 만큼 2일 0시 기준으로 발표될 신규 확진자 수는 이보다 더 늘어날 수 있다. 이에 따라 최종 집계되는 확진자 수는 21만명 안팎을 기록할 수 있을 전망이다. 
한편 전날 하루 선별진료소에서 이뤄진 검사는 70만8763건으로 검사 양성률은 40.5%다. 
양성률이 40%를 넘은 것은 이번이 처음이다. 
확산세가 계속 거세질 수 있다는 얘기다. 
이날 0시 기준 신규 확진자는 13만8993명이었다. 이틀 연속 13만명대를 이어갔다."""
infer(text)
</s><s> 코로나19 사상 처음 20만 이상 인구가 신규 확진되었다.</s>
```

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
