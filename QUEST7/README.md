# AIFFEL Campus Online 5th Code Peer Review Templete
- 코더 : 최연석
- 리뷰어 : 본인의 이름을 작성하세요.


# PRT(PeerReviewTemplate) 
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [X] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  > 네, 정상 동작하며, 문제를 해결했습니다.
  > loss, val_loss 값이 1.7, 2.2 로 꽤 좋은 수치가 나온 것 같습니다. 
- [X] 주석을 보고 작성자의 코드가 이해되었나요?
  > 네, 잘 이해됩니다.
- [X] 코드가 에러를 유발할 가능성이 없나요?
  > 네, 없습니다.
- [X] 코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > 네, 그렇다고 판단됩니다.
- [X] 코드가 간결한가요?
  > 네, 간결합니다.

# 예시
1. 코드의 작동 방식을 주석으로 기록합니다.
2. 코드의 작동 방식에 대한 개선 방법을 주석으로 기록합니다.
3. 참고한 링크 및 ChatGPT 프롬프트 명령어가 있다면 주석으로 남겨주세요.

- 주석 
```python
# 전처리
# 전처리 > 분석
# 전처리 > 분석 > 데이터 타입, null, 확인
print(data.info())

# 전처리 > 분석 > 형태 확인
print(data.shape)

# 전처리 > 정제
# 전처리 > 정제 > 중복 확인
print('Text 열에서 중복을 배제한 유일한 샘플의 수 :', data['text'].nunique())
print('headlines 열에서 중복을 배제한 유일한 샘플의 수 :', data['headlines'].nunique())

# 전처리 > 정제 > 중복 제거
data.drop_duplicates(subset = ['text'], inplace=True)
print('중복 제거 전체 샘플수 :', (len(data)))
```

- 경험해보지 못한 수치의 결과!!! 1.73, 2.2
```
Epoch 1/50
302/302 [==============================] - 29s 79ms/step - loss: 3.5931 - val_loss: 3.3149
Epoch 2/50
302/302 [==============================] - 23s 75ms/step - loss: 3.1498 - val_loss: 2.9979
...
Epoch 23/50
302/302 [==============================] - 22s 74ms/step - loss: 1.7560 - val_loss: 2.2075
Epoch 24/50
302/302 [==============================] - 22s 74ms/step - loss: 1.7348 - val_loss: 2.2057
Epoch 00024: early stopping
```



# 참고 링크 및 코드 개선
```python
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
```

> 개선을 위한 내용은 아니고, 참고해볼만한 링크 몇 개 드립니다.

- summa 요약 성능이 좋아서 개인적으로 좀 찾아봐야겠다는 생각이 들었습니다.
    - [Variations of the Similarity Function of TextRank for Automated Summarization](https://arxiv.org/abs/1602.03606)
- 저도 모델 설계 부분을 잘 이해하기 어려웠습니다.
  - [텐서플로(TensorFlow)로 LSTM 구현하기](https://www.hanbit.co.kr/media/channel/view.html?cms_code=CMS6074576268)
  - [간단한 RNN, LSTM 모델 설계](https://ghqls0210.tistory.com/128) 
