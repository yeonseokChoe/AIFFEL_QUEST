# AIFFEL Campus Online 5th Code Peer Review Templete
- 코더 : 최연석
- 리뷰어 : 조준규


# PRT(PeerReviewTemplate) 
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [X] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  
- [X] 주석을 보고 작성자의 코드가 이해되었나요?
  > 주석이 잘 작성되어 있었고 디버깅을 위한 출력이 많아 이해하기 쉬웠다.
- [ ] 코드가 에러를 유발할 가능성이 없나요?
  > 코드를 돌려보니 아웃풋 데이터가 한 문장으로 고정되어 나타났다.  
  > 아마 전처리에서 문제가 생긴 것이 아닌가 생각된다.
- [X] 코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > 주석과 디버깅 코드들로 최대한 이해하려고 노력한 것이 보였다.
- [X] 코드가 간결한가요?
  > 함수로 구현이 잘 되어있어서 코드가 간결하였다.

# 예시
1. 코드의 작동 방식을 주석으로 기록합니다.
2. 코드의 작동 방식에 대한 개선 방법을 주석으로 기록합니다.
3. 참고한 링크 및 ChatGPT 프롬프트 명령어가 있다면 주석으로 남겨주세요.
```python
# 코드 주석 예시
def encoder_layer(units, d_model, num_heads, dropout, name="encoder_layer"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")

    # 패딩 마스크 사용
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    # 첫 번째 서브 레이어 : 멀티 헤드 어텐션 수행 (셀프 어텐션)
    attention = MultiHeadAttention(
        d_model, num_heads, name="attention")({
            'query': inputs,
            'key': inputs,
            'value': inputs,
            'mask': padding_mask
        })

    # 어텐션의 결과는 Dropout과 Layer Normalization이라는 훈련을 돕는 테크닉을 수행
    attention = tf.keras.layers.Dropout(rate=dropout)(attention)
    attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention)

    # 두 번째 서브 레이어 : 2개의 완전연결층
    outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)

    # 완전연결층의 결과는 Dropout과 LayerNormalization이라는 훈련을 돕는 테크닉을 수행
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + outputs)

    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)
```

# 참고 링크 및 코드 개선
```python
# load_conversations에서 preprocessing이 적용되지 않았습니다.
# 적용해서 실행해보면 좀 더 정확한 결과가 나올 것 같습니다.😊
```
