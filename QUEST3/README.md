# AIFFEL Campus Online 5th Code Peer Review
- 코더 : 최연석
- 리뷰어 : 김성진


# PRT(PeerReviewTemplate) 
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [X] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  > 데이터 전처리, 모델 학습, 예측, submission 과정 완료되었습니다.
- [X] 주석을 보고 작성자의 코드가 이해되었나요?
  > 각 함수와 처리과정의 코드 상단에 주석이 설명되어 있습니다.
- [X] 코드가 에러를 유발할 가능성이 없나요?
  > 네
- [X] 코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > 네
- [X] 코드가 간결한가요?
  > 네

# 예시
1. 코드의 작동 방식을 주석으로 기록합니다.
2. 코드의 작동 방식에 대한 개선 방법을 주석으로 기록합니다.
3. 참고한 링크 및 ChatGPT 프롬프트 명령어가 있다면 주석으로 남겨주세요.

- 전처리 과정 일부
```python
#분포 확인
def show_sns_distribution(df):
    f, ax = plt.subplots(figsize=(8, 6))
    sns.distplot(df['price'])
    plt.show()

#정규화
#분포 확인
show_sns_distribution(df_train)
```

- 모델 학습
```python
#LinearRegression 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

print(model.score(X_train,y_train))

#학습된 모델로 X_test에 대한 예측값 출력 및 손실함수값 계산
predictions = model.predict(X_test)
print(predictions)
```

- 예측 및 submission 파일 생성

```python
#test predictions
test_predictions = model.predict(df_test_X)
test_predictions = convert_exp_data(test_predictions)
print(test_predictions)

#test predictions 저장
df_test["price"] = test_predictions
print(df_test)

save_file(FILE_PATH, "sunmission.csv", pd.DataFrame(df_test))
```


# 참고 링크 및 코드 개선
```python
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
```
- 없습니다.