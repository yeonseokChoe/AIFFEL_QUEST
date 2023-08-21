# AIFFEL Campus Online 5th Code Peer Review Templete
- 코더 : 최연석
- 리뷰어 : 조준규


# PRT(PeerReviewTemplate) 
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [X] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  
- [X] 주석을 보고 작성자의 코드가 이해되었나요?
  > 각 함수가 어떤 역할을 하는 함수인지 주석이 달려있어 이해하기 쉬웠다.
- [X] 코드가 에러를 유발할 가능성이 없나요?
  > 전체적으로 에러가 일어날 부분은 찾지 못했다.
- [X] 코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > 함수를 제대로 구현하였고 이를 알맞는 위치에 잘 사용한 것으로 보아 코드를 제대로 이해했다고 생각한다.
- [X] 코드가 간결한가요?
  > 코드가 간결해지도록 함수를 사용하여 구현하였다.

# 예시
1. 코드의 작동 방식을 주석으로 기록합니다.
2. 코드의 작동 방식에 대한 개선 방법을 주석으로 기록합니다.
3. 참고한 링크 및 ChatGPT 프롬프트 명령어가 있다면 주석으로 남겨주세요.
```python
# 코드 주석 예시
# 함수 위에 어떤 역할의 함수인지 명시해주었다.
# --------------------------------------------------------
# augmentation
# --------------------------------------------------------
def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.clip_by_value(image, 0, 1)
    return image, label

# --------------------------------------------------------
# Cutmix
# --------------------------------------------------------
def get_clip_box(image_a, image_b):
    # image.shape = (height, width, channel)
    image_size_x = image_a.shape[1]
    image_size_y = image_a.shape[0]
```
```python
# 코드 간결성 예시
# 모델 생성, compile, fit을 함수로 구현하여 중복되는 코드가 없도록 하였다.
def compile_and_fit(train_data, val_data, epochs=15, model=None):
    if model is None:
        model = get_resnet50()
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
        metrics=['accuracy'],
    )

    return model, model.fit(
        train_data, 
        steps_per_epoch=int(ds_info.splits['train'].num_examples/16),
        validation_steps=int(ds_info.splits['test'].num_examples/16),
        epochs=epochs,
        validation_data=val_data,
        verbose=1,
        use_multiprocessing=True,
    )

resnet50_no_aug, history_no_aug = compile_and_fit(ds_train_no_aug, ds_val)
resnet50_aug, history_aug = compile_and_fit(ds_train_aug, ds_val)
resnet50_aug_cutmix, history_aug_cutmix = compile_and_fit(ds_train_aug_cutmix, ds_val)
resnet50_aug_mixup, history_aug_mixup = compile_and_fit(ds_train_aug_mixup, ds_val)
```

# 참고 링크 및 코드 개선
```python
# batch size로 나눠주는 부분은 변수를 다로 지정해서 대입하는 것이 혹시 모를 오류를 방지합니다.😊
def compile_and_fit(train_data, val_data, epochs=15, batch_size=16, model=None):
    if model is None:
        model = get_resnet50()
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
        metrics=['accuracy'],
    )

    return model, model.fit(
        train_data, 
        steps_per_epoch=int(ds_info.splits['train'].num_examples/batch_size),
        validation_steps=int(ds_info.splits['test'].num_examples/batch_size),
        epochs=epochs,
        validation_data=val_data,
        verbose=1,
        use_multiprocessing=True,
    )
```
