# AIFFEL Campus Online 5th Code Peer Review Templete
- 코더 : 최연석
- 리뷰어 : 조대호


# PRT(PeerReviewTemplate) 
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [O] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  >네 코드가 정상적으로 실행되고, 주어진 문제를 해결하였습니다. 다만 gradcam과 cam간의 iou를 비교했으면 더 좋았을것 같습니다.
- [O] 주석을 보고 작성자의 코드가 이해되었나요?
  > 네 코드를 보고 이해하기 어려운 부분을 주석으로 작성해주셔서 이해하는데 도움이 되었습니다.
- [O] 코드가 에러를 유발할 가능성이 없나요?
  > 네 에러가 발생하지 않았습니다.
- [△] 코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > 라벨에 저장되어 있는 바운딩 박스를 그릴 때, 이미지에 맞게 그려야 하는데 이부분이 잘못되었던것 같습니다.  
'''python
#라벨의 바운딩 박스 이미지에 그리기
image_height, image_width, _ = item['image'].shape
bbox_normalized = item['objects']['bbox'][0]

ymin = int(bbox_normalized[0] * image_height)
xmin = int(bbox_normalized[1] * image_width)
ymax = int(bbox_normalized[2] * image_height)
xmax = int(bbox_normalized[3] * image_width)

rect1 = [[xmin, ymax], [xmin, ymin], [xmax,ymin],[xmax, ymax]]
rect1_np = np.array(rect1)
'''
- [O] 코드가 간결한가요?
  > 네 중복되는 부분을 함수로 작성하여 간결하게 작성해 주셨습니다.

# 참고 링크 및 코드 개선
```python
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
```
