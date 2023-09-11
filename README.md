# 감정 인식 + Tracking

이 저장소는 [YOLOv5](https://github.com/ultralytics/yolov5)와 [RepVGG](https://github.com/DingXiaoH/RepVGG)를 기반으로 하여 작성한 [감정 인식](https://github.com/George-Ogden/emotion)과 [Tracking](https://github.com/mikel-brostrom/yolo_tracking)을 연동하여 감정 인식할 사람을 추적함과 동시에 현재 감정을 표출하는 소스 코드이다.
기존에 작성했던 프로그램에서 감정 인식 중 **사람 인식이 끊길 경우** 현재까지 저장되던 감정 비율이 **초기화**되는 문제를 해결하기 위해 **Tracking**을 적용하였다.



## 초기 설정 및 설치

Python Version: **3.8.0**
CUDA Version: **11.1**
cuDNN Version: **8.7.0**

1. [yolov5-crowdhuman](https://github.com/deepakcrk/yolov5-crowdhuman) 저장소의 [모델](https://drive.google.com/file/d/1gglIwqxaH2iTvy6lZlXuAcMpd_U0GCUb/view)을 다운받아 **track_v5.py**와 같은 경로에 넣는다.
2. [YOLOv5](https://github.com/ultralytics/yolov5) 저장소를 `clone` 하여 **yolov5** 폴더에 넣는다.

사용 라이브러리 설치: `pip install -r requirements.txt`



## 감정 인식

8개 범주의 감정 인식이 가능하며, 그 중에 3개의 감정(행복, 분노, 보통)을 화면에 표출한다.







<details>
    <summary>만약 ImportError가 발생한다면</summary>
scale_coords, clip_coords 함수를 yolov5/utils/general.py에 추가한다.
**scale_coords**


```python
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords
```

**clip_coords**

```python
def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2
```

</details>