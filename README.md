# 감정 인식 + Tracking

### [Kor](README.md) | [Eng](README_eng.md)


실시간 영상에서 사람의 감정을 인식하고 추적하는 시스템을 구현.

YOLOv5와 RepVGG를 기반으로 한 감정 인식 모델과 Tracking 기능을 결합하여 다음과 같은 기능을 제공:
- 실시간 영상에서 사람 감지
- 감지된 사람의 감정 인식
- 인식된 사람의 지속적인 추적
- 추적 중인 사람의 감정 변화 모니터링

이 시스템은 기존의 감정 인식 프로그램에서 발생하던 문제점인 **사람 인식이 끊길 경우** 감정 비율이 **초기화**되는 현상을 Tracking 기능 적용.


## 실행 환경

- Python: **3.8.11**

- CUDA: **11.1**

- cuDNN: **8.7.0**

- PyTorch: **1.9.0**









## 실행 방법

1. yolov5-crowdhuman [모델](https://drive.google.com/file/d/1gglIwqxaH2iTvy6lZlXuAcMpd_U0GCUb/view) 다운로드.
2. [YOLOv5](https://github.com/ultralytics/yolov5) 저장소를 **yolov5** 폴더에 `clone`.
3. 사용 라이브러리 설치: `pip install -r requirements.txt`


```bash
> usage: track_v5.py [-h] [--yolo-weights YOLO_WEIGHTS [YOLO_WEIGHTS ...]]
                   [--strong-sort-weights STRONG_SORT_WEIGHTS]
                   [--config-strongsort CONFIG_STRONGSORT] [--source SOURCE]
                   [--imgsz IMGSZ [IMGSZ ...]] [--conf-thres CONF_THRES]
                   [--iou-thres IOU_THRES] [--max-det MAX_DET]
                   [--device DEVICE] [--show-vid] [--save-txt] [--save-conf]
                   [--save-crop] [--save-vid] [--nosave] [--count] [--draw]
                   [--c  lasses CLASSES [CLASSES ...]] [--agnostic-nms]
                   [--augment] [--visualize] [--update] [--project PROJECT]
                   [--save-csv] [--name NAME] [--exist-ok]
                   [--line-thickness LINE_THICKNESS] [--hide-labels]
                   [--hide-conf] [--hide-class] [--half] [--dnn]

optional arguments:
  -h, --help            show this help message and exit
  --yolo-weights YOLO_WEIGHTS [YOLO_WEIGHTS ...]
                        model.pt path(s)
  --strong-sort-weights STRONG_SORT_WEIGHTS
  --config-strongsort CONFIG_STRONGSORT
  --source SOURCE       file/dir/URL/glob, 0 for webcam
  --imgsz IMGSZ [IMGSZ ...], --img IMGSZ [IMGSZ ...], --img-size IMGSZ [IMGSZ ...]
                        inference size h,w
  --conf-thres CONF_THRES
                        confidence threshold
  --iou-thres IOU_THRES
                        NMS IoU threshold
  --max-det MAX_DET     maximum detections per image
  --device DEVICE       cuda device, i.e. 0 or 0,1,2,3 or cpu
  --show-vid            display tracking video results
  --save-txt            save results to *.txt
  --save-conf           save confidences in --save-txt labels
  --save-crop           save cropped prediction boxes
  --save-vid            save video tracking results
  --nosave              do not save images/videos
  --count               display all MOT counts results on screen
  --draw                display object trajectory lines
  --classes CLASSES [CLASSES ...]
                        filter by class: --classes 0, or --classes 0 2 3
  --agnostic-nms        class-agnostic NMS
  --augment             augmented inference
  --visualize           visualize features
  --update              update all models
  --project PROJECT     save results to project/name
  --save-csv            save detected emotion results to project/csv
  --name NAME           save results to project/name
  --exist-ok            existing project/name ok, do not increment
  --line-thickness LINE_THICKNESS
                        bounding box thickness (pixels)
  --hide-labels         hide labels
  --hide-conf           hide confidences
  --hide-class          hide IDs
  --half                use FP16 half-precision inference
  --dnn                 use OpenCV DNN for ONNX inference
```



```bash
> python track_5v.py --yolo-weights [YOLO PT FILE] --source [RTSP]
```







## 감정 인식

8개 범주의 감정 인식이 가능하며 인식된 감정 비율을 화면에 표출.

인식 가능한 얼굴 표정:
- anger: 분노
- contempt: 경멸
- disgust: 혐오
- fear: 공포
- happy: 행복
- neutral: 보통
- sad: 슬픔
- surprise: 놀람

![image](https://github.com/user-attachments/assets/f2ab7b86-f248-4f40-adcf-d3a3846e0713)






<details>
    <summary>만약 ImportError가 발생한다면</summary>

scale_coords, clip_coords 함수를 yolov5/utils/general.py에 추가한다.

- **scale_coords**

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

- **clip_coords**

```python
def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2
```

</details>





## 저장 데이터 형식

![1](1.png)

`--save-csv` 사용 시 감정 인식 데이터 저장을 시작함.


- start time: 인식이 시작되는 시간
- end time: 인식이 끝난 시간
- operation time: 감정 인식 동작 시간
- 8가지 감정 분류: 감정 분포 비율
