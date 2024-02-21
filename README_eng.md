# Emotion detecting + Tracking

### [Kor](README.md) | [Eng](README_eng.md)



This repository is a source code that integrates emotion recognition and tracking, based on [YOLOv5](https://github.com/ultralytics/yolov5) and [RepVGG](https://github.com/DingXiaoH/RepVGG). It tracks individuals undergoing emotion recognition, while simultaneously displaying their current emotions. To address the issue of emotion ratio resetting when human detection is interrupted during emotion recognition in the existing program, **Tracking** has been implemented.



## Initial setup and Installation

Python Version: **3.8.11**

CUDA Version: **11.1**

cuDNN Version: **8.7.0**

1. Download the [model](https://drive.google.com/file/d/1gglIwqxaH2iTvy6lZlXuAcMpd_U0GCUb/view) from the [yolov5-crowdhuman](https://github.com/deepakcrk/yolov5-crowdhuman) repository.
2. Clone the [YOLOv5](https://github.com/ultralytics/yolov5) repository into the **yolov5** folder.
3. Install required libraries: `pip install -r requirements.txt`
4. Install PyTorch: `pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html`





## How to run

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







## Emotion detecting

It is capable of recognizing emotions in 8 categories and displays the detected emotion ratios on the screen.

**Recognizable facial expressions:**

- anger
- contempt
- disgust
- fear
- happy
- neutral
- sad
- surprise







<details>
    <summary>If an ImportError occurs</summary>


Add the `scale_coords`, and `clip_coords` functions to `yolov5/utils/general.py`.

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





## Data storage format

![1](1.png)

When using `--save-csv`, the storage of emotion recognition data begins.

- start time: The time when recognition starts.
- end time: The time when recognition ends.
- operation time: Duration of emotion recognition operation.
- 8 emotional categories: Distribution ratio of emotions.
