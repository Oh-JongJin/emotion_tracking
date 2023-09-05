# 감정 인식 + Tracking

이 저장소는 [YOLOv5](https://github.com/ultralytics/yolov5)와 [RepVGG](https://github.com/DingXiaoH/RepVGG)를 기반으로 하여 작성한 [감정 인식](https://github.com/George-Ogden/emotion)과 [Tracking](https://github.com/mikel-brostrom/yolo_tracking)을 연동하여 감정 인식할 사람을 추적함과 동시에 현재 감정을 표출하는 소스 코드이다.
기존에 작성했던 프로그램에서 감정 인식 중 **사람 인식이 끊길 경우** 현재까지 저장되던 감정 비율이 **초기화**되는 문제를 해결하기 위해 **Tracking**을 적용하였다.



## 초기 설정

1. [yolov5-crowdhuman](https://github.com/deepakcrk/yolov5-crowdhuman) 저장소의 [모델](https://drive.google.com/file/d/1gglIwqxaH2iTvy6lZlXuAcMpd_U0GCUb/view)을 다운받아 **weights** 폴더에 넣는다.
2. [YOLOv5](https://github.com/ultralytics/yolov5) 저장소를 `clone` 하여 **yolov5** 폴더에 넣는다.
3. 