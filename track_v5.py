import os
import time
import argparse
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Process, Queue

import sys
import numpy as np
from pathlib import Path
from datetime import datetime

import pandas as pd
from collections import Counter

from torch import no_grad, from_numpy
import torch.backends.cudnn as cudnn

import warnings

warnings.filterwarnings('ignore')

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# import logging
from emotion import detect_emotion, init
from yolov5.models.common import DetectMultiBackend
from yolov5.models.experimental import attempt_load

from yolov5.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, print_args,
                                  check_file)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT


# remove duplicated stream handler to avoid duplicated logging
# logging.getLogger().removeHandler(logging.getLogger().handlers[0])


@no_grad()
def detect(
        source='0',
        yolo_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
        strong_sort_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        config_strongsort=ROOT / 'strong_sort/configs/strong_sort.yaml',
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.5,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=True,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_vid=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/track',  # save results to project/name
        save_csv=False,
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        dnn=False,  # use OpenCV DNN for ONNX inference
        half=False,
        count=False,  # get counts of every obhects
        draw=False,  # draw object trajectory lines
):
    source = str(source)
    start_time, s_time, e_time = None, None, None
    em_count, happy, neutral, anger, contempt, disgust, fear, sad, surprise = 0, 0, 0, 0, 0, 0, 0, 0, 0
    # save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = yolo_weights.stem
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(yolo_weights[0]).stem
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name else exp_name + "_" + strong_sort_weights.stem

    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    os.makedirs(f'runs/csv/{time.strftime("%Y%m%d", time.localtime(time.time()))}', exist_ok=True)  # make csv dir

    # Load model
    device = select_device(device)
    init(device)
    model = DetectMultiBackend(yolo_weights, device=device, dnn=dnn, data=None, fp16=half)
    # model = attempt_load(yolo_weights, map_location=device)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    trajectory = {}

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # initialize StrongSORT
    cfg = get_config()
    cfg.merge_from_file(opt.config_strongsort)

    # Create as many strong sort instances as there are video sources
    strongsort_list = []
    for i in range(nr_sources):
        strongsort_list.append(
            StrongSORT(
                strong_sort_weights,
                device,
                max_dist=cfg.STRONGSORT.MAX_DIST,
                max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
                max_age=cfg.STRONGSORT.MAX_AGE,
                n_init=cfg.STRONGSORT.N_INIT,
                nn_budget=cfg.STRONGSORT.NN_BUDGET,
                mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
                ema_alpha=cfg.STRONGSORT.EMA_ALPHA,

            )
        )
    outputs = [None] * nr_sources

    # Run tracking
    model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
    prev_id, id = None, None
    # box_width, box_height = None, None
    # prev_box_width, prev_box_height = None, None

    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        t1 = time_sync()
        im = from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0

        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, 1, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # nr_sources >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                s += f'{i}: '
                txt_file_name = p.name
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
            curr_frames[i] = im0

            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop

            annotator = Annotator(im0, line_width=1, pil=not ascii)
            if cfg.STRONGSORT.ECC:  # camera motion compensation
                strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                if start_time is None:
                    start_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
                    # s_time = time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time()))
                    s_time = time.time()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                images = []

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to strongsort
                t4 = time_sync()
                outputs[i] = strongsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4

                # draw boxes for visualization
                if len(outputs[i]) > 0:
                    for *xyxy, conf, cls in reversed(det):
                        x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
                        images.append(im0.astype(np.uint8)[int(y1):int(y2), int(x1):int(x2)])

                    em_i = 0
                    box_list = []
                    for j, (output, conf) in enumerate(zip(outputs[i], confs)):
                        # if box_width or box_height is not None:
                        #     print('if box_width or box_height is not None')
                        #     prev_box_width, prev_box_height = box_width, box_height
                        #     print(True if box_width == prev_box_width else False)
                        bboxes = output[0:4]    # box coordinate
                        box_width, box_height = abs(bboxes[0] - bboxes[2]), abs(bboxes[1] - bboxes[3])
                        box_area = int(box_width * box_height)
                        box_list.append(box_area)

                        if box_area == max(box_list):
                            pass
                        else:
                            continue
                        print(box_list)

                        id = output[4]
                        cls = output[5]
                        # bbox_left, bbox_top, bbox_right, bbox_bottom = bboxes

                        if draw:
                            # object trajectory
                            center = ((int(bboxes[0]) + int(bboxes[2])) // 2, (int(bboxes[1]) + int(bboxes[3])) // 2)
                            if id not in trajectory:
                                trajectory[id] = []
                            trajectory[id].append(center)
                            for i1 in range(1, len(trajectory[id])):
                                if trajectory[id][i1 - 1] is None or trajectory[id][i1] is None:
                                    continue
                                thickness = 2
                                try:
                                    cv2.line(im0, trajectory[id][i1 - 1], trajectory[id][i1], (0, 0, 255), thickness)
                                except:
                                    pass

                        if images:
                            emotions = detect_emotion(images, True)

                        if save_txt:
                            # to MOT format
                            bbox_left, bbox_top = output[0], output[1]
                            bbox_w, bbox_h = output[2] - output[0], output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 11 + '\n') % (frame_idx + 1, cls, id, bbox_left,  # MOT format
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))

                        if save_vid or save_crop or show_vid:  # Add bbox to image
                            if prev_id != id:
                                em_count, happy, neutral, anger, contempt, disgust, fear, sad, surprise = \
                                    0, 0, 0, 0, 0, 0, 0, 0, 0

                            em_label = emotions[em_i][0]
                            emotion, em_conf_thresh = emotions[i][0].split(' ')[0], emotions[i][0].split(' ')[1]

                            # if emotion == 'happy' or emotion == 'neutral' or emotion == 'anger':
                            if emotion is not None:
                                em_count += 1
                            if emotion == 'happy':
                                happy += 1
                            elif emotion == 'neutral':
                                neutral += 1
                            elif emotion == 'anger':
                                anger += 1
                            elif emotion == 'contempt':
                                contempt += 1
                            elif emotion == 'disgust':
                                disgust += 1
                            elif emotion == 'fear':
                                fear += 1
                            elif emotion == 'sad':
                                sad += 1
                            elif emotion == 'surprise':
                                surprise += 1

                            if em_count != 0:
                                happy_per = round(happy / em_count * 100, 1)
                                neutral_per = round(neutral / em_count * 100, 1)
                                anger_per = round(anger / em_count * 100, 1)
                                contempt_per = round(contempt / em_count * 100, 1)
                                disgust_per = round(disgust / em_count * 100, 1)
                                fear_per = round(fear / em_count * 100, 1)
                                sad_per = round(sad / em_count * 100, 1)
                                surprise_per = round(surprise / em_count * 100, 1)

                            else:
                                happy_per, neutral_per, anger_per, contempt_per, disgust_per, \
                                    fear_per, sad_per, surprise_per = 0, 0, 0, 0, 0, 0, 0, 0
                            em_i += 1
                            md = time.strftime('%Y%m%d', time.localtime(time.time()))
                            output_path = f'runs/csv/{md}/{start_time}_id({int(id)}).csv'
                            # cur_time = start_time
                            # print(cur_time)
                            # output_path = f'runs/csv/{md}/{int(id)}.csv'
                            # output_path = f'runs/csv/{md}/{int(id)}.parquet'
                            # epoch = datetime.now().isoformat(sep=' ', timespec='milliseconds')
                            # epoch = time.time()
                            # strtime = datetime.today()
                            # data = [[str(epoch), strtime, happy_per, neutral_per, anger_per]]
                            # result = pd.DataFrame(data)

                            # e_time = time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())),
                            e_time = time.time()
                            o_time = round(float(e_time) - float(s_time), 2)

                            s_time_str = time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(s_time))
                            e_time_str = time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(e_time))
                            result = pd.DataFrame([[s_time_str, e_time_str, o_time, happy_per, neutral_per, anger_per,
                                                    contempt_per, disgust_per, fear_per, sad_per, surprise_per]])

                            if save_csv:  # Save csv file
                                result.to_csv(output_path, mode='w', index=False,
                                              header=['start time', 'end time', 'operation time', 'happy', 'neutral',
                                                      'anger', 'contempt', 'disgust', 'fear', 'sad', 'surprise'])

                            c = int(cls)  # integer class
                            id = int(id)  # integer id
                            label = None if hide_labels else \
                                (f'{id} {names[c]}' if hide_conf else
                                 (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f} {em_label}'))
                            annotator.box_label(bboxes, label, color=colors(c, True))
                            # annotator.box_label(bboxes, label, color=(0, 255, 0))

                            if save_crop:
                                txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[
                                    c] / f'{id}' / f'{p.stem}.jpg', BGR=True)

                # LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), StrongSORT:({t5 - t4:.3f}s)')

            else:
                start_time = None

                strongsort_list[i].increment_ages()
                LOGGER.info('No detections')

            if count:
                # itemDict = {}
                # NOTE: this works only if save-txt is true
                try:
                    df = pd.read_csv(txt_path + '.txt', header=None, delim_whitespace=True)
                    df = df.iloc[:, 0:3]
                    df.columns = ["frameid", "class", "trackid"]
                    df = df[['class', 'trackid']]
                    df = (df.groupby('trackid')['class']
                          .apply(list)
                          .apply(lambda x: sorted(x))
                          ).reset_index()

                    df.colums = ["trackid", "class"]
                    df['class'] = df['class'].apply(lambda x: Counter(x).most_common(1)[0][0])
                    vc = df['class'].value_counts()
                    vc = dict(vc)

                    vc2 = {}
                    for key, val in enumerate(names):
                        vc2[key] = val
                    # itemDict = dict((vc2[key], value) for (key, value) in vc.items())
                    # itemDict = dict(sorted(itemDict.items(), key=lambda item: item[0]))
                    # print(itemDict)

                except:
                    pass

            # Saving last frame
            cv2.imwrite('testing.jpg', im0)

            if show_vid:
                sub_img = im0[0:int(im0.shape[0] * 2 / 3.5), 0:int(im0.shape[1] / 3)]
                white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 200
                res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
                im0[0:int(im0.shape[0] * 2 / 3.5), 0:int(im0.shape[1] / 3)] = res

                if em_count != 0:
                    happy_per = round(happy / em_count * 100, 1)
                    neutral_per = round(neutral / em_count * 100, 1)
                    anger_per = round(anger / em_count * 100, 1)
                    contempt_per = round(contempt / em_count * 100, 1)
                    disgust_per = round(disgust / em_count * 100, 1)
                    fear_per = round(fear / em_count * 100, 1)
                    sad_per = round(sad / em_count * 100, 1)
                    surprise_per = round(surprise / em_count * 100, 1)
                else:
                    happy_per, neutral_per, anger_per, contempt_per, disgust_per, \
                        fear_per, sad_per, surprise_per = 0, 0, 0, 0, 0, 0, 0, 0

                cv2.putText(im0, f'happy: {happy_per}%', (int(im0.shape[1] / 12), int(im0.shape[0] / 9)),
                            cv2.FONT_ITALIC, 0.5, (0, 0, 0), 1)
                cv2.putText(im0, f'neutral: {neutral_per}%', (int(im0.shape[1] / 12), int(im0.shape[0] * 1.5 / 9)),
                            cv2.FONT_ITALIC, 0.5, (0, 0, 0), 1)
                cv2.putText(im0, f'anger: {anger_per}%', (int(im0.shape[1] / 12), int(im0.shape[0] * 2 / 9)),
                            cv2.FONT_ITALIC, 0.5, (0, 0, 0), 1)
                cv2.putText(im0, f'contempt: {contempt_per}%', (int(im0.shape[1] / 12), int(im0.shape[0] * 2.5 / 9)),
                            cv2.FONT_ITALIC, 0.5, (0, 0, 0), 1)
                cv2.putText(im0, f'disgust: {disgust_per}%', (int(im0.shape[1] / 12), int(im0.shape[0] * 3 / 9)),
                            cv2.FONT_ITALIC, 0.5, (0, 0, 0), 1)
                cv2.putText(im0, f'fear: {fear_per}%', (int(im0.shape[1] / 12), int(im0.shape[0] * 3.5 / 9)),
                            cv2.FONT_ITALIC, 0.5, (0, 0, 0), 1)
                cv2.putText(im0, f'sad: {sad_per}%', (int(im0.shape[1] / 12), int(im0.shape[0] * 4 / 9)),
                            cv2.FONT_ITALIC, 0.5, (0, 0, 0), 1)
                cv2.putText(im0, f'surprise: {surprise_per}%', (int(im0.shape[1] / 12), int(im0.shape[0] * 4.5 / 9)),
                            cv2.FONT_ITALIC, 0.5, (0, 0, 0), 1)

                current_time = time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time()))
                cv2.putText(im0, f'{current_time}', (int(im0.shape[1] / 2.5), int(im0.shape[0] / 15)),
                            cv2.FONT_ITALIC, 1, (0, 0, 0), 2)

                cv2.imshow('Emotion Detection', im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    break

            # Save results (image with detections)
            if save_vid:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video (Only saving mp4 extension)
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

            prev_frames[i] = curr_frames[i]
            prev_id = id

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(
        f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms strong sort update per image at shape '
        f'{(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr(save_dir)}{s}")
    if update:
        strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=str, default=WEIGHTS / 'yolov5n.pt', help='model.pt path(s)')
    parser.add_argument('--strong-sort-weights', type=str, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--config-strongsort', type=str, default='strong_sort/configs/strong_sort.yaml')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--count', action='store_true', help='display all MOT counts results on screen')
    parser.add_argument('--draw', action='store_true', help='display object trajectory lines')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--save-csv', action='store_true', help='save detected emotion results to project/csv')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    detect(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
