from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
from collections import deque
from numpy import random
import streamlit as st
import math
import time
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import torch

model = YOLO("../model/yolov8n_custom.pt")

classNames = ["car", "motor", "truck", "bus"]

data_deque = {}

mask = cv2.imread("../assets/mask.png")

total_countsEnter = {}
total_countsLeave = {}

# initialize sets to keep track of counted IDs
counted_north_ids = set()
counted_south_ids = set()

total_countsUp = {"car": set(), "motor": set(), "truck": set(), "bus": set()}
total_countsDown = {"car": set(), "motor": set(), "truck": set(), "bus": set()}

thresh0 = 450
thresh1 = 500
thresh2 = 550

line0 = [(0, thresh0), (1280, thresh0)]
line1 = [(0, thresh1), (1280, thresh1)]
line2 = [(0, thresh2), (1280, thresh2)]

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

##########################################################################################
deepsort = None

def init_tracker():
    global deepsort
    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file('deep_sort/configs/deep_sort.yaml')
    # deepsort = DeepSort("osnet_x0_25",
    #                     max_dist=cfg.DEEPSORT.MAX_DIST,
    #                     max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
    #                     max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
    #                     use_cuda=True)
    deepsort = DeepSort("reid_model/transreid_vehicle.pth",
                    max_dist=cfg.DEEPSORT.MAX_DIST,
                    max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)
def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    if label == 0: #car
        color = (85,45,255)
    elif label == 1: # motor
        color = (222,82,175)
    elif label == 2:  # truck
        color = (0, 204, 255)
    elif label == 3:  # bus
        color = (0, 149, 255)
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]

    return tuple(color)

def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)
    
    cv2.circle(img, (x1 +r, y1+r), 2, color, 12)
    cv2.circle(img, (x2 -r, y1+r), 2, color, 12)
    cv2.circle(img, (x1 +r, y2-r), 2, color, 12)
    cv2.circle(img, (x2 -r, y2-r), 2, color, 12)
    
    return img

def UI_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl/3, thickness=tf)[0]

        img = draw_border(img, (c1[0], c1[1] - t_size[1] -3), (c1[0] + t_size[0], c1[1]+3), color, 1, 8, 2)

        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, 0.6, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


def get_direction(point1, point2):
    direction_str = ""

    # calculate y axis direction
    if point1[1] > point2[1]:
        direction_str += "South"
    elif point1[1] < point2[1]:
        direction_str += "North"
    else:
        direction_str += ""

    # calculate x axis direction
    if point1[0] > point2[0]:
        direction_str += "East"
    elif point1[0] < point2[0]:
        direction_str += "West"
    else:
        direction_str += ""

    return direction_str

def check_between_enter(center):
    if thresh1<center[1]<thresh2:
        return True
    return False

def check_between_leave(center):
    if thresh0<center[1]<thresh1:
        return True
    return False

def draw_boxes(img, bbox, classNames, clss, identities=None, offset=(0, 0)):

    # remove tracked point from buffer if object is lost
    for key in list(data_deque):
      if key not in identities:
        data_deque.pop(key)

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # code to find center of bottom edge
        center = (int((x2+x1)/ 2), int((y2+y2)/2))

        # get ID of object
        id = int(identities[i]) if identities is not None else 0

        # create new buffer for new object
        if id not in data_deque:  
          data_deque[id] = deque(maxlen= 128)

        color = compute_color_for_labels(clss[i])

        label = '{}{:d}'.format("", id) + ":"+ '%s' % (classNames[clss[i]])

        obj_name = classNames[clss[i]]
        # add center to buffer
        data_deque[id].appendleft(center)
        cv2.line(img, line0[0], line0[1], [0, 0, 255], 2)
        cv2.line(img, line1[0], line1[1], [0,0,0], 2)
        cv2.line(img, line2[0], line2[1], [255, 0, 0], 2)
        if len(data_deque[id]) >= 2:
            direction = get_direction(data_deque[id][0], data_deque[id][1])
            if "South" in direction and id not in counted_south_ids:
                if check_between_enter(center):
                    if obj_name not in total_countsEnter:
                        total_countsEnter[obj_name] = 1
                    else:
                        total_countsEnter[obj_name] += 1                
                    counted_south_ids.add(id)  # Mark this ID as counted for South
            if "North" in direction and id not in counted_north_ids:
                if check_between_leave(center):
                    if obj_name not in total_countsLeave:
                        total_countsLeave[obj_name] = 1
                    else:
                        total_countsLeave[obj_name] += 1
                    counted_north_ids.add(id)  # Mark this ID as counted for North

            # if "South" in direction:
            #     if check_between(center) or intersect(data_deque[id][0], data_deque[id][1], line1[0], line1[1]):
            #         total_countsUp[obj_name].add(id)
            
            # if "North" in direction:
            #     if check_between(center) or intersect(data_deque[id][0], data_deque[id][1], line1[0], line1[1]):
            #         total_countsDown[obj_name].add(id)

        UI_box(box, img, label=label, color=color, line_thickness=2)
        for idx, (key, value) in enumerate(total_countsEnter.items()):
            if key == 'car':
                cvzone.putTextRect(img, f'car_counts:{value}', (50,50),
                                    scale=2,
                                    thickness=1,
                                    offset=3, colorT=(0,0,0))
            if key == 'motor':
                cvzone.putTextRect(img, f'motor_counts:{value}', (50,80),
                                    scale=2,
                                    thickness=1,
                                    offset=3, colorT=(0,0,0))
            if key == 'truck':
                cvzone.putTextRect(img, f'truck_counts:{value}', (50,110),
                                    scale=2,
                                    thickness=1,
                                    offset=3, colorT=(0,0,0))
            if key == 'bus':
                cvzone.putTextRect(img, f'bus_counts:{value}', (50,140),
                            scale=2,
                            thickness=1,
                            offset=3, colorT=(0,0,0))
        for idx, (key, value) in enumerate(total_countsLeave.items()):
            if key == 'car':
                cvzone.putTextRect(img, f'car_counts:{value}', (950,50),
                                    scale=2,
                                    thickness=1,
                                    offset=3, colorT=(0,0,0))
            if key == 'motor':
                cvzone.putTextRect(img, f'motor_counts:{value}', (950,80),
                                    scale=2,
                                    thickness=1,
                                    offset=3, colorT=(0,0,0))
            if key == 'truck':  
                cvzone.putTextRect(img, f'truck_counts:{value}', (950,110),
                                    scale=2,
                                    thickness=1,
                                    offset=3, colorT=(0,0,0))
            if key == 'bus':
                cvzone.putTextRect(img, f'bus_counts:{value}', (950,140),
                                scale=2,
                                thickness=1,
                                offset=3, colorT=(0,0,0))
        
        # cvzone.putTextRect(img, f'car_counts  :{len(set(total_countsUp["car"]))}', (50,50),
        #                     scale=1,
        #                     thickness=1,
        #                     offset=3, colorT=(0,0,0))
        # cvzone.putTextRect(img, f'motor_counts  :{len(set(total_countsUp["motor"]))}', (50,80),
        #                     scale=1,
        #                     thickness=1,
        #                     offset=3, colorT=(0,0,0))
        # cvzone.putTextRect(img, f'truck_counts:{len(set(total_countsUp["truck"]))}', (50,110),
        #                     scale=1,
        #                     thickness=1,
        #                     offset=3, colorT=(0,0,0))
        # cvzone.putTextRect(img, f'bus_counts:{len(set(total_countsUp["bus"]))}', (50,140),
        #             scale=1,
        #             thickness=1,
        #             offset=3, colorT=(0,0,0))

        # cvzone.putTextRect(img, f'car_counts  :{len(set(total_countsDown["car"]))}', (950,50),
        #                     scale=1,
        #                     thickness=1,
        #                     offset=3, colorT=(0,0,0))
        # cvzone.putTextRect(img, f'motor_counts  :{len(set(total_countsDown["motor"]))}', (950,80),
        #                     scale=1,
        #                     thickness=1,
        #                     offset=3, colorT=(0,0,0))
        # cvzone.putTextRect(img, f'truck_counts:{len(set(total_countsDown["truck"]))}', (950,110),
        #                     scale=1,
        #                     thickness=1,
        #                     offset=3, colorT=(0,0,0))
        # cvzone.putTextRect(img, f'bus_counts:{len(set(total_countsDown["bus"]))}', (950,140),
        #                 scale=1,
        #                 thickness=1,
        #                 offset=3, colorT=(0,0,0))
      
    return img


# Đường dẫn đến file CSS
file_path = "../assets/css/type.css"

def load_css(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        css_content = f.read()
    st.markdown(f'<style>{css_content}</style>', unsafe_allow_html=True)

# Load CSS
load_css(file_path)

# Create directories if they do not exist
output_dir = 'results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Create directories if they do not exist
upload_dir = 'Uploads'
if not os.path.exists(upload_dir ):
    os.makedirs(upload_dir )

output_video_path = os.path.join(output_dir, 'output.avi')

st.title('Vehicle Tracking and Counting')

uploaded_file = st.file_uploader("Upload a video...", type=["mp4", "avi", "mov"])

st1 = time.time()
video_placeholder = st.empty()

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    with open(os.path.join(upload_dir, uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    video_path = os.path.join(upload_dir , uploaded_file.name)
    video_placeholder.video(video_path)

    # Create columns for the buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    init_tracker()
    if torch.cuda.is_available():
        print("True")
        model.cuda()
        deepsort.cuda()

    st1 = time.time()
    with col1:
        if st.button("Start Tracking"):
            cap = cv2.VideoCapture(video_path)  # For Video
            # Định dạng codec và video writer
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_video_path, fourcc, 24.0, (1280, 720))
            
            while cap.isOpened():
                success, img = cap.read()
                if not success:
                    print("Cannot receive frame (stream end?). Exiting ...")
                    break
                
                h, w = mask.shape[:2]
                img = cv2.resize(img, (w, h))
                results = model(img, conf=0.5)
                if results[0]:
                    xywh_bboxs = []
                    confs = []
                    oids = []
                    for res in results[0]:
                        bbox = res.boxes
                        xywh_obj = bbox.xywh.tolist()[0][:4]
                        xywh_bboxs.append(xywh_obj)
                        conf = bbox.conf.item()
                        confs.append(conf)
                        cls = bbox.cls.item()
                        oids.append(int(cls))

                    xywhs = torch.Tensor(xywh_bboxs)
                    confss = torch.Tensor(confs)
                    oids = torch.Tensor(oids)

                    start_time = time.time()
                    outputs = deepsort.update(xywhs, confss, oids, img)
                    end_time = time.time()
                    execution_time = end_time - start_time
                    print(f"Thời gian thực thi: {execution_time} giây")

                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -2]
                        object_id = outputs[:, -1]
                        draw_boxes(img, bbox_xyxy, classNames, object_id, identities)
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        video_placeholder.image(img)

                # Ghi frame vào video
                out.write(img)

            cap.release()  # Release video file
            out.release()  # Release video writer

            # Display the output video
            video_placeholder.video(output_video_path)
            check = True
    ed1 = time.time()
    print("Time: ", ed1-st1)
    with col2:
        if st.button("Reset"):
            st.experimental_rerun()
    with col3:
        # Provide a download link for the output video
        with open(output_video_path, "rb") as video_file:
            video_bytes = video_file.read()
            st.download_button(
                label="Download Tracked Video",
                data=video_bytes,
                file_name="tracked_output.avi",
                mime="video/avi"
            )
else:
    st.warning("Please upload a video file to proceed.")
