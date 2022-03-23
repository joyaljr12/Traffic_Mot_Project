import os
from absl import app
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# To import the calibration file
from Calibration_script import calibValues
#########################################################################
#  for getting video from youtube 
import gdown
import pafy
url = "https://www.youtube.com/watch?v=y7QiNgui5Tg"
video = pafy.new(url)
best = video.streams[3]
#best = video.getbest(preftype="mp4")
#   for uploading captured frames to google drive for calibration
url = 'https://drive.google.com/uc?id=1dAOpHGpb4NwwLxRp0JDBXIXVBk3nehev'
output = 'calibValues.py'
import requests

# for adding time to final excel sheet
from datetime import datetime
import time

##########################################################################

# tensorflow initialisation with gpu if available
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow.python.saved_model import tag_constants
from tensorflow.compat.v1 import ConfigProto


import cv2
import numpy as np
import matplotlib.pyplot as plt


# Project specific local functions
# deep sort imports
#from tools import generate_detections as gdet
from deep_sort_script import nn_matching
from deep_sort_script.detection import Detection
from deep_sort_script.tracker import Tracker

from deep_sort_script import utils as ut


def main(_argv):
    # Definition of the parameters
    file = open("Detection_report.csv", "w")
    file.write("Vehicle ; Departing from ; Destination ; Time \n")
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    #cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    vid = cv2.VideoCapture(best.url)

    # load converted yolov4 tf model
    saved_model_loaded = tf.saved_model.load('./yolov4_detection_model', tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    # initialize deep sort model 
    model_filename = 'mars_feature_extraction_model/mars-small128.pb'
    encoder = ut.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    input_size = 416




    # capture the bboxes of direction for calibration if turned on

    cal = 0
    cal = int(input("Do you want to calibrate : 0 / 1 "))
    vclass = {}

    if cal == 1:
        return_value, frame = vid.read()
        cv2.imwrite('traffic.png',frame)

        # upload the image to Gdrive to calibrate
        headers = {
            "Authorization": "Bearer ya29.a0ARrdaM-WkR7khHcVcxznZjSaefVmOdG7p2i1OKcUC3lR4sqDl8KzDAv0G3Oavq80Zdx_wm3ctd4HMX5d0vv9dvAIsEydtn13ST-z35C24u817cHDVoLjqWdpujy76j_TWtnWYP0bmrG4_0a8qwcmignpzQLS"}
        para = {
            "name": "traffic.png",
            "parents": ["1ko8N_drf3Sm-lRErQJgeH-X2DjF9Ze7B"]
        }
        files = {
            'data': ('metadata', json.dumps(para), 'application/json; charset=UTF-8'),
            'file': open("./traffic.png", "rb")
        }
        r = requests.post(
            "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart",
            headers=headers,
            files=files
        )
        print(r.text)
        # download autpmatically from gdrive
        temp = int(input("Place the calibrate file on the Gdrive Folder of code as calibValues.py and confirm 1"))
        if temp == 1:
            gdown.download(url, output, quiet=False)

    # importing direction bboxes variables from calibration file
    bbox_d = calibValues.bbox_d
    dir = calibValues.dir
    frame_num = 0
    F_Occu = {}
    L_Occu = {}
    Final_ID = {}
    ccar = 1
    # while loop where each frame is processed , 'sk' is the number of skipped frames based on fps required
    fps_r = 30
    sk = 1
    while True:
        return_value, frame = vid.read()
        if not(return_value):
            break

        frame_num +=1
        
        start_time = time.time()
        if (frame_num % 1) == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #print('Frame #: ', frame_num)
            frame_size = frame.shape[:2]
            image_data = cv2.resize(frame, (input_size, input_size))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)
            
            ## get the prediction data from yolov4 model
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)

            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]
            # applying NMS to take only the relevant boxes
            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=0.45,
                score_threshold=0.5
            )

            # convert data to numpy arrays and slice out unused elements
            num_objects = valid_detections.numpy()[0]
            bboxes = boxes.numpy()[0]
            bboxes = bboxes[0:int(num_objects)]
            scores = scores.numpy()[0]
            scores = scores[0:int(num_objects)]
            classes = classes.numpy()[0]
            classes = classes[0:int(num_objects)]

            # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
            original_h, original_w, _ = frame.shape

            bboxes = ut.format_boxes(bboxes, original_h, original_w)

            # store all predictions in one parameter for simplicity when calling functions
            pred_bbox = [bboxes, scores, classes, num_objects]
            # read in all class names from config
            class_names = ut.read_class_names("./yolov4_detection_model/coco.names")

            # by default allow all classes in .names file
            #allowed_classes = list(class_names.values())

            # custom allowed classes (uncomment line below to customize tracker for only people)
            allowed_classes = ['car','bus','motorbike']

            # loop through objects and use class index to get class name, allow only classes in allowed_classes list
            names = []
            deleted_indx = []
            for i in range(num_objects):
                class_indx = int(classes[i])
                class_name = class_names[class_indx]
                if class_name not in allowed_classes:
                    deleted_indx.append(i)
                else:
                    names.append(class_name)
            names = np.array(names)

            # delete detections that are not in allowed_classes
            bboxes = np.delete(bboxes, deleted_indx, axis=0)
            scores = np.delete(scores, deleted_indx, axis=0)

            # Obtain all the detections for the given frame.
            features = encoder(frame, bboxes)
            detections = [Detection(bbox, score, class_name, feature)\
                 for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

            #Pass detections to the deepsort object and obtain the track information.
            tracker.predict()
            tracker.update(detections)

            #initialize color map
            cmap = plt.get_cmap('tab20b')
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]


            # Obtain info from the tracks.
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()
                class_name = track.get_class()

            # draw bbox on screen
                color = colors[int(track.track_id) % len(colors)]
                color = [i * 255 for i in color]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

                # store the bbox of the first occurrence and last occurrence of each ID
                # getting the center of the bounding box

                # the bbox has the top left coordinates(xmin,ymin) and
                # the bottom right( xmax , ymax ) coordinate hence below
                # calculation is required to find the center point of detected box
                del1 = time.time()

                x_c = int(bbox[0]) + (int(bbox[2]) - int(bbox[0]))/2
                y_c = int(bbox[1]) + (int(bbox[3]) - int(bbox[1]))/2

                if str(track.track_id) in F_Occu:   # check if its not a new track ID
                    if str(track.track_id) not in Final_ID: # check if the ID has already been to an exit direction
                        L_Occu[str(track.track_id)] = (x_c , y_c) # save the last occured location of the ID
                        count_l = 0
                        for j in bbox_d:      # loop through all directions
                            if (j[0] < x_c < (j[0] + j[2])) and (j[1] < y_c < (j[1]\
                                 + j[3])): # check if ID is in any of the directional bbox 
                                count_f = 0
                                for k in bbox_d: # check if ID already has a 'Travelling from direction' 
                                    if (k[0] < F_Occu[str(track.track_id)][0] < (k[0] + k[2]))\
                                         and (k[1] < F_Occu[str(track.track_id)][1] < (k[1] + k[3])):
                                        if count_f != count_l:
                                            Final_ID[str(track.track_id)] = (count_f, count_l)
                                            now = datetime.now()
                                            current_time = now.strftime("%H:%M:%S")
                                            print("car {} ; Travelling from {} ; Travelling to {}".format\
                                                (ccar,dir[count_f],dir[count_l])) 
                                            file.write("car" + repr(ccar) + ";" + repr(dir[count_f]) + ";" \
                                                +repr(dir[count_l])+";"+ current_time +"\n")
                                            ccar+=1 # save data for the ID in excel along with time info
                                        break
                                    count_f += 1
                            count_l += 1
                else:
                    F_Occu[str(track.track_id)] = ( x_c ,y_c)

                vclass[str(track.track_id)] = class_name
            
            # calculate frames per second of running detections
            fps = sk / (time.time() - start_time)
            sk = int(fps_r*(time.time() - start_time))
            if sk == 0:
                sk = 1
            result = np.asarray(frame)
            result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            cv2.imshow("Output Video", result)


            if cv2.waitKey(1) & 0xFF == ord('q'): break
        else:
            pass
    cv2.destroyAllWindows()

    file.close()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass




