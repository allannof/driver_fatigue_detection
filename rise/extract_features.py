# -*- coding: utf-8 -*-
"""
@author: andreasth, davidek, Abhishek Tandon
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import time
import datetime
import sys
import os
from matplotlib import pyplot as plt
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
import pickle
from tensorflow import keras

all_landmarks = list(range(468))
all_right_eye = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
all_left_eye = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
right_eye = [[33, 133], [160, 144], [159, 145], [158, 153]] # right eye landmark positions
left_eye = [[263, 362], [387, 373], [386, 374], [385, 380]] # left eye landmark positions
mouth = [[61, 291], [39, 181], [0, 17], [269, 405]] # mouth landmark coordinates
headline = [10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94]

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

header = "Y,Participant,EAR,MAR,PUC,MOE,Frame\n"

crop = ["Fold2_part2_20_0.mp4", "Fold2_part2_20_5.MOV"]

rotate = ["Fold3_part2_33_0.mp4", "Fold3_part2_33_5.mp4", "Fold1_part1_01_0.mov",
          "Fold1_part1_02_10.MOV", "Fold1_part1_02_5.MOV", "Fold1_part1_03_0.MOV",
          "Fold1_part1_05_0.MOV", "Fold1_part1_05_10.MOV", "Fold1_part1_05_5.MOV",
          "Fold1_part2_07_0.mp4", "Fold1_part2_07_10.mp4", "Fold1_part2_08_0.mp4",
          "Fold1_part2_08_10.mp4", "Fold1_part2_08_5.mp4", "Fold1_part2_10_0.MOV",
          "Fold1_part2_10_10.MOV", "Fold1_part2_10_5.MOV", "Fold2_part1_13_0.mp4",
          "Fold2_part1_13_10.mp4", "Fold2_part1_13_5.mp4", "Fold2_part1_14_0.mp4",
          "Fold2_part2_19_0.MOV", "Fold2_part2_19_10.MOV", "Fold2_part2_19_5.MOV",
          "Fold2_part2_20_10.mp4", "Fold2_part2_21_0.MOV", "Fold2_part2_21_10.MOV",
          "Fold2_part2_21_5.MOV", "Fold2_part2_22_0.MOV", "Fold2_part2_22_10.MOV",
          "Fold2_part2_22_5.MOV", "Fold3_part1_25_0.mp4", "Fold3_part1_25_10.mp4",
          "Fold3_part1_25_5.mp4", "Fold3_part1_26_10.mp4", "Fold3_part1_28_0.MOV",
          "Fold3_part1_28_10.MOV", "Fold3_part1_28_5.MOV", "Fold3_part1_30_10.mp4",
          "Fold4_part1_38_0.mp4", "Fold4_part1_38_10.mp4", "Fold4_part1_38_5.mp4",
          "Fold4_part1_41_10.MOV", "Fold4_part2_42_0.mp4", "Fold4_part2_42_5.mp4",
          "Fold4_part2_42_10.mp4", "Fold4_part2_43_0.mov", "Fold4_part2_43_10.mov",
          "Fold4_part2_43_5.mp4", "Fold4_part2_47_0.mp4", "Fold4_part2_47_10.mp4",
          "Fold4_part2_47_5.mp4", "Fold5_part1_49_0.mp4", "Fold5_part1_49_10_1.mp4",
          "Fold5_part1_49_10_2.mp4", "Fold5_part1_49_5.mp4", "Fold5_part1_50_0.MOV",
          "Fold5_part1_50_10.MOV", "Fold5_part1_50_5.MOV", "Fold5_part1_52_0.mov",
          "Fold5_part1_52_10.MOV", "Fold5_part1_52_5.MOV", "Fold5_part1_53_0.MOV",
          "Fold5_part1_53_10.MOV", "Fold5_part1_53_5.MOV", "Fold5_part2_56_0.MOV",
          "Fold5_part2_56_10.MOV", "Fold5_part2_56_5.MOV", "Fold5_part2_57_0.MOV",
          "Fold5_part2_57_10.MOV", "Fold5_part2_57_5.MOV", "Fold5_part2_58_0.mp4",
          "Fold5_part2_58_10.mp4", "Fold5_part2_58_5.mp4", "Fold5_part2_59_0.MOV",
          "Fold5_part2_59_10.MOV", "Fold5_part2_59_5.MOV", "Fold3_part2_33_10.mp4"] 

def distance(p1, p2):
    """Compute distance."""
    return (((p1[:2] - p2[:2])**2).sum())**0.5

def eye_aspect_ratio(landmarks, eye):
    """Compute eye aspect ratio."""
    N1 = distance(landmarks[eye[1][0]], landmarks[eye[1][1]])
    N2 = distance(landmarks[eye[2][0]], landmarks[eye[2][1]])
    N3 = distance(landmarks[eye[3][0]], landmarks[eye[3][1]])
    D = distance(landmarks[eye[0][0]], landmarks[eye[0][1]])
    return (N1 + N2 + N3) / (3 * D)

def eye_forehead_ratio(landmarks, eye, scale=1.0):
    """Compute eye to forehead ratio."""
    N1 = distance(landmarks[eye[1][0]], landmarks[eye[1][1]])
    N2 = distance(landmarks[eye[2][0]], landmarks[eye[2][1]])
    N3 = distance(landmarks[eye[3][0]], landmarks[eye[3][1]])
    D = distance(landmarks[headline[1]], landmarks[headline[3]])
    return scale*(N1 + N2 + N3) / (3 * D)
    
def eye_feature(landmarks, right=False):
    """Mean of eye aspect ratios."""
    if right:
        return eye_aspect_ratio(landmarks, right_eye)
    else:
        return (eye_aspect_ratio(landmarks, left_eye) + eye_aspect_ratio(landmarks, right_eye))/2

def mouth_feature(landmarks):
    """Compute mouth aspect ratio."""
    N1 = distance(landmarks[mouth[1][0]], landmarks[mouth[1][1]])
    N2 = distance(landmarks[mouth[2][0]], landmarks[mouth[2][1]])
    N3 = distance(landmarks[mouth[3][0]], landmarks[mouth[3][1]])
    D = distance(landmarks[mouth[0][0]], landmarks[mouth[0][1]])
    return (N1 + N2 + N3)/(3*D)

def pupil_circularity(landmarks, eye):
    """Compute pupil circularity."""
    perimeter = distance(landmarks[eye[0][0]], landmarks[eye[1][0]]) + \
            distance(landmarks[eye[1][0]], landmarks[eye[2][0]]) + \
            distance(landmarks[eye[2][0]], landmarks[eye[3][0]]) + \
            distance(landmarks[eye[3][0]], landmarks[eye[0][1]]) + \
            distance(landmarks[eye[0][1]], landmarks[eye[3][1]]) + \
            distance(landmarks[eye[3][1]], landmarks[eye[2][1]]) + \
            distance(landmarks[eye[2][1]], landmarks[eye[1][1]]) + \
            distance(landmarks[eye[1][1]], landmarks[eye[0][0]])
    area = math.pi * ((distance(landmarks[eye[1][0]], landmarks[eye[3][1]]) * 0.5) ** 2)
    return (4*math.pi*area)/(perimeter**2)

def pupil_feature(landmarks):
    """Mean of pupil circularities."""
    return (pupil_circularity(landmarks, left_eye) + \
        pupil_circularity(landmarks, right_eye))/2

def pixelcoord(landmarks, indecies):
    points = landmarks[indecies]
    points_int = points.astype(int)[:,:2]
    return points_int

def my_draw_landmarks(landmarks, image, indecies):
    points_int = pixelcoord(landmarks, indecies)
    for p in points_int:
        cv2.circle(image, p, 2, (0,255,0), -1)
    return image

def draw_lines(landmarks, image, pairs, color = (255,0,0), thickness=1):
    pairs_int = [pixelcoord(landmarks, pair) for pair in pairs]
    for start_point, end_point in pairs_int:
        cv2.line(image, start_point, end_point, color, thickness)
        cv2.circle(image, start_point, 2, (0,255,0), -1)
        cv2.circle(image, end_point, 2, (0,255,0), -1)

def contract(V):
    """Return lengts of segments of consequitive 1:s of binary vector V."""
    V = np.array([0] + V + [0])
    W = (V[:-1] != V[1:]).astype(int)
    out = np.diff(W.nonzero()[0])[::2]
    return out

def get_events(ears, ll=-0.03, lh=0.01):
    """Detect blinking type event."""
    state = None
    reset = 0
    earp = None
    events = []

    for ear in ears:
        if earp is not None:
            eardiff = ear - earp

            if eardiff < ll:
                state = "n"
                reset = 0

            if (state == "n") and (eardiff > lh):
                for kk in range(1, reset):
                    events[-kk] = 1
                events.append(1)
                state = None
                earp = ear
                continue

            events.append(0)
            if state == "n":
                reset += 1
                # if reset > 100:
                # skipping reset
                if False:
                    state = None
                    reset = 0
        earp = ear

    return events

def eyes_long_closed(ears, interval=50, threshold=0.1):
    """Detect if eyes closed over prolonged period."""
    val = np.array(ears[-interval:]).mean()
    return (val < threshold).astype(int)

def yawning(ears, mars, interval=100, thresholds=[0.25,0.5]):
    """Detect yawning."""
    tear = thresholds[0]
    tmar = thresholds[1]
    val_ear = np.array(ears[-20:]).mean()
    val_mar = np.array(mars[-interval:]).mean()
    xx = ((val_ear < tear) and (val_mar > tmar)).astype(int)
    print(xx, val_ear, val_mar)
    return ((val_ear < tear) and (val_mar > tmar)).astype(int)

def features_from_landmarks(results, image):
    """Compute eye and mouth aspect ratio, pupil circularity and moe ratio."""
    if results.multi_face_landmarks:
        landmarks_positions = []
        # assume that only face is present in the image
        for _, data_point in enumerate(results.multi_face_landmarks[0].landmark):
            landmarks_positions.append([data_point.x, data_point.y, data_point.z]) # saving normalized landmark positions
        landmarks_positions = np.array(landmarks_positions)
        landmarks_positions[:, 0] *= image.shape[1]
        landmarks_positions[:, 1] *= image.shape[0]

        ear = eye_feature(landmarks_positions)
        mar = mouth_feature(landmarks_positions)
        puc = pupil_feature(landmarks_positions)
        moe = mar/ear
        # efr = eye_forehead_ratio(landmarks_positions, right_eye, scale=0.9)
    else:
        ear = "NaN"
        mar = "NaN"
        puc = "NaN"
        moe = "NaN"

    return ear, mar, puc, moe

def print_text(image, toprint, position=(5,100), fontsize=1, fontthickness=1, color=None):
    cgreen = (0,255,0)
    cred = (0,0,255)
    if color==None:
        color= cgreen
    cv2.putText(image, toprint, position, cv2.FONT_HERSHEY_SIMPLEX, fontsize, color, fontthickness)

def compare_facial_features(ears_window, efr_window):
    plt.title("Comparing facial features")
    plt.plot(ears_window, label="EAR")
    plt.plot(efr_window, label="EFR")
    plt.xlabel("Frame")
    plt.ylabel("Feature value")
    plt.legend()
    plt.show()

def plot_ear(ears_window):
    plt.title("Blinking events")
    plt.plot(ears_window, label="EAR")
    plt.xlabel("Frame")
    plt.ylabel("EAR")
    plt.show()


def get_features(image, face_mesh, draw=True, save=False):
    """Compute eye and mouth aspect ratio, pupil circularity and moe ratio."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # To improve performance
    image.flags.writeable = False

    # Detect the face landmarks
    results = face_mesh.process(image)

    # To improve performance
    image.flags.writeable = True

    # Convert back to the BGR color space
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw the face mesh annotations on the image.
    if results.multi_face_landmarks:

      landmarks_positions = []
      # assume that only face is present in the image
      for _, data_point in enumerate(results.multi_face_landmarks[0].landmark):
        landmarks_positions.append([data_point.x, data_point.y, data_point.z]) # saving normalized landmark positions
      landmarks_positions = np.array(landmarks_positions)
      landmarks_positions[:, 0] *= image.shape[1]
      landmarks_positions[:, 1] *= image.shape[0]

      for face_landmarks in results.multi_face_landmarks:
        ear, mar, puc, moe = features_from_landmarks(results, image)

        if draw:

            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_LEFT_EYE,
                # connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style())

            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_RIGHT_EYE,
                # connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style())

            # my_draw_landmarks(landmarks_positions, image, all_landmarks)

            # draw_lines(landmarks_positions, image, [[headline[1],headline[3]]])
            # draw_lines(landmarks_positions, image, left_eye+right_eye)

            # toprint = "EAR: %s" % '0.' + str(int(100*np.around(ear, 2)))
            # print_text(image, toprint)
            # toprint = "EFR: %s" % '0.' + str(int(100*np.around(efr, 2)))
            # print_text(image, toprint, position=(5,150))

        cv2.imshow('MediaPipe FaceMesh', image)

        if save:
            cv2.imwrite('./presentation_material/test.jpg', image)

        return ear, mar, puc, moe, image
    else:
      return "NaN", "NaN", "NaN", "NaN", image

def entire_dataset(basepath, draw=True):
    """Extract features from entire UTARLDD dataset."""
    for fold in sorted(os.listdir(basepath)):
        foldpath = basepath + fold + "/"
        for part in sorted(os.listdir(basepath+fold)):
            partpath = foldpath + part + "/"
            for film in sorted(os.listdir(partpath)):

                filename = "./generated_data/"+fold+"_"+part+"_"+film[:-4]+".csv"
                if os.path.exists(filename):
                    continue

                data = header

                path = partpath + "/" + film
                cap = cv2.VideoCapture(path)

                with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
                    count = -1
                    failures = 0
                    success = True

                    while success:
                        count += 1

                        video_name = (fold+"_"+part+"_"+film)

                        if count == 100:
                            print("Diretory:", partpath)
                            print("File:", film)
                            print(fold+"_"+part+"_"+film)
                        if count % 1000 == 0:
                            print("count:", count)

                        success, image = cap.read()

                        if (video_name in crop) and (count < 600):
                            continue

                        if not success:
                            print("not success")
                            continue

                        image = cv2.flip(image, 1) # flipping all (correct?)
                        if video_name in rotate:
                            image = cv2.rotate(image, cv2.ROTATE_180)

                        if cv2.waitKey(5) & 0xFF == 27:
                            break

                        ear, mar, puc, moe, image = get_features(image, face_mesh, draw=draw)

                        y = film[:-4]
                        if (y == "10_1") or (y == "10_2"):
                            y = "10"

                        assert (y in  ["0", "5", "10"])

                        data += ",".join([str(x) for x in [y, part, ear, mar, puc, moe, count]]) + "\n"

                        if (ear == "none"):
                            failures += 1

                        #if failures > 20:                        
                        #    print("Too many failures")
                        #    print(video_name)
                        #    sys.exit()

                with open(filename, 'w') as f:
                    f.write(data)


                cap.release()

    print("Data generation finished.")


def single_source(live=True, path=None, draw=True, window=2400, store_data=False,
                  store_images=False, apply_model=False, model_params=[0.010,0.028]):
    """Extract features from specific video or live video stream and run model."""
    
    if live:
        cap = cv2.VideoCapture(0)
    else:
        if path is None:
            print("Error: Define path to video.")
            sys.exit()
        cap = cv2.VideoCapture(path)

    if apply_model:
        model = keras.models.load_model('./saved_models/my_model')

    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_landmarks=True) as face_mesh:
      count = -1
      ears_window = []
      mars_window = []
      data = header
      success = True
      while success:
          count += 1

          success, image = cap.read()

          if not success:
              print("not success")
              continue

          #if count == 1000:
              #compare_facial_features(ears_window, efr_window)
              #plot_ear(ears_window)

          # Flip the image horizontally and convert the color space from BGR to RGB
          if live:
              image = cv2.flip(image, 1)
          else:
              # pass
              image = cv2.rotate(image, cv2.ROTATE_180)

          # save = (count == 26)
          ear, mar, puc, moe, image = get_features(image, face_mesh, draw=draw, save=False)

          if live:
              part = "live"
          else:
              part = "single_source"
          data += ",".join([str(x) for x in ["none", part, ear, mar, puc, moe, count]]) + "\n"
          
          if len(ears_window) == window:
              ears_window.pop(0)
          if len(mars_window) == window:
              mars_window.pop(0)
          if (ear != "NaN") and (mar != "NaN"):
              ears_window.append(ear)
              mars_window.append(mar)

          events = get_events(ears_window)
          contr = contract(events)
          if apply_model:
              mp1, mp2 = model_params
              pred = model(np.array([[np.array(events).mean()-mp1, np.array(events).mean()-mp2]])).numpy()[0][0]
              pred = np.round(pred)
              print("count: ", count)
              print(contr)
              print("pred: ", pred)
              print()

          if store_data and (count > 1000):
              timestamp = round(time.time()*1000)
              filename = './single_source_data/' + datetime.datetime.now().date().isoformat() + "_" + str(timestamp) + ".csv"
              with open(filename, 'w') as f:
                  f.write(data)

              data = header
              count = -1

          if store_images:
            basedir = './single_source_images/' + datetime.datetime.now().date().isoformat()
            if not os.path.isdir(basedir):
                os.mkdir(basedir)
            index = '0'*(4-len(str(count)))+str(count)
            filename = basedir + "/" + index + ".jpg"
            print("count: ", count)
            print(filename)
            cv2.imwrite(filename, image)

          # Terminate the process
          if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()


def main():
    # testing
    path = "/media/david/Seagate Expansion Drive/dais/UTA-RLDD/Fold1_part1/05/0.MOV"
    # path = "/media/david/Seagate Expansion Drive/dais/UTA-RLDD/Fold1_part1/05/10.MOV"

    # yawning
    # path = "/media/david/Seagate Expansion Drive/dais/UTA-RLDD/Fold1_part1/01/10.MOV"
    # path = "/media/david/Seagate Expansion Drive/dais/UTA-RLDD/Fold1_part1/06/10.mp4"
    basepath = "/media/david/Seagate Expansion Drive/dais/UTA-RLDD/"

    single_source(live=False, path=path, store_images=False, apply_model=True)
    
    # single_source(live=True, path=path, window=300, apply_model=True, model_params=[0.02, 0.04])

    # entire_dataset(basepath=basepath)

if __name__ == '__main__':
    main()
