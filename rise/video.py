import cv2
import numpy as np
import os

from os.path import isfile, join

max_frames = None

def convert_frames_to_video(pathIn,pathOut,fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

    #for sorting the file names properly
    files.sort(key = lambda x: int(x[:-4]))

    if max_frames is None:
        numb_frames = len(files)
    else:
        numb_frames = min(len(files), max_frames)

    for i in range(numb_frames):
        filename=pathIn + files[i]
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        print(filename)
        #inserting the frames into an image array
        frame_array.append(img)

    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()

def main():
    pathIn= './single_source_images/2023-02-06/'
    pathOut = 'presentation_material/video_abc.avi'
    fps = 40.0
    convert_frames_to_video(pathIn, pathOut, fps)

if __name__=="__main__":
    main()
