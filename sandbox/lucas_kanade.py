#!/usr/bin/env python

'''
Lucas-Kanade tracker
====================

Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.

Usage
-----
lk_track.py [<video_source>]


Keys
----
ESC - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import math

from common import anorm2, draw_str

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

class App:
    def __init__(self, video_src):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.cap = cv.VideoCapture(video_src)
        self.frame_idx = 0

        

    def run(self):

        total_distance = 0
        while True:
            total_distance = 0
            _ret, frame = self.cap.read()
            frame = cv.resize(frame, dsize=None, dst=None, fx=2.0, fy=2.0, interpolation=cv.INTER_LINEAR)
            # frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) adjust for eye perceivable brighness and bad at purple/blue blobs
            frame_gray = np.dot(frame[:, :, :3], [0.3, 0.3, 0.3])
            frame_gray = frame_gray.astype(np.uint8)
            
            vis = frame.copy()

            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, _st, _err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, _st, _err = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1
                new_tracks = []
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    cv.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
                self.tracks = new_tracks
                
                # Creating display image, slow and unecessary
                cv.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))

            # Reads last points and compares it to previous to calculate distances
            for i in self.tracks:
                x_dist = i[-1][0] - i[-2][0]
                y_dist = i[-1][1] - i[-2][1]
                total_distance += math.sqrt((x_dist**2) + (y_dist**2))
            if len(self.tracks) > 0:
                total_distance /= len(self.tracks)
            print(total_distance)
            
            if self.frame_idx % self.detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv.circle(mask, (x, y), 5, 0, -1)
                p = cv.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])


            self.frame_idx += 1
            self.prev_gray = frame_gray
            cv.imshow('lk_track', vis)
            cv.imshow('original', frame)

            ch = cv.waitKey(1)
            if ch == 27:
                break

        return total_distance

def main():

    App('videos/vibrating_matter.mp4').run()
    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()