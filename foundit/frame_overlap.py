from __future__ import annotations

import numpy as np
import cv2


class FrameTracker:
    def __init__(self):
        self.last_kf = None
        self.kf_pts = None
        self.kf_gray = None
        self.initial_pts = None
        self.original_num_pts = None
        self.tracked_points_ratio = 0.75

    def reset(self):
        self.last_kf = None
        self.kf_pts = None
        self.kf_gray = None
        self.initial_pts = None
        self.original_num_pts = None

    def initialize_keyframe(self, image):
        self.last_kf = image
        self.kf_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.kf_pts = cv2.goodFeaturesToTrack(
        self.kf_gray,
        maxCorners=200,
        qualityLevel=0.01,
        minDistance=10,
        blockSize=7
        )
        if self.kf_pts is None:
            self.initial_pts = None
            self.original_num_pts = 0
            return
        self.initial_pts = self.kf_pts.copy()
        self.original_num_pts = len(self.kf_pts)

    def compute_disparity(self, image, min_disparity, visualize=False):
        if self.last_kf is None or self.kf_pts is None or self.initial_pts is None:
            self.initialize_keyframe(image)
            return True

        curr_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Track from previous frame points to current frame
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        self.kf_gray, curr_gray, self.kf_pts, None,
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        status = status.flatten() == 1

        self.initial_pts = self.initial_pts[status]
        self.kf_pts = next_pts[status]
        self.kf_gray = curr_gray

        if len(self.kf_pts) < 50 or len(self.kf_pts) < self.original_num_pts * self.tracked_points_ratio:
            self.initialize_keyframe(image)
            return True

        # Calculate disparity relative to the starting positions
        displacement = np.linalg.norm(self.kf_pts.reshape(-1, 2) - self.initial_pts.reshape(-1, 2), axis=1)
        mean_disparity = np.mean(displacement)

        if visualize:
            vis = image.copy()
            for p_start, p_curr in zip(self.initial_pts, self.kf_pts):
                pt_start = tuple(p_start.ravel().astype(int))
                pt_curr = tuple(p_curr.ravel().astype(int))
                # Draw the total displacement vector from keyframe start to now
                cv2.line(vis, pt_start, pt_curr, (255, 255, 0), 1)
                cv2.circle(vis, pt_curr, 3, (0, 255, 0), -1)

            cv2.putText(vis, f"Total Disparity: {mean_disparity:.1f}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Incremental Tracking", vis)
            cv2.waitKey(0)

        if mean_disparity > min_disparity:
            self.initialize_keyframe(image)
            return True
        return False