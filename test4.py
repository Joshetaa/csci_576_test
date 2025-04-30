# bigpano_superpoint.py

import cv2
import numpy as np
import torch
import os
import urllib.request

# ========== SuperPoint model ==========
import torch.nn as nn
import torch.nn.functional as F

class SuperPointNet_real(nn.Module):
    def __init__(self):
        super(SuperPointNet_real, self).__init__()
        # Shared Encoder
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1a = nn.Conv2d(1, 64, 3, 1, 1)
        self.conv1b = nn.Conv2d(64, 64, 3, 1, 1)

        self.conv2a = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv2b = nn.Conv2d(64, 64, 3, 1, 1)

        self.conv3a = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv3b = nn.Conv2d(128, 128, 3, 1, 1)

        self.conv4a = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv4b = nn.Conv2d(128, 128, 3, 1, 1)

        # HEADS
        self.convPa = nn.Conv2d(128, 256, 3, 1, 1)  # 3x3
        self.convPb = nn.Conv2d(256, 65, 1, 1, 0)   # 1x1

        self.convDa = nn.Conv2d(128, 256, 3, 1, 1)  # 3x3
        self.convDb = nn.Conv2d(256, 256, 1, 1, 0)  # 1x1



    def forward(self, x):
        # Shared Encoder
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)

        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Detector head
        semi = self.convPb(self.convPa(x))

        # Descriptor head
        desc = self.convDb(self.convDa(x))
        return semi, desc

def download_superpoint_weights(path="superpoint_v1.pth"):
    if not os.path.exists(path):
        print("Downloading SuperPoint weights...")
        url = "https://github.com/magicleap/SuperPointPretrainedNetwork/raw/master/superpoint_v1.pth"
        urllib.request.urlretrieve(url, path)
    return path

def nms_heatmap(heatmap, pool_size=3):
    return (heatmap == cv2.dilate(heatmap, np.ones((pool_size, pool_size)))).astype(np.float32)

def extract_superpoint_keypoints_and_descriptors(model, img, conf_thresh=0.015):
    img_torch = torch.from_numpy(img.astype(np.float32) / 255.).unsqueeze(0).unsqueeze(0)
    semi, desc = model(img_torch)
    semi = semi.squeeze().detach().numpy()
    desc = desc.squeeze().detach().numpy()  # Shape: [C, H/8, W/8]

    exp_semi = np.exp(semi[:-1] - np.max(semi[:-1], axis=0, keepdims=True))
    heatmap = exp_semi / (exp_semi.sum(0) + 1e-8)
    heatmap = heatmap.transpose(1, 2, 0).reshape(img.shape[0], img.shape[1])
    heatmap_nms = nms_heatmap(heatmap)

    keypoints = np.column_stack(np.nonzero(heatmap_nms > conf_thresh))  # [y, x]
    
    if keypoints.shape[0] == 0:
        return [], np.zeros((0, desc.shape[0]), dtype=np.float32)

    # Scale keypoints down for descriptor indexing
    scale = 8
    scaled_kp = keypoints // scale
    scaled_kp[:, 0] = np.clip(scaled_kp[:, 0], 0, desc.shape[1] - 1)
    scaled_kp[:, 1] = np.clip(scaled_kp[:, 1], 0, desc.shape[2] - 1)

    descriptors = desc[:, scaled_kp[:, 0], scaled_kp[:, 1]].T  # Shape: [N, C]

    keypoints_cv = [cv2.KeyPoint(float(pt[1]), float(pt[0]), 1) for pt in keypoints]  # (x, y)
    return keypoints_cv, descriptors


def superpoint_match(kp1, des1, kp2, des2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    return pts1, pts2

# ========== Basic Steps ==========

def extract_frames(video_path, every_n=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % every_n == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames

def estimate_homography(pts1, pts2):
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    return H

def multiband_blend(img1, img2, levels=5):
    gp1, gp2 = [img1.copy()], [img2.copy()]
    for _ in range(levels):
        img1 = cv2.pyrDown(img1)
        img2 = cv2.pyrDown(img2)
        gp1.append(img1)
        gp2.append(img2)
    lp1, lp2 = [gp1[-1]], [gp2[-1]]
    for i in range(levels-1, 0, -1):
        GE1 = cv2.pyrUp(gp1[i])
        GE2 = cv2.pyrUp(gp2[i])
        lp1.append(cv2.subtract(gp1[i-1], GE1))
        lp2.append(cv2.subtract(gp2[i-1], GE2))
    LS = []
    for l1, l2 in zip(lp1, lp2):
        rows, cols, dpt = l1.shape
        ls = np.hstack((l1[:, :cols//2], l2[:, cols//2:]))
        LS.append(ls)
    img = LS[0]
    for i in range(1, levels):
        img = cv2.pyrUp(img)
        if img.shape != LS[i].shape:
            img = cv2.resize(img, (LS[i].shape[1], LS[i].shape[0]))
        img = cv2.add(img, LS[i])
    return img

def stitch(frames, model):
    base_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    base_kp, base_des = extract_superpoint_keypoints_and_descriptors(model, base_gray)

    base = frames[0]
    for i in range(1, len(frames)):
        print(f"Stitching frame {i}...")
        gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        kp, des = extract_superpoint_keypoints_and_descriptors(model, gray)

        pts1, pts2 = superpoint_match(base_kp, base_des, kp, des)
        H = estimate_homography(pts2, pts1)  # Note: aligning to base
        
        warped = cv2.warpPerspective(frames[i], H, (base.shape[1]*2, base.shape[0]))
        base = multiband_blend(base, warped)

        base_kp, base_des = kp, des  # Update

    return base

# ========== Main Runner ==========

if __name__ == "__main__":
    video_path = "deer1.mp4"
    superpoint_weights_path = download_superpoint_weights()

    model = SuperPointNet_real()
    model.load_state_dict(torch.load(superpoint_weights_path, map_location='cpu'))
    model.eval()
    torch.set_grad_enabled(False)


    frames = extract_frames(video_path, every_n=10)
    print(f"Extracted {len(frames)} frames.")

    panorama = stitch(frames, model)

    cv2.imwrite("bigpano_superpoint_output.jpg", panorama)
    print("✅ Panorama saved as bigpano_superpoint_output.jpg")
