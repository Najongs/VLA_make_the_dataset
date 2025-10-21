# import cv2

# cap = cv2.VideoCapture("/dev/video8", cv2.CAP_V4L2)

# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# cap.set(cv2.CAP_PROP_FPS, 30)

# # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
# # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# # cap.set(cv2.CAP_PROP_FPS, 30)


# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("프레임 읽기 실패")
#         break
#     cv2.imshow("HCAM01N MJPEG", frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2, json, numpy as np

def normalize_dist(dist_in):
    d = np.array(dist_in, dtype=np.float64).reshape(-1)  # 1D로
    n = d.size
    if n in (4,5,8,12,14):
        return d.reshape(1, -1)
    if n == 2:  # k1,k2만
        d = np.array([d[0], d[1], 0.0, 0.0], dtype=np.float64)             # 4개
    elif n == 3:  # k1,k2,k3
        d = np.array([d[0], d[1], 0.0, 0.0, d[2]], dtype=np.float64)       # 5개
    elif n == 6:  # 관례: [k1,k2,p1,p2,k3,k4] 로 가정 → k5,k6=0 채워 8개
        d = np.array([d[0], d[1], d[2], d[3], d[4], d[5], 0.0, 0.0], dtype=np.float64)
    else:
        raise ValueError(f"지원 불가한 왜곡계수 길이({n}). 재캘리브 권장.")
    return d.reshape(1, -1)

with open("calibration_result/camera_calibration_1th_200iter.json", "r") as f:
    data = json.load(f)

K   = np.array(data["camera_matrix"], dtype=np.float64)
# dist= np.array(data["dist_coeffs"], dtype=np.float64)
dist = normalize_dist(data["dist_coeffs"])
w, h = data["image_size"]["width"], data["image_size"]["height"]

cap = cv2.VideoCapture("/dev/video8", cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
cap.set(cv2.CAP_PROP_FPS, 30)

# alpha=0: 왜곡 최소(크롭 큼), alpha=1: 크롭 최소(가장자리 왜곡 잔존)
newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), alpha=0.0)

cv2.namedWindow("undistorted")
cv2.createTrackbar("alpha x100", "undistorted", 0, 100, lambda v: None)  # 0~1.0

while True:
    ok, frame = cap.read()
    if not ok: break
    alpha = cv2.getTrackbarPos("alpha x100", "undistorted") / 100.0
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), alpha)
    und = cv2.undistort(frame, K, dist, None, newK)
    cv2.imshow("undistorted", und)
    cv2.imshow("raw", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release(); cv2.destroyAllWindows()
