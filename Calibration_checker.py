import cv2
import numpy as np
import json
import os

# ---------------- Config ----------------
pattern_size = (9, 6)       # 내부 코너 수 (cols, rows)
square_size  = 0.025        # 한 칸 크기 [m]
min_samples  = 200           # 필요한 체커보드 검출 샷 수
remove_outlier_ratio = 0.2  # 상위 20% 오차 이미지 제거
use_rational = True         # K1..K6 모델
# ----------------------------------------

# 3D 보드 좌표 (z=0 평면)
objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints, imgpoints, grays = [], [], []

# 코너 refinement 기준
criteria_subpix = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)

# 카메라 열기 (정확도↑ 위해 YUY2 사용)
cap = cv2.VideoCapture("/dev/video8", cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

print("실시간 체커보드 캡처 시작... (q: 종료)")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 체커보드 검출 (SB 우선, 실패 시 일반)
    try:
        ret, corners = cv2.findChessboardCornersSB(gray, pattern_size, None)
    except AttributeError:
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret:
        corners_refined = cv2.cornerSubPix(
            gray, corners.astype(np.float32), (5,5), (-1,-1), criteria_subpix
        )
        objpoints.append(objp.copy())
        imgpoints.append(corners_refined)
        grays.append(gray.shape[::-1])
        cv2.drawChessboardCorners(frame, pattern_size, corners_refined, ret)
        print(f"체커보드 검출 성공! ({len(objpoints)} / {min_samples})")

    cv2.imshow("Calibration", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if len(objpoints) >= min_samples:
        break

cap.release()
cv2.destroyAllWindows()

if len(objpoints) < 8:
    raise SystemExit("⚠️ 캘리브레이션용 이미지가 부족합니다.")

# --- 1차 캘리브레이션 ---
flags = 0
if use_rational:
    flags |= cv2.CALIB_RATIONAL_MODEL

img_size = grays[0]
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img_size, None, None, flags=flags
)

# --- 이미지별 오차 ---
per_img_err = []
for i in range(len(objpoints)):
    proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
    err = cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / len(proj)
    per_img_err.append((err, i))

per_img_err.sort(reverse=True)
mean_error_1 = np.mean([e for e,_ in per_img_err])

# --- 아웃라이어 제거 후 2차 캘리브레이션 ---
cut = int(len(per_img_err) * remove_outlier_ratio)
keep_idx = sorted([idx for _, idx in per_img_err[cut:]])

objpoints2 = [objpoints[i] for i in keep_idx]
imgpoints2 = [imgpoints[i] for i in keep_idx]
ret2, K2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(
    objpoints2, imgpoints2, img_size, None, None, flags=flags
)

per_img_err2 = []
for i in range(len(objpoints2)):
    proj, _ = cv2.projectPoints(objpoints2[i], rvecs2[i], tvecs2[i], K2, dist2)
    err = cv2.norm(imgpoints2[i], proj, cv2.NORM_L2) / len(proj)
    per_img_err2.append(err)
mean_error_2 = float(np.mean(per_img_err2))

# --- 결과 출력 ---
print("1차 평균 재투영 오차(px):", float(mean_error_1))
print("2차(아웃라이어 제거) 평균 재투영 오차(px):", mean_error_2)
print("최종 Camera Matrix K:\n", K2)
print("최종 Distortion Coeffs:\n", dist2.ravel())

# --- JSON 저장 ---
os.makedirs("calibration_result", exist_ok=True)
result = {
    "image_size": {"width": int(img_size[0]), "height": int(img_size[1])},
    "pattern_size": {"cols": pattern_size[0], "rows": pattern_size[1]},
    "square_size_m": square_size,
    "flags": int(flags),
    "camera_matrix": K2.tolist(),
    "dist_coeffs": dist2.tolist(),
    "mean_reproj_error_px_before": float(mean_error_1),
    "mean_reproj_error_px": mean_error_2
}
with open("calibration_result/camera_calibration.json", "w") as f:
    json.dump(result, f, indent=2)

print("✅ calibration_result/camera_calibration.json 저장 완료")
