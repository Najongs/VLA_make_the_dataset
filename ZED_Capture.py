import os
import sys
import time
import csv
import cv2
import random
import argparse
import logging
import pathlib
import threading
from queue import Queue

import depthai as dai
import pyzed.sl as sl
import mecademicpy.robot as mdr
import mecademicpy.robot_initializer as initializer
import mecademicpy.tools as tools

# 저장 폴더
OUTPUT_DIR = "./dataset/ZED_Captures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 0️⃣ 비동기 이미지 저장기
# ============================================================
class AsyncImageWriter(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.queue = Queue(maxsize=5000)
        self.stop_evt = threading.Event()

    def submit(self, filename, frame):
        """이미지 저장 요청"""
        if not self.stop_evt.is_set():
            self.queue.put((filename, frame))

    def stop(self):
        """큐 종료 신호"""
        self.stop_evt.set()
        self.queue.put((None, None))

    def run(self):
        """큐에 들어온 이미지를 순차적으로 저장"""
        while True:
            filename, frame = self.queue.get()
            if filename is None:
                break
            try:
                cv2.imwrite(filename, frame)
            except Exception as e:
                print(f"[AsyncWriter] Error saving {filename}: {e}")
        print("[AsyncWriter] All pending images written. Exiting...")

# ============================================================
# 1️⃣ Global Clock: UNIX 시간 기준 공유
# ============================================================
class GlobalClock(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.timestamp = round(time.time(), 3)  # UNIX time 초기값
        self.running = True
        self.lock = threading.Lock()

    def now(self):
        with self.lock:
            return self.timestamp

    def run(self):
        while self.running:
            with self.lock:
                self.timestamp = round(time.time(), 3)
            time.sleep(0.05) # 0.1초로 잡으면 HD1200/HD1080/SVGA 에서는 불가능해보임

    def stop(self):
        self.running = False

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-tag", default=None, help="출력 폴더 접미사 (예: 20th)")
    p.add_argument("--robot", choices=["on", "off"], default="on",
                   help="로봇 제어 활성화 여부 (기본값: on)")
    return p.parse_args()

# ============================================================
# 2️⃣ OAK 카메라 캡처
# ============================================================
def run_oak_capture(mxid, output_subdir, start_event, stop_event, clock, writer):
    output_dir = os.path.join(OUTPUT_DIR, output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    WARMUP_FRAMES = 5
    frame_count = 0

    # ⚙️ Pipeline 설정
    pipeline = dai.Pipeline()
    cam = pipeline.createColorCamera()
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)  # ✅ RGB 대신 CAM_A 사용 (Deprecation 해결)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setFps(60)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.initialControl.setManualFocus(110)

    xout = pipeline.createXLinkOut()
    xout.setStreamName("video")
    cam.video.link(xout.input)

    device_info = dai.DeviceInfo(mxid)
    usb_speed = dai.UsbSpeed.SUPER  # 또는 SUPER (자동 감지 가능)
    
    try:
        with dai.Device(pipeline, device_info, usb_speed) as device:
            q_video = device.getOutputQueue(name="video", maxSize=8, blocking=False)
            print(f"✅ OAK {mxid} initialized successfully. Waiting for start_event...")

            # ZED와 동일한 구조
            start_event.wait()
            print(f"OAK {mxid} started capturing.")

            while not stop_event.is_set():
                frame = q_video.tryGet()
                if frame is None:
                    time.sleep(0.005)
                    continue

                frame_count += 1
                if frame_count <= WARMUP_FRAMES:
                    if frame_count == 1:
                        print("Initial OAK frame received. Warming up...")
                    continue

                img = frame.getCvFrame()
                if img is None:
                    continue

                t_rel = clock.now()
                filename = os.path.join(output_dir, f"oak_{mxid}_{t_rel:.3f}.jpg")
                writer.submit(filename, img)

            print(f"OAK {mxid} stopped capturing.")

    except Exception as e:
        print(f"❌ Failed to start OAK camera {mxid}: {e}")


# class HandEyeCamera(threading.Thread):
#     def __init__(self, device_path, serial_name, output_subdir, start_event, stop_event):
#         super().__init__(daemon=True)
#         self.device_path = device_path
#         self.serial_name = serial_name
#         self.output_dir = os.path.join(OUTPUT_DIR, output_subdir)
#         os.makedirs(self.output_dir, exist_ok=True)
#         self.start_event = start_event
#         self.stop_event = stop_event
#         self.cap = None
#         self.ready = False

#     def init_camera(self):
#         cap = cv2.VideoCapture(self.device_path, cv2.CAP_V4L2)
#         cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#         cap.set(cv2.CAP_PROP_FPS, 30)
#         if cap.isOpened():
#             self.cap = cap
#             self.ready = True
#             print(f"HandEye camera {self.serial_name} initialized")
#         else:
#             print(f"Failed to open HandEye camera {self.serial_name}")
#             sys.exit(1)

#     def run(self):
#         if not self.ready:
#             print(f"HandEye camera {self.serial_name} not ready. Skipping capture.")
#             return

#         self.start_event.wait()
#         print(f"HandEye camera {self.serial_name} started capturing")

#         while not self.stop_event.is_set():
#             ret, frame = self.cap.read()
#             if not ret:
#                 continue
#             t = time.time()
#             filename = os.path.join(self.output_dir, f"handeye/handeye_{self.serial_name}_{t:.3f}.jpg")
#             cv2.imwrite(filename, frame)

#         self.cap.release()
#         print(f"HandEye camera {self.serial_name} stopped")


# ============================================================
# 3️⃣ ZED 카메라 캡처
# ============================================================
class ZedCamera(threading.Thread):
    def __init__(self, serial_number, output_subdir, start_event, stop_event, clock, writer):
        super().__init__(daemon=True)
        self.serial_number = serial_number
        self.output_dir = os.path.join(OUTPUT_DIR, output_subdir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.zed = sl.Camera()
        self.runtime_params = sl.RuntimeParameters()
        self.start_event = start_event
        self.stop_event = stop_event
        self.ready = False
        self.clock = clock
        self.writer = writer

    def init_camera(self):
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD1080  # HD1200, HD1080, SVGA
        init_params.svo_real_time_mode = True
        init_params.camera_fps = 60
        init_params.set_from_serial_number(self.serial_number)
        init_params.depth_mode = sl.DEPTH_MODE.NONE   #NEURAL

        if self.zed.open(init_params) == sl.ERROR_CODE.SUCCESS:
            if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
                self.ready = True
            print(f"ZED {self.serial_number} initialized")
        else:
            print(f"Failed to open ZED {self.serial_number}")
            sys.exit(1)

    def run(self):
        if not self.ready:
            print(f"ZED {self.serial_number} not ready. Skipping.")
            return

        left_image, right_image = sl.Mat(), sl.Mat()
        self.start_event.wait()
        print(f"ZED {self.serial_number} started capturing")

        while not self.stop_event.is_set():
            self.runtime_params.enable_fill_mode = False
            if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(left_image, sl.VIEW.LEFT)
                self.zed.retrieve_image(right_image, sl.VIEW.RIGHT)

                t_rel = self.clock.now()
                left_path = os.path.join(self.output_dir, f"zed_{self.serial_number}_left_{t_rel:.3f}.jpg")
                right_path = os.path.join(self.output_dir, f"zed_{self.serial_number}_right_{t_rel:.3f}.jpg")

                self.writer.submit(left_path, left_image.get_data()[:, :, :3])
                self.writer.submit(right_path, right_image.get_data()[:, :, :3])

        self.zed.close()
        print(f"ZED {self.serial_number} stopped")

# ============================================================
# 4️⃣ 실시간 로봇 데이터 샘플러
# ============================================================
class RtSampler(threading.Thread):
    def __init__(self, robot, out_csv, clock, rate_hz=100):
        super().__init__(daemon=True)
        self.robot = robot
        self.out_csv = out_csv
        self.dt = 1.0 / float(rate_hz)
        self.clock = clock
        self.stop_evt = threading.Event()

    def stop(self):
        self.stop_evt.set()

    def run(self):
        with open(self.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp",
                "joint_angle_1", "joint_angle_2", "joint_angle_3",
                "joint_angle_4", "joint_angle_5", "joint_angle_6",
                "EE_x", "EE_y", "EE_z", "EE_a", "EE_b", "EE_r"
            ])
            next_t = time.time()
            while not self.stop_evt.is_set():
                q, p = None, None
                for name in ("GetJoints", "GetJointPos", "GetJointAngles"):
                    fn = getattr(self.robot, name, None)
                    if callable(fn):
                        try:
                            q = list(fn())
                            break
                        except Exception:
                            pass
                for name in ("GetPose", "GetPoseXYZABC", "GetCartesianPose"):
                    fn = getattr(self.robot, name, None)
                    if callable(fn):
                        try:
                            p = list(fn())
                            break
                        except Exception:
                            pass

                if q is not None and len(q) >= 6 and p is not None and len(p) >= 6:
                    w.writerow([f"{self.clock.now():.6f}"] + q[:6] + p[:6])

                next_t += self.dt
                sleep_dt = next_t - time.time()
                if sleep_dt > 0:
                    time.sleep(sleep_dt)
                    
# ============================================================
# 5️⃣ 로봇 매니저 (기존 그대로)
# ============================================================
class RobotManager:
    def __init__(self, address="192.168.0.100"):
        self.address = address
        self.robot = None
        self.logger = logging.getLogger(__name__)

    def __enter__(self):
        tools.SetDefaultLogger(logging.INFO, f'{pathlib.Path(__file__).stem}.log')
        self.robot = initializer.RobotWithTools()
        self.robot.__enter__()
        self.robot.Connect(address=self.address, disconnect_on_exception=False)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.robot and self.robot.IsConnected():
            try:
                if self.robot.GetStatusRobot().error_status:
                    self.logger.info('Robot had an error, resetting...')
                    self.robot.ResetError()
                    self.robot.ResumeMotion()
            except Exception as e:
                self.logger.warning(f'Error check/clear failed: {e}')
            try:
                self.robot.DeactivateRobot()
            except Exception as e:
                self.logger.warning(f'Deactivate failed: {e}')
        if self.robot:
            self.robot.__exit__(exc_type, exc_value, traceback)

    def setup(self):
        self.logger.info('Activating and homing robot...')
        initializer.reset_sim_mode(self.robot)
        initializer.reset_motion_queue(self.robot, activate_home=True)
        initializer.reset_vacuum_module(self.robot)
        self.robot.WaitHomed()
        self.robot.SetCartLinVel(100)
        self.robot.SetJointVel(0.5)
        self.robot.SetBlending(50)
        self.robot.WaitIdle(30)

    def move_angle_points(self, points):
        if tools.robot_model_is_meca500(self.robot.GetRobotInfo().robot_model):
            self.robot.SetConf(1, 1, 1)
            for idx, (x, y, z, a, b, r) in enumerate(points):
                self.robot.MoveJoints(x, y, z, a, b, r)
                self.robot.WaitIdle(60)
        else:
            raise mdr.MecademicException("Unsupported robot model")
        
    def move_EE_points(self, points): 
        if tools.robot_model_is_meca500(self.robot.GetRobotInfo().robot_model): 
            self.robot.SetConf(1, 1, 1) 
            for idx, (x, y, z, a, b, r) in enumerate(points): 
                self.logger.info(f'Moving to point {idx+1}: ({x}, {y}, {z})') 
                self.robot.MovePose(x, y, z, a, b, r) 
                self.robot.WaitIdle(60) 
        else: 
            raise mdr.MecademicException( f'Unsupported robot model: {self.robot.GetRobotInfo().robot_model}' ) 
    
        
# ============================================================
# 6️⃣ 메인 함수
# ============================================================
def main():
    args = parse_args()

    global OUTPUT_DIR
    if args.run_tag:
        OUTPUT_DIR = f"./dataset/ZED_Captures_{args.run_tag}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    start_event = threading.Event()
    stop_event = threading.Event()
    clock = GlobalClock()
    clock.start()

    writer = AsyncImageWriter()
    writer.start()

    oak_mxid = "1944301011169A4800"
    oak_thread = threading.Thread(
        target=run_oak_capture,
        args=(oak_mxid, "view5_oak", start_event, stop_event, clock, writer),
        daemon=True
    )

    cameras = [
        ZedCamera(41182735, "view1", start_event, stop_event, clock, writer),
        ZedCamera(49429257, "view2", start_event, stop_event, clock, writer),
        ZedCamera(44377151, "view3", start_event, stop_event, clock, writer),
        ZedCamera(49045152, "view4", start_event, stop_event, clock, writer),
    ]

    for cam in cameras:
        cam.init_camera()
        
    while not all(cam.ready for cam in cameras):
        print("Waiting for all cameras to be ready...")
        time.sleep(1)
        
    oak_thread.start()
    for cam in cameras:
        cam.start()
     
    print("Starting data capture in 3 seconds...")
    time.sleep(1)
    
    start_event.set()

    try:
        if args.robot == "on":
            with RobotManager() as manager:
                manager.setup()
                sampler_csv = os.path.join(OUTPUT_DIR, f"robot_rt_{clock.now():.3f}.csv")
                sampler = RtSampler(manager.robot, sampler_csv, clock, rate_hz=100)
                sampler.start()

            # manager.move_EE_points([
            #     # (245.330857, 71.314158, 70.579333, -174.999591, 14.169026, 161.421036)
            #     # (218.546433, -131.00556, 70, 172.224203, 13.788987, -151.514115)
            #     # (151.111573, 83.935818, 62.537654, -179.385704, 1.156863, 152.029116)
            # ])

            manager.move_angle_points([
                (-0.338223, 1.107869, -2.314018, -0.304322, 70.844049, -2.447558),
                (1.439584, 7.482698, 5.040527, -0.815003, 59.736926, -0.392272)
            ])

            # 로봇 동작 종료 직후: 카메라 먼저 정지 → 샘플러 종료
            stop_event.set()
            sampler.stop()
            sampler.join()

            manager.move_angle_points([
                (-0.338223, 1.107869, -2.314018, -0.304322, 70.844049, -2.447558)
            ])

            noise = random.randint(-20, 20)
            manager.move_EE_points([
                (190.0+noise, 0.0+noise, 308.0+noise, 0.0+noise, 90.0+noise, 0.0+noise)
            ])

    finally:
        for cam in cameras:
            cam.join()
        print("Data collection finished.")


if __name__ == "__main__":
    main()

# white block:
#     joint angles: 15.991698, 65.637785, -23.417432, 1.460125, 32.800052, -3.088784
#     EE pose: 245.330857, 71.314158, 70.579333, -174.999591, 14.169026, 161.421036
