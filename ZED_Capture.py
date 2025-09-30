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

import pyzed.sl as sl
import mecademicpy.robot as mdr
import mecademicpy.robot_initializer as initializer
import mecademicpy.tools as tools

# 저장 폴더
OUTPUT_DIR = "./dataset/ZED_Captures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-tag", default=None, help="출력 폴더 접미사 (예: 20th)")
    return p.parse_args()


class ZedCamera(threading.Thread):
    def __init__(self, serial_number, output_subdir, start_event, stop_event):
        super().__init__(daemon=True)
        self.serial_number = serial_number
        self.output_dir = os.path.join(OUTPUT_DIR, output_subdir)
        os.makedirs(self.output_dir, exist_ok=True)

        self.zed = sl.Camera()
        self.runtime_params = sl.RuntimeParameters()
        self.start_event = start_event
        self.stop_event = stop_event
        self.ready = False

    def init_camera(self):
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD1200
        init_params.camera_fps = 30
        init_params.set_from_serial_number(self.serial_number)
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL

        if self.zed.open(init_params) == sl.ERROR_CODE.SUCCESS:
            if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
                self.ready = True
            print(f"Camera {self.serial_number} initialized")
        else:
            print(f"Failed to open camera {self.serial_number}")
            sys.exit(1)

    def run(self):
        if not self.ready:
            print(f"Camera {self.serial_number} not ready. Skipping capture.")
            return

        left_image, right_image = sl.Mat(), sl.Mat()
        self.start_event.wait()
        print(f"Camera {self.serial_number} started capturing")

        while not self.stop_event.is_set():
            if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(left_image, sl.VIEW.LEFT)
                t_left = time.time()
                self.zed.retrieve_image(right_image, sl.VIEW.RIGHT)
                t_right = time.time()

                left_data = left_image.get_data()
                right_data = right_image.get_data()

                left_path = os.path.join(self.output_dir, f"zed_{self.serial_number}_left_{t_left:.3f}.jpg")
                right_path = os.path.join(self.output_dir, f"zed_{self.serial_number}_right_{t_right:.3f}.jpg")

                ok_l = cv2.imwrite(left_path, left_data[:, :, :3])
                ok_r = cv2.imwrite(right_path, right_data[:, :, :3])
                if not (ok_l and ok_r):
                    print(f"Camera {self.serial_number} - Failed to save images")

        self.zed.close()
        print(f"Camera {self.serial_number} stopped")

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
                    self.logger.info('Robot has encountered an error, attempting to clear...')
                    self.robot.ResetError()
                    self.robot.ResumeMotion()
            except Exception as e:
                self.logger.warning(f'Error check/clear failed: {e}')
            try:
                self.robot.DeactivateRobot()
                self.logger.info('Robot is deactivated.')
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
        self.logger.info('Robot is homed and ready.')
        self.robot.SetCartLinVel(100)
        self.robot.SetJointVel(1)
        self.robot.SetBlending(50)
        # self.robot.MoveJoints(*([0] * self.robot.GetRobotInfo().num_joints))
        self.robot.WaitIdle(30)

    def move_points(self, points):
        if tools.robot_model_is_meca500(self.robot.GetRobotInfo().robot_model):
            self.robot.SetConf(1, 1, 1)
            for idx, (x, y, z, a, b, r) in enumerate(points):
                self.logger.info(f'Moving to point {idx+1}: ({x}, {y}, {z})')
                self.robot.MovePose(x, y, z, a, b, r)
                self.robot.WaitIdle(60)
        else:
            raise mdr.MecademicException(
                f'Unsupported robot model: {self.robot.GetRobotInfo().robot_model}'
            )
        self.logger.info('Finished executing move_points.')


class RtSampler(threading.Thread):
    """비-RT 폴링으로 조인트/EE 포즈를 epoch 기준으로 CSV 저장"""
    def __init__(self, robot, out_csv, rate_hz=100):
        super().__init__(daemon=True)
        self.robot = robot
        self.out_csv = out_csv
        self.dt = 1.0 / float(rate_hz)
        self.stop_evt = threading.Event()

    def stop(self):
        self.stop_evt.set()

    def run(self):
        with open(self.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "epoch_s",
                "joint_angle_1","joint_angle_2","joint_angle_3",
                "joint_angle_4","joint_angle_5","joint_angle_6",
                "EE_x","EE_y","EE_z","EE_a","EE_b","EE_r"
            ])
            next_t = time.time()
            while not self.stop_evt.is_set():
                t = time.time()

                # 조인트
                q = None
                for name in ("GetJoints", "GetJointPos", "GetJointAngles"):
                    fn = getattr(self.robot, name, None)
                    if callable(fn):
                        try:
                            q = list(fn())
                            break
                        except Exception:
                            pass

                # EE 포즈
                p = None
                for name in ("GetPose", "GetPoseXYZABC", "GetCartesianPose"):
                    fn = getattr(self.robot, name, None)
                    if callable(fn):
                        try:
                            p = list(fn())
                            break
                        except Exception:
                            pass

                if q is not None and len(q) >= 6 and p is not None and len(p) >= 6:
                    w.writerow([f"{t:.6f}"] + q[:6] + p[:6])

                next_t += self.dt
                sleep_dt = next_t - time.time()
                if sleep_dt > 0:
                    time.sleep(sleep_dt)


def main():
    args = parse_args()

    global OUTPUT_DIR

    if args.run_tag:
        OUTPUT_DIR = f"./dataset/ZED_Captures_{args.run_tag}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    start_event = threading.Event()
    stop_event = threading.Event()

    cameras = [
        ZedCamera(41182735, "view1", start_event, stop_event),
        ZedCamera(49429257, "view2", start_event, stop_event),
        ZedCamera(44377151, "view3", start_event, stop_event),
        ZedCamera(49045152, "view4", start_event, stop_event),
    ]

    for cam in cameras:
        cam.init_camera()
    while not all(cam.ready for cam in cameras):
        print("Waiting for all cameras to be ready...")
        time.sleep(1)
    for cam in cameras:
        cam.start()

    print("Starting data capture in 3 seconds...")
    time.sleep(3)
    start_event.set()

    try:
        with RobotManager() as manager:
            manager.setup()

            sampler_csv = os.path.join(OUTPUT_DIR, f"robot_rt_{time.time():.3f}.csv")
            sampler = RtSampler(manager.robot, sampler_csv, rate_hz=100)
            sampler.start()

            manager.move_points([
                # (245.330857, 71.314158, 70.579333, -174.999591, 14.169026, 161.421036)
                # (218.546433, -131.00556, 70, 172.224203, 13.788987, -151.514115)
                (151.111573, 83.935818, 62.537654, -179.385704, 1.156863, 152.029116)
            ])

            # 로봇 동작 종료 직후: 카메라 먼저 정지 → 샘플러 종료
            stop_event.set()
            sampler.stop()
            sampler.join()
            noise = random.randint(-20, 20)
            manager.move_points([
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
