import os
import time
import cv2
import depthai as dai

output_dir = "./oak_images"
os.makedirs(output_dir, exist_ok=True)

mxid = "1944301011169A4800" # OAK-D Pro

# 1. Warm-up을 위해 건너뛸 프레임 수 지정
WARMUP_FRAMES = 10 
frame_count = 0

# Create pipeline
with dai.Pipeline() as pipeline:
    cam = pipeline.create(dai.node.Camera).build()
    cam.initialControl.setManualFocus(110) # 0..255 (far..near)
    videoQueue = cam.requestOutput((1920,1200)).createOutputQueue()

    # Connect to device and start pipeline
    pipeline.start()
    print("Pipeline started. Warming up the camera...")

    while pipeline.isRunning():
        videoIn = videoQueue.get()
        assert isinstance(videoIn, dai.ImgFrame)
        frame = videoIn.getCvFrame()

        # 2. Warm-up 로직: 지정된 수의 프레임이 지날 때까지는 저장을 건너뜁니다.
        frame_count += 1
        if frame_count <= WARMUP_FRAMES:
            if frame_count == 1:
                print("Initial frame received. Waiting for stabilization...")
            # 진행 상황을 10프레임마다 표시 (선택사항)
            if frame_count % 10 == 0:
                print(f"Warm-up in progress... {frame_count}/{WARMUP_FRAMES}")
            if frame_count == WARMUP_FRAMES:
                print("Warm-up complete. Starting capture.")
            continue # 저장 로직을 건너뛰고 다음 프레임으로 이동

        # 3. Warm-up이 끝난 후에만 이미지 저장 로직 실행
        t = time.time()
        filename = os.path.join(output_dir, f"oak_{mxid}_{t:.3f}.jpg")
        if not cv2.imwrite(filename, frame):
            print(f"OAK camera {mxid} - Failed to save image.")
            
        if cv2.waitKey(1) == ord("q"):
            break

print("Capture finished.")

# import depthai as dai

# # 연결된 모든 OAK 장치 정보 가져오기
# available_devices = dai.Device.getAllAvailableDevices()

# if not available_devices:
#     print("OAK 카메라를 찾을 수 없습니다. USB 연결 상태를 확인해주세요.")
# else:
#     print(f"총 {len(available_devices)}개의 OAK 카메라를 찾았습니다:")
#     for device_info in available_devices:
#         # getMxId() 메서드 대신 mxid 속성을 직접 사용하도록 수정
#         print(f"  - MXID: {device_info}, 상태: {device_info.state}")