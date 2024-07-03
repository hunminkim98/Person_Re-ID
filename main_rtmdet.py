import os
import random
import cv2
import numpy as np
from tracker import Tracker
from rtmdet import RTMDet

# RTMDet 모델 설정
onnx_model_path = r'C:\Users\82102\One\바탕 화면\personreid\Person_Re-ID\model_data\rtmdet-088d2e.onnx'
model = RTMDet(onnx_model_path, model_input_size=(640, 640))

video_path = r"D:\reid\basketball.mp4"
video_out_path = r'D:\reid\basketball_out.mp4'
save_dir = r'D:\reid\images'
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
cap_out = cv2.VideoWriter(video_out_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (frame.shape[1], frame.shape[0]))

tracker = Tracker()
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

detection_threshold = 0.4  # 이 값을 조정하여 필터링 임계값 변경
cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
frame_count = 0

tracking_ids = set()

print("프로그램 시작.")
print("'t'를 눌러 추적할 인물 ID를 입력하세요. (형식: person1, person2 등)")
print("'q'를 눌러 종료합니다.")
print("'+'를 눌러 detection threshold를 증가시킵니다.")
print("'-'를 눌러 detection threshold를 감소시킵니다.")

while ret:
    detections = model(frame)
    
    # threshold 이상의 값을 가진 bounding box만 선택
    filtered_detections = []
    for det in detections:
        if len(det) >= 5 and det[4] > detection_threshold:
            filtered_detections.append(det)
    
    print(f"Number of detections: {len(detections)}")
    print(f"Number of filtered detections: {len(filtered_detections)}")
    
    tracker.update(frame, filtered_detections)

    for track in tracker.tracks:
        bbox = track.bbox
        x1, y1, x2, y2 = map(int, bbox)
        track_id = track.track_id
        person_id = track.person_id

        cv2.rectangle(frame, (x1, y1), (x2, y2), (colors[track_id % len(colors)]), 3)
        cv2.putText(frame, f"{person_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (colors[track_id % len(colors)]), 2)

        if person_id in tracking_ids:
            person_dir = os.path.join(save_dir, person_id)
            os.makedirs(person_dir, exist_ok=True)
            
            crop_img = frame[y1:y2, x1:x2]
            
            if crop_img.size != 0:
                save_path = os.path.join(person_dir, f"{person_id}_{frame_count}.jpg")
                try:
                    success = cv2.imwrite(save_path, crop_img)
                    if success:
                        print(f"프레임 {frame_count}: {person_id} 크롭 이미지 저장됨 - {save_path}")
                    else:
                        print(f"프레임 {frame_count}: {person_id} 크롭 이미지 저장 실패 - {save_path}")
                except Exception as e:
                    print(f"크롭 이미지 저장 중 오류 발생: {e}")
            else:
                print(f"프레임 {frame_count}: {person_id} 크롭 영역이 유효하지 않음")

    cv2.putText(frame, f"Total Persons: {len(tracker.tracks)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Threshold: {detection_threshold:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow("Tracking", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("프로그램 종료")
        break
    elif key == ord('t'):
        new_tracking_id = input("추적할 person_id를 입력하세요 (예: person1, person2 등): ")
        if new_tracking_id.startswith("person") and new_tracking_id[6:].isdigit():
            tracking_ids.add(new_tracking_id)
            print(f"{new_tracking_id}의 추적 및 데이터 저장을 시작합니다.")
        else:
            print("잘못된 형식입니다. 'person'으로 시작하고 그 뒤에 숫자가 오는 형식이어야 합니다.")
    elif key == ord('+'):
        detection_threshold = min(1.0, detection_threshold + 0.05)
        print(f"Detection threshold increased to {detection_threshold:.2f}")
    elif key == ord('-'):
        detection_threshold = max(0.0, detection_threshold - 0.05)
        print(f"Detection threshold decreased to {detection_threshold:.2f}")

    cap_out.write(frame)
    frame_count += 1
    ret, frame = cap.read()

cap.release()
cap_out.release()
cv2.destroyAllWindows()

print(f"총 {frame_count}개의 프레임이 처리되었습니다.")
print(f"저장된 이미지는 {save_dir} 폴더 내의 각 person_id 폴더에 있습니다.")