import time
import cv2
import numpy as np


class ObjectDetection():
    #Frame RGB -> HSV
    def frames_rgb_to_hsv(self, frames):
        g_frames = []
        for frame in frames:
            g_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
        return np.array(g_frames, dtype='uint8')
    #Frame HSV -> RGB
    def frames_hsv_to_rgb(self, frames):
        g_frames = []
        for frame in frames:
            g_frames.append(cv2.cvtColor(frame, cv2.COLOR_HSV2BGR))
        return np.array(g_frames, dtype='uint8')
    #Delete Background
    def delete_background(self, frames):
        avg = np.zeros_like(frames[0], dtype='uint64')
        i=0
        # 프레임들의 평균 값 추출
        for frame in frames:
            i+=1
            avg += frame
        # 배경(프레임들의 평균 값) 제거
        avg = np.array(avg / i, dtype='uint8') 
        for j, frame in enumerate(frames):
            frames[j] =  frame - avg
        return frames
    
    def delete_background_MOG(self, frames):
        fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=50, history=500, detectShadows=False)
        fg_masks = [fgbg.apply(frame) for frame in frames]
        processed_frames = frames.copy()
        
        for i in range(len(frames)):
            processed_frames[i][fg_masks[i] == 0] = 0
            processed_frames[i] -= fg_masks[i]
        
        return processed_frames

    #Thresholding video
    def thresholding(self, frames, thres_min, thres_max):
        for i, frame in enumerate(frames):
            _, frames[i] = cv2.threshold(frame, thres_min, 255, cv2.THRESH_TOZERO)
            _, frames[i] = cv2.threshold(frame, thres_max, 0, cv2.THRESH_TOZERO_INV)
        return frames
    
    def connect_components(self, frames, original_frames, area_min):
        for k, frame in enumerate(frames):
            cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(frame[:,:,2])
            
            components = []
            
            for i in range(1, cnt):
                if stats[i][0] == 0 and stats[i][1] == 0:
                    continue
                if np.any(np.isnan(centroids[i])):
                    continue
                (x, y, w, h, area) = stats[i]
                if area < area_min:
                    continue
                
                components.append((x, y, w, h, area))
            
            merged_components = self.merge_rectangles(components)
            
            for component in merged_components:
                (x, y, w, h, area) = component
                cv2.rectangle(original_frames[k], (x, y), (x+w, y+h), (0,0,255), 2)
        
        return original_frames

    def merge_rectangles(self, components):
        merged_components = []
        
        sorted_components = sorted(components, key=lambda c: c[4], reverse=True)
        
        while sorted_components:
            current = sorted_components.pop(0)
            merged = False
            
            for i in range(len(sorted_components)):
                overlap_area = self.calculate_overlap(current, sorted_components[i])
                
                if overlap_area > 0:
                    merged = True
                    current = self.merge_two_rectangles(current, sorted_components[i])
                    sorted_components.pop(i)
                    break
            
            if not merged:
                merged_components.append(current)
        
        return merged_components

    def calculate_overlap(self, rect1, rect2):
        x1 = max(rect1[0], rect2[0])
        y1 = max(rect1[1], rect2[1])
        x2 = min(rect1[0] + rect1[2], rect2[0] + rect2[2])
        y2 = min(rect1[1] + rect1[3], rect2[1] + rect2[3])
        
        overlap_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        return overlap_area


    def merge_two_rectangles(self, rect1, rect2):
        x = min(rect1[0], rect2[0])
        y = min(rect1[1], rect2[1])
        w = max(rect1[0] + rect1[2], rect2[0] + rect2[2]) - x
        h = max(rect1[1] + rect1[3], rect2[1] + rect2[3]) - y
        
        return (x, y, w, h, w * h)
    
    def connect_components_hand(self, frames, original_frames, area_min):
        for k, frame in enumerate(frames):
            cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(frame[:,:,2])
            
            if cnt > 1:
                # 모든 박스의 좌표 정보 수집
                boxes = []
                for i in range(1, cnt):
                    (x, y, w, h, area) = stats[i]
                    if area >= area_min:
                        boxes.append((x, y, x+w, y+h))
                
                # 모든 박스를 포함하는 최상위 박스 계산
                top_left_x = min(box[0] for box in boxes)
                top_left_y = min(box[1] for box in boxes)
                bottom_right_x = max(box[2] for box in boxes)
                bottom_right_y = max(box[3] for box in boxes)
                
                # 최상위 박스 그리기
                cv2.rectangle(original_frames[k], (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 0, 255), 2)
        
        return original_frames
    #Hand Detection
    def hand_detection(self, frames):
        hsv_frames = self.frames_rgb_to_hsv(frames)
        back_deleted_frames = self.delete_background(hsv_frames[:,:,:,2].copy())    # Intensity에서 배경 제거]
        # return back_deleted_frames
        hsv_frames[:,:,:,2] = self.thresholding(back_deleted_frames,7, 200)      # Thresholding

        hsv_frames1 = self.frames_hsv_to_rgb(hsv_frames)
        # return hsv_frames
        connected_frames = self.connect_components_hand(hsv_frames1, frames, 15000)        # Component 연결

        # return hsv_frames
        return back_deleted_frames, hsv_frames,connected_frames
        #Vehicle Detection
    def vehicle_detection(self, frames):
        hsv_frames = self.frames_rgb_to_hsv(frames)
        back_deleted_frames = self.delete_background_MOG(hsv_frames[:,:,:,2].copy())
        for i, frame in enumerate(back_deleted_frames):
            back_deleted_frames[i] = cv2.medianBlur(frame,3)
        # return back_deleted_frames
        # hsv_frames[:,:,:,2] = self.thresholding(back_deleted_frames, 14, 160)      # Thresholding
        hsv_frames[:,:,:,2] = back_deleted_frames
        # hsv_frames[:,:,:,2] = back_deleted_frames
        hsv_frames1 = self.frames_hsv_to_rgb(hsv_frames)
        connected_frames = self.connect_components(hsv_frames1, frames, 800)

        return back_deleted_frames, hsv_frames1,connected_frames
      
    
def get_video_frames(filename):
        address =  filename
        capture = cv2.VideoCapture(address)
        frames = []
        while capture.isOpened():
            run, frame = capture.read()
            if not run:
                break
            img = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
            frames.append(img)
        capture.release()
        return np.array(frames, dtype='uint8')
    
def play_selected_video(video1, video2):

    cv2.namedWindow("video_1", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('video_2', cv2.WINDOW_AUTOSIZE)
   

    # 창의 위치와 크기를 조정합니다.
    cv2.moveWindow("video_1",0, 0)
    cv2.moveWindow('video_2', 0, 700)

    for i in range(len(video1)):
        cv2.imshow("video_1", video1[i])
        cv2.imshow('video_2', video2[i])
        # time.sleep(0.1)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    while True:
        sel = input("실행할 영상을 선택하세요 1:(hand detection) 2:(vehicle detection) 3:(vehicle detection): ")
        if sel != "1" and sel != "2" and sel != "3":
            print("잘못된 입력입니다.")
        else:
            break
    
    video = get_video_frames("./실습동영상/영상"+sel+".mp4")
    # video = get_video_frames("./실습동영상/영상"+sel+"_short.mp4") // 영상 길이 줄여서 테스트용
    if sel == "1":
        back_deleted_frames, hsi_delete, processed_video = ObjectDetection().hand_detection(video)
    else:
        back_deleted_frames, hsi_delete, processed_video = ObjectDetection().vehicle_detection(video)
    

    # 영상 출력
    play_selected_video(back_deleted_frames, processed_video)
