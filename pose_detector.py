"""
Pose Detection Module
MediaPipe ile pose, hand ve face detection
"""

import cv2
import mediapipe as mp
import numpy as np


class PoseDetector:
    """MediaPipe ile pose algılama - 6 poz: el kaldırma (tek parmak), düşünme, gülümseme, şaşırma, rahat duruş, varsayılan"""
    
    def __init__(self):
        # MediaPipe modüllerini başlat
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Debug bilgileri
        self.debug_info = {
            'mouth_ratio': 0.0,
            'hand_height': 0.0,
            'hands_detected': 0,
            'face_detected': False,
            'smile_score': 0.0,
            'eye_openness': 0.0,
            'fingers_up': []
        }
        
    def detect_pose(self, frame):
        """Frame üzerinde pose detection yapar"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detection'lar
        pose_results = self.pose.process(rgb_frame)
        hand_results = self.hands.process(rgb_frame)
        face_results = self.face_mesh.process(rgb_frame)
        
        # Debug sıfırla
        self.debug_info['hands_detected'] = 0
        self.debug_info['face_detected'] = False
        
        # Sadece ağız bölgesi çiz
        if face_results.multi_face_landmarks:
            self.debug_info['face_detected'] = True
            for face_landmarks in face_results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                h, w = frame.shape[:2]
                
                # Sadece dudak konturları
                self.mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    self.mp_face_mesh.FACEMESH_LIPS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1)
                )
        
        # Eller çiz
        if hand_results.multi_hand_landmarks:
            self.debug_info['hands_detected'] = len(hand_results.multi_hand_landmarks)
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        # Pozu belirle
        pose_name = self._determine_pose(pose_results, hand_results, face_results)
        
        # Debug bilgileri göster
        y_pos = 30
        cv2.putText(frame, f"Eller: {self.debug_info['hands_detected']}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_pos += 30
        cv2.putText(frame, f"Yuz: {'VAR' if self.debug_info['face_detected'] else 'YOK'}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_pos += 30
        cv2.putText(frame, f"Agiz: {self.debug_info['mouth_ratio']:.3f}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_pos += 30
        cv2.putText(frame, f"El Yukseklik: {self.debug_info['hand_height']:.3f}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_pos += 30
        cv2.putText(frame, f"Gulumseme: {self.debug_info['smile_score']:.3f}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_pos += 30
        cv2.putText(frame, f"Goz Aciklik: {self.debug_info['eye_openness']:.3f}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_pos += 30
        fingers_str = str(self.debug_info['fingers_up']) if self.debug_info['fingers_up'] else "[]"
        cv2.putText(frame, f"Parmaklar: {fingers_str}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Poz göster
        cv2.putText(frame, f"Pose: {pose_name}", (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return frame, pose_name
    
    def _determine_pose(self, pose_results, hand_results, face_results):
        """Pozu belirler - öncelik: el kaldırma (tek parmak) > düşünme > gülümseme > şaşırma > rahat duruş > varsayılan"""
        if self._is_raising_hand(pose_results, hand_results):
            return "raising_hand"
        
        if self._is_thinking(pose_results, hand_results, face_results):
            return "thinking"
        
        if self._is_smiling(face_results):
            return "smile"
        
        if self._is_shocking(face_results):
            return "shocking"
        
        if self._is_relaxing(face_results):
            return "relaxing"
        
        return "default"
    
    def _is_raising_hand(self, pose_results, hand_results):
        """El baş hizasından yukarıda VE sadece işaret parmağı açık mı"""
        if not pose_results.pose_landmarks or not hand_results.multi_hand_landmarks:
            self.debug_info['hand_height'] = 0.0
            return False
        
        pose_landmarks = pose_results.pose_landmarks.landmark
        nose_y = pose_landmarks[self.mp_pose.PoseLandmark.NOSE].y
        
        for hand_landmarks in hand_results.multi_hand_landmarks:
            wrist_y = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].y
            height_diff = nose_y - wrist_y
            self.debug_info['hand_height'] = height_diff
            
            # Yükseklik kontrolü (0.01 threshold)
            if height_diff > 0.01:
                # Parmak kontrolü - sadece işaret parmağı açık mı?
                fingers = self._count_fingers(hand_landmarks)
                self.debug_info['fingers_up'] = fingers
                
                # Sadece işaret parmağı açık: [_, 1, 0, 0, 0]
                # Baş parmak açık veya kapalı olabilir
                if fingers[0] == 0 and fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
                    return True
        
        return False
    
    def _is_shocking(self, face_results):
        """Ağız açık mı"""
        if not face_results.multi_face_landmarks:
            self.debug_info['mouth_ratio'] = 0.0
            return False
        
        face_landmarks = face_results.multi_face_landmarks[0].landmark
        
        # Ağız landmark'ları
        upper_lip = face_landmarks[13].y
        lower_lip = face_landmarks[14].y
        forehead = face_landmarks[10].y
        chin = face_landmarks[152].y
        face_height = abs(chin - forehead)
        
        mouth_opening = abs(lower_lip - upper_lip)
        mouth_ratio = mouth_opening / face_height if face_height > 0 else 0
        self.debug_info['mouth_ratio'] = mouth_ratio
        
        return mouth_ratio > 0.15
    
    def _is_smiling(self, face_results):
        """Gülümseme tespiti - ağız köşeleri yukarı kalkmış mı"""
        if not face_results.multi_face_landmarks:
            self.debug_info['smile_score'] = 0.0
            return False
        
        face_landmarks = face_results.multi_face_landmarks[0].landmark
        
        # Ağız landmark'ları
        left_mouth_corner = face_landmarks[61]   # Sağ köşe (kamera ters)
        right_mouth_corner = face_landmarks[291]  # Sol köşe (kamera ters)
        upper_lip_center = face_landmarks[13]
        lower_lip_center = face_landmarks[14]
        
        # Ağız merkezinin y koordinatı
        mouth_center_y = (upper_lip_center.y + lower_lip_center.y) / 2
        
        # Ağız köşelerinin y koordinatları
        left_corner_y = left_mouth_corner.y
        right_corner_y = right_mouth_corner.y
        avg_corner_y = (left_corner_y + right_corner_y) / 2
        
        smile_diff = mouth_center_y - avg_corner_y
        
        # Ağız genişliği (gülümsemede ağız genişler)
        mouth_width = abs(left_mouth_corner.x - right_mouth_corner.x)
        
        # Yüz genişliği referansı için
        face_width = abs(face_landmarks[454].x - face_landmarks[234].x)  # Yanak kemikleri arası
        mouth_width_ratio = mouth_width / face_width if face_width > 0 else 0
        
        # Gülümseme skoru: köşeler yukarıda + ağız geniş
        smile_score = smile_diff + (mouth_width_ratio * 0.5)
        self.debug_info['smile_score'] = smile_score
        
        # Gülümseme eşiği
        return smile_score > 0.20
    
    def _is_thinking(self, pose_results, hand_results, face_results):
        """El ağza/çeneye değiyor mu (thinking pozu)"""
        if not face_results.multi_face_landmarks or not hand_results.multi_hand_landmarks:
            return False
        
        face_landmarks = face_results.multi_face_landmarks[0].landmark
        
        # Ağız bölgesi (üst dudak, alt dudak, çene)
        mouth_points = [
            face_landmarks[13],   # Üst dudak
            face_landmarks[14],   # Alt dudak
            face_landmarks[152],  # Çene
            face_landmarks[0],    # Ağız merkezi
        ]
        
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # El parmaklarının uçları
            finger_tips = [
                hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP],
                hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP],
                hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
            ]
            
            # Herhangi bir parmak ağız bölgesine çok yakın mı?
            for finger_tip in finger_tips:
                for mouth_point in mouth_points:
                    distance = np.sqrt((finger_tip.x - mouth_point.x)**2 + (finger_tip.y - mouth_point.y)**2)
                    
                    # Threshold çok düşük - neredeyse değmeli
                    if distance < 0.08:
                        return True
        
        return False
    
    def _count_fingers(self, hand_landmarks):
        """Her parmağın açık/kapalı olduğunu tespit eder - [Baş, İşaret, Orta, Yüzük, Serçe]"""
        fingers = []
        
        # Parmak uçları ve eklem noktaları
        # Baş parmak: 4 (uç), 3 (IP eklem), 2 (MCP)
        # İşaret: 8 (uç), 6 (PIP), 5 (MCP)
        # Orta: 12 (uç), 10 (PIP), 9 (MCP)
        # Yüzük: 16 (uç), 14 (PIP), 13 (MCP)
        # Serçe: 20 (uç), 18 (PIP), 17 (MCP)
        
        # Baş parmak - x koordinatını karşılaştır (özel durum)
        thumb_tip = hand_landmarks.landmark[8]
        thumb_ip = hand_landmarks.landmark[6]
        thumb_mcp = hand_landmarks.landmark[5]
        
        # Baş parmak açık mı? (ucun x'i IP'den daha uzakta)
        if abs(thumb_tip.x - thumb_mcp.x) > abs(thumb_ip.x - thumb_mcp.x):
            fingers.append(1)  # Açık
        else:
            fingers.append(0)  # Kapalı
        
        # Diğer 4 parmak - y koordinatını karşılaştır
        finger_tips = [4, 12, 16, 20]  # İşaret, Orta, Yüzük, Serçe uçları
        finger_pips = [3, 10, 14, 18]  # PIP eklemleri
        
        for tip_id, pip_id in zip(finger_tips, finger_pips):
            tip = hand_landmarks.landmark[tip_id]
            pip = hand_landmarks.landmark[pip_id]
            
            # Parmak açık mı? (uç PIP'den daha yukarıda = y değeri daha küçük)
            if tip.y < pip.y:
                fingers.append(1)  # Açık
            else:
                fingers.append(0)  # Kapalı
        
        return fingers
    
    def _is_relaxing(self, face_results):
        """Rahat duruş tespiti - gözler kapalı mı (Eye Aspect Ratio)"""
        if not face_results.multi_face_landmarks:
            self.debug_info['eye_openness'] = 0.0
            return False
        
        face_landmarks = face_results.multi_face_landmarks[0].landmark
        
        # Sol göz landmark'ları
        left_eye_top = face_landmarks[159]      # Üst kapak
        left_eye_bottom = face_landmarks[145]   # Alt kapak
        left_eye_left = face_landmarks[33]      # Sol köşe
        left_eye_right = face_landmarks[133]    # Sağ köşe
        
        # Sağ göz landmark'ları
        right_eye_top = face_landmarks[386]     # Üst kapak
        right_eye_bottom = face_landmarks[374]  # Alt kapak
        right_eye_left = face_landmarks[362]    # Sol köşe
        right_eye_right = face_landmarks[263]   # Sağ köşe
        
        # Sol göz açıklığı (dikey mesafe / yatay mesafe)
        left_vertical = abs(left_eye_top.y - left_eye_bottom.y)
        left_horizontal = abs(left_eye_left.x - left_eye_right.x)
        left_ratio = left_vertical / left_horizontal if left_horizontal > 0 else 0
        
        # Sağ göz açıklığı (dikey mesafe / yatay mesafe)
        right_vertical = abs(right_eye_top.y - right_eye_bottom.y)
        right_horizontal = abs(right_eye_left.x - right_eye_right.x)
        right_ratio = right_vertical / right_horizontal if right_horizontal > 0 else 0
        
        # Ortalama göz açıklığı (Eye Aspect Ratio - EAR)
        eye_openness = (left_ratio + right_ratio) / 2
        self.debug_info['eye_openness'] = eye_openness
        
        # Gözler kapalıysa EAR çok düşük olur (threshold: 0.15)
        return eye_openness < 0.15
    
    def release(self):
        """Kaynakları serbest bırak"""
        self.pose.close()
        self.hands.close()
        self.face_mesh.close()
