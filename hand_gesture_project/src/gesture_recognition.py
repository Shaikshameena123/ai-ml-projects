import mediapipe as mp

mp_hands = mp.solutions.hands

def is_thumbs_up(landmarks):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP].y
    thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP].y
    thumb_mcp = landmarks[mp_hands.HandLandmark.THUMB_MCP].y
    index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
    
    return thumb_tip < thumb_ip < thumb_mcp < index_mcp
