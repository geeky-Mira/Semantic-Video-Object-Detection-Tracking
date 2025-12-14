import streamlit as st
import cv2
import torch
import clip
from PIL import Image
import tempfile
import os
from ultralytics import YOLO

# --- UI SETUP ---
st.set_page_config(page_title="YOLOv8 + CLIP Tracker", layout="wide")
st.title("👁️ AI Object Search & Tracking")

search_threshold = st.sidebar.slider("Search Similarity Threshold", 0.0, 1.0, 0.25, 0.01)
confidence_threshold = st.sidebar.slider("YOLO Confidence Threshold", 0.0, 1.0, 0.30, 0.05)
search_query = st.sidebar.text_input("🔍 Search Object (e.g., red car)")
use_clip = st.sidebar.checkbox("Enable CLIP Search")

# --- LOAD MODELS (cached) ---
@st.cache_resource
def load_models():
    yolo = YOLO('yolov8s.pt')  
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    return yolo, clip_model, preprocess, device

yolo_model, clip_model, clip_preprocess, device = load_models()

# --- UPLOAD ---
uploaded_file = st.file_uploader("Upload Video (mp4, avi, mov)", type=['mp4','avi','mov'])

if uploaded_file:
    # Write to temp file
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(uploaded_file.read())
    video_path = tmp.name

    # Ensure we only process once per upload
    if "processed" not in st.session_state:
        st.session_state.processed = False

    if not st.session_state.processed:
        cap = cv2.VideoCapture(video_path)
        
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)

        # Write initial AVI (unencoded) then transcode
        avi_path = "temp_output.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(avi_path, fourcc, fps, (width, height))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress = st.progress(0)
        status_text = st.empty()

        # Precompute CLIP text feature
        text_features = None
        if use_clip and search_query:
            token = clip.tokenize([search_query]).to(device)
            with torch.no_grad():
                text_features = clip_model.encode_text(token)
                text_features /= text_features.norm(dim=-1, keepdim=True)

        frame_index = 0
        status_text.text("Processing video…")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO + track & clip
            results = yolo_model.track(frame, conf=confidence_threshold, persist=True, tracker="bytetrack.yaml")
            if results and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                for (x1, y1, x2, y2) in boxes:
                    is_match = False
                    if use_clip and text_features is not None:
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                            inp = clip_preprocess(pil).unsqueeze(0).to(device)
                            with torch.no_grad():
                                img_feat = clip_model.encode_image(inp)
                                img_feat /= img_feat.norm(dim=-1, keepdim=True)
                                sim = float(img_feat @ text_features.T)
                                is_match = sim >= search_threshold

                    if is_match:
                        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
                        cv2.putText(frame, search_query, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            writer.write(frame)
            frame_index += 1
            progress.progress(frame_index / total_frames)
            status_text.text(f"Processing… {int((frame_index/total_frames)*100)}%")

        cap.release()
        writer.release()

        # Transcode to H.264 MP4 so browser can play it
        mp4_path = "processed_output.mp4"
        os.system(f"ffmpeg -i {avi_path} -vcodec libx264 -preset fast {mp4_path} -y")

        st.session_state.processed = True
        status_text.text("Processing complete!")

    # --- SHOW RESULT ---
    if st.session_state.processed:
        st.subheader("🎉 Processed Video")
        with open("processed_output.mp4", "rb") as f:
            video_bytes = f.read()
            st.video(video_bytes, format="video/mp4")
            st.download_button(
                "📥 Download Processed Video",
                data=video_bytes,
                file_name="tracked_output.mp4",
                mime="video/mp4"
            )
