# Semantic Video Object Detection & Tracking (YOLOv8 + CLIP + OpenCV + Streamlit)

A Python project that combines **deep learning based object detection and tracking** with **semantic search using CLIP**, wrapped in an **interactive Streamlit application** that processes videos and highlights objects matching natural language queries.

This tool allows users to upload a video, enter a text query (for example, *"red car"*), and get a processed output video that only highlights objects semantically matching the query.

---

## 🧠 Project Overview

This project demonstrates a complete, end-to-end computer vision pipeline:

- Object detection using **YOLOv8**
- Multi-object tracking using **ByteTrack**
- Semantic object filtering using **CLIP embeddings**
- Video processing using **OpenCV**
- Interactive UI using **Streamlit**

It is designed as a practical, production-style demo suitable for portfolios, internships, and junior AI / computer vision roles.

---

## ✨ Key Features

- 📹 Upload videos (MP4 / AVI / MOV)
- 🔍 Search objects using natural language (CLIP)
- 📦 YOLOv8 detection with persistent tracking
- 🧠 CLIP-based semantic similarity filtering
- 📊 Real-time progress indicator during processing
- ✅ Browser-playable MP4 output (H.264)
- 📥 Download processed video without re-running the app
- ⚡ GPU-accelerated inference (Colab-friendly)

---

## 🛠️ Tech Stack

- **Python**
- **OpenCV** – video I/O and drawing
- **YOLOv8 (Ultralytics)** – object detection
- **ByteTrack** – multi-object tracking
- **CLIP (OpenAI)** – text-image semantic matching
- **Streamlit** – interactive UI
- **FFmpeg** – MP4 video encoding

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPO>.git
cd <YOUR_REPO>
```

(Optional) Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Ensure FFmpeg is available:

```bash
ffmpeg -version
```

---

## 🚀 Running the App

Start the Streamlit application:

```bash
streamlit run app.py
```

If running in **Google Colab**, expose the app using **ngrok** or a similar tunneling service.

---

## 🎯 How It Works

1. User uploads a video.
2. YOLOv8 detects objects in each frame.
3. ByteTrack assigns consistent IDs across frames.
4. Detected object crops are encoded using CLIP.
5. Text query is encoded using CLIP.
6. Cosine similarity is used to filter relevant objects.
7. Bounding boxes are drawn only for matching objects.
8. Output video is encoded as MP4 and displayed in browser.

---

## 📁 Project Structure

```
.
├── app.py                 # Streamlit application
├── notebooks/             # Experimentation / demo notebooks
│   └── demo.ipynb
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── assets/                # Screenshots / GIFs
└── sample_videos/         # Example input videos
```

---

## 📌 Use Cases

- Semantic video search
- Intelligent video analytics
- Surveillance & traffic analysis demos
- Computer vision portfolio project
- Learning multi-model AI pipelines

---

## 🚧 Limitations

- CLIP similarity is computed per object crop, not per pixel
- Performance depends on video length and GPU availability
- Not intended for real-time processing (batch-oriented)

---

## 🔮 Future Improvements

- FastAPI backend for REST APIs
- Dockerized deployment
- Real-time webcam inference
- Multi-query object filtering
- Model quantization for faster inference

---

## 🙌 Acknowledgements

- Ultralytics YOLOv8
- OpenAI CLIP
- Streamlit Community

---

*This project is intended for educational and portfolio use.*
