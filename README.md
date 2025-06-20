# 🚶‍♀️ Attention-Enhanced Part-Aware Person Re-Identification 🔍

This is the implementation of the research project **"Attention-Enhanced Part-Aware Person Re-Identification"**, developed for my Masters' Thesis at the University of Moratuwa. This project explores an attention-based architecture to improve person detection consistency across video frames.

---

## 📌 Overview

In real-world video surveillance, individuals often appear in multiple frames with varying poses, lighting, and partial occlusions. This project introduces a **hybrid model** combining:

- 🎯 **Channel Attention** (SE block)
- 🎯 **Spatial Attention** (CBAM-style)
- 📐 **Horizontal Part-based Pooling**
- 🔁 **Lightweight Transformer Encoder**
- 🧠 **Contextual Descriptor Head**
- 📊 **Cross-Entropy + Triplet Loss**

All components work together to enhance identity discrimination and suppress duplicates across frames.

---

## 🧱 Architecture

![image](https://github.com/user-attachments/assets/87748717-2f50-4a90-a136-01a6f88b55a0)

## 📂 Project Structure

```bash
attention-part-aware-reid/
├── market/                   # Market-1501 dataset<br>
├── reid.py                   # Training entry point<br>
├── reid_visual_inference.py  # Query image inference<br>
├── real.py                   # Real-time video inference<br>
├── best_model.pth            # Saved model weights<br>
└── README.md                 # You are here<br>
```

## 🚀 Getting Started

Follow these steps to set up and run the project locally:

### 📦 1. Clone the Repository

```bash
git clone https://github.com/yourusername/attention-part-aware-reid.git
cd attention-part-aware-reid
```

### 🧰 2. Install Required Dependencies
Make sure Python 3.8+ and pip are installed.

```bash
pip install -r requirements.txt
```

### 📁 3. Prepare the Dataset
Download the Market-1501 dataset their official page.

Extract it and place it inside the datasets/ folder like this:

market/<br>
├── bounding_box_train/<br>
&nbsp; &nbsp;    ├── query/<br>
&nbsp; &nbsp;    └── bounding_box_test/<br>

### 🏋️‍♀️ 4. Train and Evaluate the Model
You can enable/disable components using flags:

```bash
python main.py --use_ca --use_sa --use_pp --use_tr --use_tri
```
This will:

&nbsp; &nbsp; Enable channel and spatial attention<br>
&nbsp; &nbsp; Apply part pooling and Transformer<br>
&nbsp; &nbsp; Use both Cross-Entropy and Triplet Loss<br>

### 🔍 6. Run Inference on a Query Image

```bash
python inference.py --img_path path/to/query.jpg
```
This script will:

&nbsp; &nbsp; Extract features from the image<br>
&nbsp; &nbsp; Compare it with the gallery<br>
&nbsp; &nbsp; Output the top retrieved results<br>
