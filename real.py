import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from reid import ReIDModel  # Replace with actual import from your model definition

# Initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = ReIDModel(num_parts=6, feat_dim=2048, num_classes=751,
                  use_ca=True, use_sa=True, use_pp=True, use_tr=True)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.to(device)
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((256,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])

# Memory bank for identities
memory = []
id_counter = 0
SIM_THRESHOLD = 0.6

def cosine_sim(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    return np.dot(a, b)

def assign_id(feat):
    global id_counter
    for entry in memory:
        if cosine_sim(feat, entry['feature']) > SIM_THRESHOLD:
            return entry['id']
    memory.append({'id': id_counter, 'feature': feat})
    id_counter += 1
    return id_counter - 1

# Start webcam
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret: break

    # Simulated person crop (you should integrate YOLOv8 or any detector here)
    h, w = frame.shape[:2]
    crop = frame[int(h*0.2):int(h*0.9), int(w*0.3):int(w*0.7)]
    image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model(input_tensor, return_feats=True).cpu().numpy().flatten()
    person_id = assign_id(feat)

    # Draw on original frame
    cv2.rectangle(frame, (int(w*0.3), int(h*0.2)), (int(w*0.7), int(h*0.9)), (0,255,0), 2)
    cv2.putText(frame, f"ID: {person_id}", (int(w*0.3), int(h*0.2)-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imshow("Real-time Person Re-ID", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
