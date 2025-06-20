
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import os
from reid import ReIDModel  # Make sure this import matches your project structure

# --- Load model ---
model = ReIDModel(
    num_parts=6,
    feat_dim=2048,
    num_classes=751,  # adjust to match your training setup
    use_ca=True,
    use_sa=True,
    use_pp=True,
    use_tr=True
)
model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
model.eval()

# --- Define preprocessing ---
IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD = [0.229, 0.224, 0.225]
preprocess = transforms.Compose([
    transforms.Resize((256,128)),
    transforms.ToTensor(),
    transforms.Normalize(IMGNET_MEAN, IMGNET_STD)
])

# --- Load query and gallery images ---
def load_image(path):
    img = Image.open(path).convert('RGB')
    return preprocess(img).unsqueeze(0)

def extract_descriptor(image_tensor):
    with torch.no_grad():
        feat = model(image_tensor, return_feats=True)
    return feat.squeeze(0).numpy()

# Example usage
query_path = 'market/query/0542_c4s3_008073_00.jpg'
gallery_folder = 'market/bounding_box_test'

query_img = load_image(query_path)
query_feat = extract_descriptor(query_img)

# Compute distance to gallery
gallery_feats = []
gallery_imgs = []
gallery_paths = []

for fname in os.listdir(gallery_folder):
    if not fname.lower().endswith('.jpg'): continue
    path = os.path.join(gallery_folder, fname)
    img = load_image(path)
    feat = extract_descriptor(img)
    gallery_feats.append(feat)
    gallery_imgs.append(np.array(Image.open(path)))
    gallery_paths.append(path)

gallery_feats = np.stack(gallery_feats)
dists = np.linalg.norm(gallery_feats - query_feat, axis=1)
top_k = np.argsort(dists)[:5]

# --- Plot results ---
query_disp = np.array(Image.open(query_path))
fig, axs = plt.subplots(1, 6, figsize=(18,4))
axs[0].imshow(query_disp)
axs[0].set_title('Query')
axs[0].axis('off')

for i, idx in enumerate(top_k):
    axs[i+1].imshow(gallery_imgs[idx])
    axs[i+1].set_title(f'Rank-{i+1}')
    axs[i+1].axis('off')

plt.tight_layout()
plt.savefig('retrieval_examples.png')
plt.show()
