import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import SamProcessor
from torch.utils.data import DataLoader
from torch.optim import Adam
import monai
from statistics import mean
import torch
import random
import logging
from torch.nn import functional as F
from transformers import SamModel
from torchvision.transforms.functional import resize, to_pil_image, rotate, hflip, vflip
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = "8"
DATASET_PATH = "/tmp/ahsan/sqfs/storage_local/datasets/public/ufpr-alpr"
torch.cuda.empty_cache()
class UFPR_ALPR_Dataset:
    def __init__(self, root, split="training"):
        self.data_dir = os.path.join(root, split)
        self.image_list = self.build_image_list()
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        logger.info(f"Dataset initialized with {len(self.image_list)} images.")

    def build_image_list(self):
        image_list = []
        for i in range(len(os.listdir(self.data_dir))):
            path = os.path.join(self.data_dir, os.listdir(self.data_dir)[i])
            files = os.listdir(path)
            for j in range(len(files)):
                if os.path.splitext(files[j])[-1] == ".png":
                    image_list.append(os.path.join(path, files[j]))
        return image_list

    def load_image(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img = img.astype(np.uint8)
        return img

    def load_annotations(self, path):
        file = path.replace("png", "txt")
        with open(file, "r") as f:
            data = f.read()
        coordinates = data.splitlines()[7].split(":")[1].strip().split(" ")
        corners = []
        for coord in coordinates:
            x, y = coord.split(',')
            corners.append((int(x), int(y)))
        corners = np.array(corners, dtype=np.float32)
        x_min = np.min(corners[:, 0])
        y_min = np.min(corners[:, 1])
        x_max = np.max(corners[:, 0])
        y_max = np.max(corners[:, 1])

        label = np.array([x_min, y_min, x_max, y_max])
        label = label.reshape((1, 4))
        return label

    def plate_mask(self, img, annot):
        h, w = img.shape[0], img.shape[1]
        mask = np.zeros((h, w))

        x_min, y_min, x_max, y_max = annot[0]

        mask[int(y_min):int(y_max), int(x_min):int(x_max)] = 1
        mask = mask.astype(np.uint8)

        return mask

    def save_and_plot_img_with_mask(self, mask, annotations, path):
        img = self.load_image(path=path)
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        x_min, y_min, x_max, y_max = annotations[0]

        ax[0].imshow(img)
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='r', facecolor='none')
        ax[0].add_patch(rect)
        ax[0].set_title("Image with Bounding Box")

        # Overlay Mask
        masked_img = cv2.addWeighted(img, 0.7, mask, 0.3, 0)
        ax[1].imshow(masked_img)
        ax[1].set_title("Image with Mask")

        dirs = os.path.join(os.getcwd(), "sam_lp", "datasets", "output")
        os.makedirs(dirs, exist_ok=True)
        fname = path.split("/")[-1]
        output_path = os.path.join(dirs, fname)

        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int):
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)
    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape BxHxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(
            image.shape[0], image.shape[1], 1024
        )
        # print(image_batch.shape)
        return np.array(resize(to_pil_image(image), target_size))
    def pad_image(self, x):
        h, w = x.shape[-2:]
        padh = 1024 - h
        padw = 1024 - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def normalize_image(self, x):
        x = (x - self.pixel_mean) / self.pixel_std
        return x

    def prepare_data(self):
        data = []
        for idx, path in tqdm(enumerate(self.image_list), desc="Data Loading in process"):
            if idx==10:
                break
            img = self.load_image(path)
            plate_annot = self.load_annotations(path)
            label = self.plate_mask(img, plate_annot)

            image = self.apply_image(img)
            label = self.apply_image(label)

            image_torch = (
                torch.as_tensor(image).permute(2, 0, 1).contiguous()
            )  # [None, :, :, :]
            label_torch = torch.as_tensor(label).contiguous()

            image_torch = self.pad_image(self.normalize_image(image_torch))
            label_torch = self.pad_image(label_torch)[None, :, :]

            if random.random() > 0.5:
                image_torch, label_torch = random_rot_flip_torch(image_torch, label_torch)
            elif random.random() > 0.5:
                image_torch, label_torch = random_rotate_torch(image_torch, label_torch)

            low_res_label = resize(label_torch, 1024 // 4)  # .squeeze()
            data.append({
                "image": image,
                "label": label,
                "low_res_label":low_res_label
            })
        logger.info("Data preparation completed.")
        return data
def random_rot_flip_torch(image, label):
    k = np.random.randint(0, 4)
    image = rotate(image, k * 90)
    label = rotate(label, k * 90)
    axis = np.random.randint(0, 2)
    if axis == 0:
        image = hflip(image)
        label = hflip(label)
    elif axis == 1:
        image = vflip(image)
        label = vflip(label)
    return image, label
def random_rotate_torch(image, label):
    angle = np.random.randint(-20, 20)
    image = rotate(image, angle)
    label = rotate(label, angle)
    return image, label
def create_hf_dataset(root, split="training"):
    dataset = UFPR_ALPR_Dataset(root, split)
    data = dataset.prepare_data()
    # hf_dataset = datasets.Dataset.from_list(data, split=split)
    # logger.info("HF dataset created.")
    return  data

dataset_train = create_hf_dataset(DATASET_PATH, "training")

def get_bounding_box(ground_truth_map):
    if len(ground_truth_map.shape) > 2:
        ground_truth_map = ground_truth_map.squeeze()

    y_indices, x_indices = np.where(ground_truth_map > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    H, W = ground_truth_map.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))
    bbox = [x_min, y_min, x_max, y_max]

    return bbox

def collater(data):
    images = [s["image"] for s in data]
    labels = [s["label"] for s in data]
    low_res_labels = [s["low_res_label"] for s in data]

    images = torch.stack(images, dim=0)
    labels = torch.stack(labels, dim=0).squeeze()
    low_res_labels = torch.stack(low_res_labels, dim=0).squeeze()

    return {"image": images, "label": labels, "low_res_label": low_res_labels}
def worker_init_fn(worker_id):
    random.seed(0 + worker_id)
class SAMDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        ground_truth_mask = np.array(item["low_res_label"])

        image = np.array(image)

        # Get bounding box prompt
        prompt = get_bounding_box(ground_truth_mask)

        # Prepare image and prompt for the model
        inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

        # Remove batch dimension which the processor adds by default
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # Add ground truth segmentation
        inputs["ground_truth_mask"] = ground_truth_mask

        return inputs




processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
train_dataset = SAMDataset(dataset=dataset_train, processor=processor)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True,
pin_memory=True,num_workers=2,
        worker_init_fn=worker_init_fn,)



model = SamModel.from_pretrained("facebook/sam-vit-base")

# make sure we only compute gradients for mask decoder
for name, param in model.named_parameters():
    if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
        param.requires_grad_(False)

optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

num_epochs = 2
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
logger.info(f"Training on {device} ...")

model.train()
for epoch in range(num_epochs):
    torch.cuda.empty_cache() 
    epoch_losses = []
    for batch in tqdm(train_dataloader):
        outputs = model(pixel_values=batch["pixel_values"].to(device),
                        input_boxes=batch["input_boxes"].to(device),
                        multimask_output=False)

        predicted_masks = outputs.pred_masks.squeeze(1)
        ground_truth_masks = batch["ground_truth_mask"].float().to(device)
        loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())

    logger.info(f'EPOCH: {epoch}')
    logger.info(f'Mean loss: {mean(epoch_losses)}')

logger.info("Training complete.")

# Free up CUDA resources
del model
torch.cuda.empty_cache()

logger.info("CUDA resources freed.")
model_save_path = "sam_model.pth"
torch.save(model.state_dict(), model_save_path)
logger.info(f"Model saved to {model_save_path}")

