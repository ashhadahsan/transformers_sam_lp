import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from datasets import Dataset,DatasetDict,Features,Array3D,Array2D,NamedSplit

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATASET_PATH = "/tmp/ahsan/sqfs/storage_local/datasets/public/ufpr-alpr"

class UFPR_ALPR_Dataset:
    def __init__(self, root, split="training"):
        self.data_dir = os.path.join(root, split)
        self.image_list = self.build_image_list()
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

    def prepare_data(self):
        images = []
        masks = []
        for idx, path in tqdm(enumerate(self.image_list), desc="Data Loading in process"):
            img = self.load_image(path)
            plate_annot = self.load_annotations(path)
            mask = self.plate_mask(img, plate_annot)
            images.append(img)
            masks.append(mask)
        logger.info("Data preparation completed.")
        return {"image": images, "mask": masks}


def create_hf_dataset(root, split="training"):
    dataset = UFPR_ALPR_Dataset(root, split)
    data = dataset.prepare_data()
    features = Features({
        'image': Array3D(dtype="uint8", shape=(1080, 1920, 3)),  # Fixed shape
        'mask': Array2D(dtype="uint8", shape=(1080, 1920))       # Fixed shape
    })
    hf_dataset = Dataset.from_dict(data, split=NamedSplit(split)
,features=features)

    logger.info("HF dataset created.")
    return hf_dataset

train_hf_dataset = create_hf_dataset(DATASET_PATH,"training")
# validation_hf_dataset = create_hf_dataset(DATASET_PATH,"testing")

# Combine into a DatasetDict
hf_dataset_dict = DatasetDict({
    'train': train_hf_dataset,
    # 'test': validation_hf_dataset
})
hf_dataset_dict.push_to_hub("ashhadahsan/ufpr_dataset", private=True,max_shard_size="20GB")
# import os
# save_path=os.path.join(os.getcwd(),"ufpr_hf_dataset/")
# hf_dataset_dict.save_to_disk(save_path)
