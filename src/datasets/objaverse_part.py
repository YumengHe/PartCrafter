from src.utils.typing_utils import *

import json
import os
import random

import accelerate
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.utils.data_utils import load_surface, load_surfaces

class ObjaversePartDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        configs: DictConfig, 
        training: bool = True, 
    ):
        super().__init__()
        self.configs = configs
        self.training = training

        self.min_num_parts = configs['dataset']['min_num_parts']
        self.max_num_parts = configs['dataset']['max_num_parts']
        self.val_min_num_parts = configs['val']['min_num_parts']
        self.val_max_num_parts = configs['val']['max_num_parts']

        self.max_iou_mean = configs['dataset'].get('max_iou_mean', None)
        self.max_iou_max = configs['dataset'].get('max_iou_max', None)

        self.shuffle_parts = configs['dataset']['shuffle_parts']
        self.training_ratio = configs['dataset']['training_ratio']
        self.balance_object_and_parts = configs['dataset'].get('balance_object_and_parts', False)

        self.rotating_ratio = configs['dataset'].get('rotating_ratio', 0.0)
        self.rotating_degree = configs['dataset'].get('rotating_degree', 10.0)
        self.transform = transforms.Compose([
            transforms.RandomRotation(degrees=(-self.rotating_degree, self.rotating_degree), fill=(255, 255, 255)),
        ])

        if isinstance(configs['dataset']['config'], ListConfig):
            data_configs = []
            for config in configs['dataset']['config']:
                local_data_configs = json.load(open(config))
                if self.balance_object_and_parts:
                    if self.training:
                        local_data_configs = local_data_configs[:int(len(local_data_configs) * self.training_ratio)]
                    else:
                        local_data_configs = local_data_configs[int(len(local_data_configs) * self.training_ratio):]
                        local_data_configs = [config for config in local_data_configs if self.val_min_num_parts <= config['num_parts'] <= self.val_max_num_parts]
                data_configs += local_data_configs
        else:
            data_configs = json.load(open(configs['dataset']['config']))
        data_configs = [config for config in data_configs if config['valid']]
        data_configs = [config for config in data_configs if self.min_num_parts <= config['num_parts'] <= self.max_num_parts]
        if self.max_iou_mean is not None and self.max_iou_max is not None:
            data_configs = [config for config in data_configs if config['iou_mean'] <= self.max_iou_mean]
            data_configs = [config for config in data_configs if config['iou_max'] <= self.max_iou_max]
        if not self.balance_object_and_parts:
            if self.training:
                data_configs = data_configs[:int(len(data_configs) * self.training_ratio)]
            else:
                data_configs = data_configs[int(len(data_configs) * self.training_ratio):]
                data_configs = [config for config in data_configs if self.val_min_num_parts <= config['num_parts'] <= self.val_max_num_parts]
        self.data_configs = data_configs
        self.image_size = (512, 512)

    def __len__(self) -> int:
        return len(self.data_configs)
    
    def _get_data_by_config(self, data_config):
        if 'surface_path' in data_config:
            surface_path = data_config['surface_path']
            surface_data = np.load(surface_path, allow_pickle=True).item()
            # If parts is empty, the object is the only part
            part_surfaces = surface_data['parts'] if len(surface_data['parts']) > 0 else [surface_data['object']]
            num_parts = len(part_surfaces)

            # Create part indices for tracking shuffle order
            part_indices = list(range(num_parts))

            if self.shuffle_parts:
                # Shuffle both parts and indices together to maintain correspondence
                combined = list(zip(part_surfaces, part_indices))
                random.shuffle(combined)
                part_surfaces, part_indices = zip(*combined)
                part_surfaces = list(part_surfaces)
                part_indices = list(part_indices)

            part_surfaces = load_surfaces(part_surfaces) # [N, P, 6]
        else:
            part_surfaces = []
            for surface_path in data_config['surface_paths']:
                surface_data = np.load(surface_path, allow_pickle=True).item()
                part_surfaces.append(load_surface(surface_data))
            part_surfaces = torch.stack(part_surfaces, dim=0) # [N, P, 6]
            num_parts = part_surfaces.shape[0]
            part_indices = list(range(num_parts))

        # Load per-part images
        # Extract base directory from image_path (remove filename)
        base_dir = os.path.dirname(data_config['image_path'])

        images = []
        for i, part_idx in enumerate(part_indices):
            # Load image for each part: link_0.png, link_1.png, etc.
            part_image_path = os.path.join(base_dir, f"link_{part_idx}.png")

            if os.path.exists(part_image_path):
                # Load part-specific image
                image = Image.open(part_image_path).resize(self.image_size)
                # Only print debug info for first few samples
                if self.training and i == 0:  # Only log for first part of each object
                    print(f"DEBUG: Loading part {i} (original index {part_idx}) image from {part_image_path}")
            else:
                # Fallback to global image if part image doesn't exist
                if self.training:
                    print(f"WARNING: Part image {part_image_path} not found, using global image")
                image = Image.open(data_config['image_path']).resize(self.image_size)

            if random.random() < self.rotating_ratio:
                image = self.transform(image)
            image = np.array(image)
            image = torch.from_numpy(image).to(torch.uint8) # [H, W, 3]
            images.append(image)

        images = torch.stack(images, dim=0) # [N, H, W, 3]

        if self.training:
            print(f"DEBUG: Loaded {len(images)} images for {num_parts} parts, shuffle_parts={self.shuffle_parts}")
            print(f"DEBUG: Part indices order: {part_indices}")
            print(f"DEBUG: Images shape: {images.shape}, Part surfaces shape: {part_surfaces.shape}")

        return {
            "images": images,
            "part_surfaces": part_surfaces,
        }
    
    def __getitem__(self, idx: int):
        # The dataset can only support batchsize == 1 training. 
        # Because the number of parts is not fixed.
        # Please see BatchedObjaversePartDataset for batched training.
        data_config = self.data_configs[idx]
        data = self._get_data_by_config(data_config)
        return data
        
class BatchedObjaversePartDataset(ObjaversePartDataset):
    def __init__(
        self,
        configs: DictConfig,
        batch_size: int,
        is_main_process: bool = False,
        shuffle: bool = True,
        training: bool = True,
    ):
        assert training
        assert batch_size > 1
        super().__init__(configs, training)
        self.batch_size = batch_size
        self.is_main_process = is_main_process
        if batch_size < self.max_num_parts:
            self.data_configs = [config for config in self.data_configs if config['num_parts'] <= batch_size]
        
        if shuffle:
            random.shuffle(self.data_configs)

        self.object_configs = [config for config in self.data_configs if config['num_parts'] == 1]
        self.parts_configs = [config for config in self.data_configs if config['num_parts'] > 1]
        
        self.object_ratio = configs['dataset']['object_ratio']
        # Here we keep the ratio of object to parts
        self.object_configs = self.object_configs[:int(len(self.parts_configs) * self.object_ratio)]

        dropped_data_configs = self.parts_configs + self.object_configs
        if shuffle:
            random.shuffle(dropped_data_configs)

        self.data_configs = self._get_batched_configs(dropped_data_configs, batch_size)
    
    def _get_batched_configs(self, data_configs, batch_size):
        batched_data_configs = []
        num_data_configs = len(data_configs)
        progress_bar = tqdm(
            range(len(data_configs)),
            desc="Batching Dataset",
            ncols=125,
            disable=not self.is_main_process,
        )
        while len(data_configs) > 0:
            temp_batch = []
            temp_num_parts = 0
            unchosen_configs = []
            while temp_num_parts < batch_size and len(data_configs) > 0:
                config = data_configs.pop() # pop the last config
                num_parts = config['num_parts']
                if temp_num_parts + num_parts <= batch_size:
                    temp_batch.append(config)
                    temp_num_parts += num_parts
                    progress_bar.update(1)
                else:
                    unchosen_configs.append(config) # add back to the end
            data_configs = data_configs + unchosen_configs # concat the unchosen configs
            if temp_num_parts == batch_size:
                # Successfully get a batch
                if len(temp_batch) < batch_size:
                    # pad the batch
                    temp_batch += [{}] * (batch_size - len(temp_batch))
                batched_data_configs += temp_batch
                # Else, the code enters here because len(data_configs) == 0
                # which means in the left data_configs, there are no enough 
                # "suitable" configs to form a batch. 
                # Thus, drop the uncompleted batch.
        progress_bar.close()
        return batched_data_configs
        
    def __getitem__(self, idx: int):
        data_config = self.data_configs[idx]
        if len(data_config) == 0:
            # placeholder
            return {}
        data = self._get_data_by_config(data_config)
        return data
    
    def collate_fn(self, batch):
        batch = [data for data in batch if len(data) > 0]

        # Collect all images and part surfaces
        all_images = []
        all_surfaces = []
        num_parts_list = []

        # Initialize a class variable to track batch count for debug output
        if not hasattr(self, '_debug_batch_count'):
            self._debug_batch_count = 0
        self._debug_batch_count += 1

        # Only print debug info for first few batches
        debug_this_batch = self._debug_batch_count <= 3

        if debug_this_batch:
            print(f"DEBUG: Collating batch #{self._debug_batch_count} with {len(batch)} objects")

        for obj_idx, data in enumerate(batch):
            obj_images = data['images']
            obj_surfaces = data['part_surfaces']
            obj_num_parts = obj_surfaces.shape[0]

            if debug_this_batch:
                print(f"DEBUG: Object {obj_idx}: {obj_num_parts} parts, images shape: {obj_images.shape}, surfaces shape: {obj_surfaces.shape}")

            all_images.append(obj_images)
            all_surfaces.append(obj_surfaces)
            num_parts_list.append(obj_num_parts)

        images = torch.cat(all_images, dim=0) # [N, H, W, 3]
        surfaces = torch.cat(all_surfaces, dim=0) # [N, P, 6]
        num_parts = torch.LongTensor(num_parts_list)

        total_parts = num_parts.sum().item()

        if debug_this_batch:
            print(f"DEBUG: Final batch - Images: {images.shape}, Surfaces: {surfaces.shape}, Num parts: {num_parts.tolist()}")
            print(f"DEBUG: Total parts: {total_parts}, Expected batch size: {self.batch_size}")

        assert images.shape[0] == surfaces.shape[0] == total_parts == self.batch_size, \
            f"Shape mismatch: images={images.shape[0]}, surfaces={surfaces.shape[0]}, total_parts={total_parts}, batch_size={self.batch_size}"

        batch = {
            "images": images,
            "part_surfaces": surfaces,
            "num_parts": num_parts,
        }
        return batch