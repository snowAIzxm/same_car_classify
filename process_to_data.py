from typing import Dict, List

import clip
import numpy as np
import torch
from PIL import Image

from bean import CarData


class EmbedingModel:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device=device)

    @staticmethod
    def normalized(a, axis=-1, order=2):
        l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
        l2[l2 == 0] = 1
        return a / np.expand_dims(l2, axis)

    def predict_batch_image_list(self, image_path_list):

        batch_image_list = []
        batch_file_list = []
        for file in image_path_list:
            try:
                image = Image.open(file)
                image = self.clip_preprocess(image)
                batch_image_list.append(image)
                batch_file_list.append(file)
            except Exception as e:
                print(f"error load file:{file}")
        image_tensor = torch.stack(batch_image_list)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_tensor.cuda())
            norm_features = self.normalized(image_features.cpu().numpy())
        return norm_features

    def process_car_image_path_list_dict(self, car_id_image_path_list_dict: Dict[any, List[str]]):
        car_data_list = []
        for car_id, image_path_list in car_id_image_path_list_dict.items():
            feature = self.predict_batch_image_list(image_path_list)
            car_data = CarData(id=car_id, feature_list=feature)
            car_data_list.append(car_data)
        return car_data_list
