import torch
import numpy as np
from src.model.unet import UNet

class Predictor:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.model = UNet()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)

    def preprocess(self, obstacle_map, start_map, goal_map, distance_map):
        x = np.stack([obstacle_map, start_map, goal_map, distance_map], axis=0).astype(np.float32)
        x = torch.from_numpy(x).unsqueeze(0).to(self.device)
        return x

    def predict(self, obstacle_map, start_map, goal_map, distance_map):
        x = self.preprocess(obstacle_map, start_map, goal_map, distance_map)
        with torch.no_grad():
            prob_map = self.model(x).cpu().numpy()[0, 0]
        return prob_map