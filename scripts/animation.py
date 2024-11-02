"""A simple script to create more animations."""
import os
import torch
from src.models import LSTM
from src.task import MorrisWaterMaze
from src.test import test

src_directory = '../../morris-water-maze'

model_parameters = torch.load(os.path.join(src_directory, 'data/model_6942069.pt'))
model = model = LSTM(49, 500, 3)
model.load_state_dict(model_parameters)
model.eval()

environment_path = os.path.join(src_directory, 'data/environment_6942069.pt')
enviro = MorrisWaterMaze(420)
enviro.ReAssignReward((10, 10))
enviro.Load(environment_path)

animation_loc = os.path.join(src_directory, 'results')
test(enviro, model, 10, 100, 3, 'test', animation_loc, fps=5)