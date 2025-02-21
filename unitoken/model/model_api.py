import os
import sys
import glob
import json

sys.path.append(os.path.abspath(__file__).rsplit("/", 2)[0])

from PIL import Image

from inference_solver_base import FlexARInferenceSolverBase
from inference_solver_anyres import FlexARInferenceSolverAnyRes

def create_model(pretrained_path, precision, target_size):
    ## For basic "UniToken" 
    inference_solver = FlexARInferenceSolverBase(
        model_path=pretrained_path,
        precision=precision,
        target_size=target_size,
    )
    ## For AnyRes "UniToken" 
    # inference_solver = FlexARInferenceSolverAnyRes(
    #     model_path=pretrained_path,
    #     precision=precision,
    #     target_size=target_size,
    # )

    return inference_solver