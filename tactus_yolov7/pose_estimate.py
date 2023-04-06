from pathlib import Path
import os
import sys
import numpy as np
import torch
from torchvision import transforms
from torch.hub import download_url_to_file
from tactus_yolov7.utils.datasets import letterbox
from .models.experimental import attempt_load
from tactus_yolov7.utils.plots import output_to_keypoint
from tactus_yolov7.utils.general import non_max_suppression_kpt


class Yolov7:
    def __init__(self, model_weights: Path, device: str = 'cuda:0') -> None:
        self._select_device(device)
        self._load_model(model_weights)

    def _select_device(self, device):
        if device is None:
            device = "cuda:0"
        cpu = device.lower() == 'cpu'
        cuda = not cpu and torch.cuda.is_available()
        self._device = torch.device(device if cuda else 'cpu')

    def _load_model(self, model_weights):
        # allow the detection of the module 'models' from yolov7
        # when loading the model with attempt_load()
        sys.path.append(os.path.join(os.path.dirname(__file__), ""))
        check_weights(model_weights)
        self._model = attempt_load(model_weights, map_location=self._device)
        self._model.eval()

    def predict_frame(self, img: np.ndarray) -> list[dict]:
        """
        return the list of every skeleton keypoints in the image

        Parameters
        ----------
        img : np.ndarray
            an image with a width and height dividable by 64.
            resize() can be used to get the new resized image.

        Returns
        -------
        list
            list of dictionnaries
        """
        image = transforms.ToTensor()(img)
        image = torch.tensor(np.array([image.numpy()]))
        image = image.to(self._device)
        image = image.float()

        with torch.no_grad():
            output, _ = self._model(image)

        output = non_max_suppression_kpt(output, 0.25, 0.65,
                                         nc=self._model.yaml['nc'],
                                         kpt_label=True)
        output = output_to_keypoint(output)

        skeletons = []
        for idx in range(output.shape[0]):
            skeletons.append({
                "keypoints": output[idx][7:58].tolist(),
                "score": output[idx][6].tolist(),
                "box": output[idx][2:6].tolist(),
            })

        return skeletons

def resize(img: np.ndarray) -> np.ndarray:
    """
    return t

    Parameters
    ----------
    img : np.ndarray
        the img to be resized

    Returns
    -------
    np.ndarray
        the new transformed image
    """
    new_width = (img.shape[0] // 64 + 1) * 64
    new_height = (img.shape[1] // 64 + 1) * 64

    image = letterbox(img, (new_width, new_height), stride=64, auto=True)[0]

    return image

def download_weights(weights_path: Path):
    """Download yolov7 pose weights"""

    url = "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    download_url_to_file(url, weights_path.absolute())

def check_weights(weights_path: Path):
    """check that the weights file exists"""
    if not weights_path.exists():
        download_weights(weights_path)
