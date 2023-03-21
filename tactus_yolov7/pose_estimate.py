from pathlib import Path
import os
import sys
import numpy as np
import torch
from torchvision import transforms
from .utils.datasets import letterbox
from .models.experimental import attempt_load
from .utils.plots import output_to_keypoint
from .utils.general import non_max_suppression_kpt


class Yolov7:
    def __init__(self, model_weights: Path, device: str = 'cuda:0') -> None:
        cpu = device.lower() == 'cpu'
        cuda = not cpu and torch.cuda.is_available()
        self._device = torch.device(device if cuda else 'cpu')

        sys.path.append(os.path.join(os.path.dirname(__file__), ""))
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
                "idx": ["To be calculated using Deepsort"]
            })

        return skeletons

def resize(img: np.ndarray) -> np.ndarray:
    """
    return t

    Parameters
    ----------
    in_shape : tuple[int, int]
        _description_

    Returns
    -------
    tuple[int, int]
        _description_
    """
    new_width = (img.shape[0] // 64 + 1) * 64
    new_height = (img.shape[1] // 64 + 1) * 64

    image = letterbox(img, (new_width, new_height), stride=64, auto=True)[0]

    return image
