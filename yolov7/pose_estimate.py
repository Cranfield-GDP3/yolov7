from pathlib import Path
import numpy as np
import cv2
import torch
from torchvision import transforms
from .utils.datasets import letterbox
from .models.experimental import attempt_load
from .utils.plots import output_to_keypoint
from .utils.general import non_max_suppression_kpt


def pred_frame(img: np.ndarray, device, model) -> dict:
    image = letterbox(img, 960, stride=64, auto=True)[0]
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))
    image = image.to(device)
    image = image.float()

    with torch.no_grad():
        output, _ = model(image)

    output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
    output = output_to_keypoint(output)

    # Convert every keypoint detection in frame to List
    skeletons = []
    for idx in range(output.shape[0]):
        skeletons.append({
            "keypoints": output[idx][7:58].tolist(),
            "score": output[idx][6].tolist(),
            "box": output[idx][2:6].tolist(),
            "idx": ["To be calculated using Deepsort"]
        })

    return skeletons

class Yolov7:
    def __init__(self, model_weights: Path, device: str = 'cuda:0') -> None:
        cpu = device.lower() == 'cpu'
        cuda = not cpu and torch.cuda.is_available()
        self._device = torch.device(device if cuda else 'cpu')

        self._model = attempt_load(model_weights, map_location=self._device)
        self._model.eval()

    def predict_dir(self, input_dir: Path):
        formatted_json = {}
        formatted_json["frames"] = []
        for frame_path in Path(input_dir).glob("*.jpg"):
            frame_json = {"frame_id": frame_path.stem}

            img = cv2.imread(str(frame_path))
            skeletons = self.predict_frame(img)

            frame_json["skeletons"] = skeletons
            formatted_json["frames"].append(frame_json)

        return formatted_json

    def predict_frame(self, img: np.ndarray):
        image = letterbox(img, 960, stride=64, auto=True)[0]
        image = transforms.ToTensor()(image)
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
