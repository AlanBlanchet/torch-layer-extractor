from pathlib import Path
from torchmo import ConvObserver
import torchvision.models as models
import torchvision.transforms.functional as F
from PIL import Image


img_p = Path(__file__).parents[1].resolve() / "images/dog.jpg"


img = Image.open(img_p).resize((256, 256))

img_t = F.to_tensor(img)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)


def test_conv():
    observer = ConvObserver(model)

    assert observer is not None
    assert len(list(observer(img_t.unsqueeze(dim=0)))) != 0


def test_limit():
    observer = ConvObserver(model, limit=2)

    assert len(list(observer(img_t.unsqueeze(dim=0)))) == 2


def test_valid_module():
    observer = ConvObserver(model, watch="conv1")

    assert len(list(observer(img_t.unsqueeze(dim=0)))) == 1


def test_valid_modules():
    observer = ConvObserver(model, watch=["conv1", "layer2.0.conv2"])

    assert len(list(observer(img_t.unsqueeze(dim=0)))) == 2
