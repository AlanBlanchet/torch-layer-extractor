from pathlib import Path
from torchmo import ConvObserver
import torchvision.models as models
import torchvision.transforms.functional as F
from PIL import Image


img_p = Path(__file__).parents[1].resolve() / "images/dog.jpg"


def test_conv():
    img = Image.open(img_p).resize((256, 256))

    img_t = F.to_tensor(img)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    observer = ConvObserver(model)

    assert observer is not None
    assert len(list(observer(img_t.unsqueeze(dim=0)))) != 0
