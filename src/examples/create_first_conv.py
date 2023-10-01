from pathlib import Path
from torchmo import ConvObserver
import torchvision.models as models
import torchvision.transforms.functional as F
from PIL import Image

if __name__ == "__main__":
    root_p = Path(__file__).parents[2].resolve()

    img_p = root_p / "images/dog.jpg"

    img_size = 256

    img = Image.open(img_p).resize((img_size, img_size))

    img_t = F.to_tensor(img)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    observer = ConvObserver(model)

    observer(img_t.unsqueeze(dim=0))

    cache_p = root_p / ".cache"
    cache_p.mkdir(exist_ok=True)

    observer.save_figs(cache_p)
