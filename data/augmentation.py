from torch.nn import Module
from PIL import Image
from numpy import array, zeros_like
from cv2 import GaussianBlur, Canny, addWeighted
from torch import Tensor, float32
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import Resize, ToDtype, Normalize
from torchvision.transforms.v2.functional import to_image, to_dtype


class CannyEdgeDetection(Module):
    def __init__(self, threshold1: float, threshold2: float):
        super().__init__()
        self.threshold1 = threshold1
        self.threshold2 = threshold2
    
    def forward(self, inpt: Image.Image, target: Image.Image) -> tuple[Image.Image, Image.Image]:
        image = array(inpt)

        # Apply Gaussian blur and then use Canny edge detection
        image_blur = GaussianBlur(image, (9, 9), 0)
        edges = Canny(image_blur, self.threshold1, self.threshold2)

        # Highlight edges
        edges_colored_dilated = zeros_like(image)
        edges_colored_dilated[edges > 0] = 255

        # Edges -> Original img
        image_with_edges = addWeighted(image, 1., edges_colored_dilated, 1., 0.)
        return Image.fromarray(image_with_edges), target


class MaskPreprocessing(Module):
    def __init__(self, explicit_edge: bool = False):
        super().__init__()
        self.explicit_edge = explicit_edge
    
    def forward(self, inpt: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        mask = array(mask)
        # Mask == 1: Foreground
        mask[mask == 1] = 255
        # Mask == 2: Background
        mask[mask == 2] = 0
        # Mask == 3: Edge
        mask[mask == 3] = 127 if self.explicit_edge else 255
        return inpt, Image.fromarray(mask)


class Preprocess(Module):
    def __init__(self):
        super().__init__()

    def forward(self, inpt: Image.Image, mask: Image.Image) -> tuple[Tensor, Tensor]:
        ## Image processing
        # resize with bicubic interpolation
        inpt = Resize((224, 224), InterpolationMode.BICUBIC)(inpt)
        # convert to tensor
        inpt = to_image(inpt)
        inpt = to_dtype(inpt, float32)
        # normalize with mean and std if ImageNet
        # Normalize((.485, .456, .406), (.229, .224, .225), True)(inpt)

        ## Mask processing
        # 处理三色图
        mask = array(mask)
        mask[mask == 2] = 0
        mask[mask == 3] = 1
        mask = Image.fromarray(mask)
        # resize with nearest interpolation
        mask = Resize((224, 224), InterpolationMode.NEAREST)(mask)
        # convert to tensor
        mask = to_image(mask)
        mask = to_dtype(mask, float32)

        return inpt, mask
