from torch.nn import Module
from numpy import array, zeros_like, float32
from PIL import Image
from cv2 import GaussianBlur, Canny, addWeighted


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
    @staticmethod
    def forward(inpt: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        mask = array(mask)
        mask[mask == 2] = 0
        mask[mask == 3] = 255
        return inpt, Image.fromarray(mask)

class Remap(Module):
    def __init__(self, min_val: float, max_val: float):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
    
    def forward(self, input: Image.Image, target: Image.Image) -> tuple[Image.Image, Image.Image]:
        org_min, org_max = input.getextrema()
        
        # Remap the values
        target = (input - org_min) / (org_max - org_min) * (self.max_val - self.min_val) + self.min_val
        return target, target