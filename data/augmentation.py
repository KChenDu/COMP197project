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
