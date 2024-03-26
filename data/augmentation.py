from torch.nn import Module
from numpy import array, zeros_like, float32
from PIL import Image
from cv2 import GaussianBlur, Canny, addWeighted


class CannyEdgeDetection(Module):
    def __init__(self, threshold1: float, threshold2: float):
        super().__init__()
        self.threshold1 = threshold1
        self.threshold2 = threshold2

    def preprocess_mask(self, mask):
        mask = array(mask, dtype=float32)
        mask[mask == 2.0] = 0.0
        mask[(mask == 1.0) | (mask == 3.0)] = 1.0
        return Image.fromarray(mask)
    
    def forward(self, inpt: Image.Image, target: Image.Image) -> tuple[Image.Image, Image.Image]:
        image = array(inpt)
        target = self.preprocess_mask(target)

        # Apply Gaussian blur and then use Canny edge detection
        image_blur = GaussianBlur(image, (9, 9), 0)
        edges = Canny(image_blur, self.threshold1, self.threshold2)

        # Highlight edges
        edges_colored_dilated = zeros_like(image)
        edges_colored_dilated[edges > 0] = 255

        # Edges -> Original img
        image_with_edges = addWeighted(image, 1., edges_colored_dilated, 1., 0.)
        return Image.fromarray(image_with_edges), target
