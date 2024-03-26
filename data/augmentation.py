from torch.nn import Module
from numpy import array, zeros_like
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
