import cv2 
import numpy as np

class ImageSteganalysis:

    def residual_energy(self, image):
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )

        residual = cv2.Laplacian(
            gray,
            cv2.CV_32F
        )

        return float(
            np.mean(residual ** 2)
        )

    def process(self, image):
        return self.residual_energy(image)