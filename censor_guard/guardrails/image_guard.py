from image_preprocess import ImagePreprocessor, ImageDenoiser
from image_steganalysis import ImageSteganalysis
# from censor_guard.schemas import SignalResult

from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Denoises the image, sanitizes it and returns its residual metrics (high texture)
class ImageAnalyzer:

    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.steganalysis = ImageSteganalysis()
        self.denoiser = ImageDenoiser()

        # self.max_residual_energy = 5000

    def sanitize(self, image):
        return self.preprocessor.preprocess(image)

    # def is_usable(self, image):

    #     residual = self.steganalysis.process(
    #         image
    #     )

    #     return (residual < self.max_residual_energy, residual)

    def process(self, image):

        if isinstance(image, Image.Image):
            image = self.pil_to_np(image)

        denoised = self.denoiser.denoise_if_needed(image)
        print(denoised[2], denoised[1])
        image = denoised[0]

        sanitized = self.sanitize(image)

        risk = self.steganalysis.process(sanitized)

        return {
            "noise_score": denoised[2],
            "denoised": denoised[1],
            "jpeg_recompressed": True,
            "residual": risk,
            "image": self.np_to_pil(sanitized)
        }

    def compare_embeds(self, embed_before, embed_after):
        pass


    def pil_to_np(self, img: Image.Image) -> np.ndarray:
        return np.array(img.convert("RGB"))

    def np_to_pil(self, arr: np.ndarray) -> Image.Image:
        return Image.fromarray(arr.astype(np.uint8))


if __name__ == '__main__':
    guard = ImageGuard()
    img = Image.open(Path("demo/boobs_noise.jpg"))
    sanitized = guard.process(img)
    print(sanitized)
    plt.imshow(sanitized["image"])
    plt.show()


# class ImageQuality:

#     def blur_score(self, image):
#         gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#         return cv2.Laplacian(gray, cv2.CV_32F).var()

#     def contrast_score(self, image):
#         gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#         return gray.std()

#     def noise_score(self, image):
#         gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

#         blur = cv2.GaussianBlur(gray, (5, 5), 0)

#         residual = gray.astype(np.float32) - blur.astype(np.float32)

#         return residual.std()

#     def process(self, image):
#         blur_score = self.blur_score(image)
#         contrast_score = self.contrast_score(image)
#         noise_score = self.noise_score(image)

#         quality_score = {
#             "blur": blur_score,
#             "contrast": contrast_score,
#             "noise": noise_score
#         }