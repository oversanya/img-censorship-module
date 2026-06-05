
import io
import cv2
import numpy as np
from PIL import Image


class ImagePreprocessor:
    """
    Implementation of the paper's adaptive preprocessing layer.

    Features:
      - Content-aware filtering
      - Selective recompression
      - Randomized processing
    """

    def __init__(
        self,
        patch_size=32,
        sigma_range=(0.5, 1.0),
        jpeg_quality=88,
        texture_min=10.0,
        texture_max=500.0,
        randomization_strength=0.10,
        noise_std=0.2
    ):
        self.patch_size = patch_size

        self.min_sigma = sigma_range[0]
        self.max_sigma = sigma_range[1]

        self.jpeg_quality = jpeg_quality

        self.texture_min = texture_min
        self.texture_max = texture_max

        self.randomization_strength = randomization_strength

        self.noise_std = noise_std

    # --------------------------------------------------
    # Texture analysis
    # --------------------------------------------------

    def texture_score(self, patch):
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)

        lap = cv2.Laplacian(
            gray,
            cv2.CV_32F
        )

        return float(lap.var())

    def sigma_from_texture(self, texture):

        texture = np.clip(
            texture,
            self.texture_min,
            self.texture_max
        )

        normalized = (
            (texture - self.texture_min)
            / (self.texture_max - self.texture_min)
        )

        sigma = (
            self.max_sigma
            - normalized
            * (self.max_sigma - self.min_sigma)
        )

        return sigma

    # --------------------------------------------------
    # Content-aware filtering
    # --------------------------------------------------

    def adaptive_gaussian_blur(self, image):

        h, w = image.shape[:2]

        result = image.copy()

        for y in range(0, h, self.patch_size):
            for x in range(0, w, self.patch_size):

                patch = image[
                    y:y+self.patch_size,
                    x:x+self.patch_size
                ]

                if patch.size == 0:
                    continue

                texture = self.texture_score(patch)

                sigma = self.sigma_from_texture(texture)

                # Randomized processing
                sigma *= np.random.uniform(
                    1.0 - self.randomization_strength,
                    1.0 + self.randomization_strength
                )

                sigma = max(
                    self.min_sigma,
                    min(self.max_sigma, sigma)
                )

                filtered = cv2.GaussianBlur(
                    patch,
                    (3, 3),
                    sigmaX=sigma
                )

                result[
                    y:y+self.patch_size,
                    x:x+self.patch_size
                ] = filtered

        return result

    def add_noise(self, image):

        noise = np.random.normal(
            0,
            self.noise_std,
            image.shape
        )

        out = image.astype(np.float32) + noise

        return np.clip(
            out,
            0,
            255
        ).astype(np.uint8)

    def jpeg_recompress(self, image):

        buffer = io.BytesIO()

        Image.fromarray(image).save(
            buffer,
            format="JPEG",
            quality=self.jpeg_quality
        )

        buffer.seek(0)

        return np.array(
            Image.open(buffer).convert("RGB")
        )

    def preprocess(self, image):

        image = self.adaptive_gaussian_blur(image)
        image = self.add_noise(image)
        image = self.jpeg_recompress(image)

        return image


class ImageDenoiser:
    def __init__(
        self,
        noise_threshold=25.0,
        denoise_strength=25
    ):
        self.noise_threshold = noise_threshold
        self.denoise_strength = denoise_strength

    def estimate_noise(self, image):
        """
        Estimate noise level using a high-pass residual.

        Returns:
            float: estimated noise standard deviation.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        smooth = cv2.GaussianBlur(gray, (5, 5), 0)

        residual = gray.astype(np.float32) - smooth.astype(np.float32)

        return float(residual.std())

    def denoise_if_needed(self, image):
        """
        Apply denoising only when estimated noise
        exceeds the configured threshold.
        """
        noise_level = self.estimate_noise(image)

        if noise_level < self.noise_threshold:
            return image, noise_level, False

        denoised = cv2.fastNlMeansDenoisingColored(
            image,
            None,
            h=self.denoise_strength,
            hColor=self.denoise_strength,
            templateWindowSize=7,
            searchWindowSize=21,
        )

        return denoised, noise_level, True
