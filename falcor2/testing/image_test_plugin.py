# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Image testing plugin for pytest.

This plugin provides functionality to compare rendered images against reference images,
store test results, and manage image test data.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
from botocore.exceptions import PartialCredentialsError
import warnings

from slangpy import Bitmap, Tensor
import falcor2.testing.helpers as helpers


class ImageTest:
    """
    A class that handles image testing for a specific test case.

    This class provides methods to compare images against reference images,
    store test results, and manage image test metadata.
    """

    def __init__(
        self,
        test_name: str,
        test_file: str,
        test_dir: Path,
        gen_images: bool,
        plugin: "ImageTestPlugin",
    ):
        super().__init__()
        self.test_name = test_name
        self.test_file = test_file
        self.test_dir = test_dir
        self.gen_images = gen_images
        self.plugin = plugin
        self.image_test_dir = helpers.PROJECT_ROOT / ".imagetests"
        self.image_test_dir.mkdir(exist_ok=True, parents=True)
        self.json_file = self.test_dir / f"{test_file}.images.json"
        self.references_dir = self.image_test_dir / "references"
        self.references_dir.mkdir(exist_ok=True, parents=True)

    def __call__(
        self,
        image: Union[np.ndarray[Any, Any], Bitmap, Tensor],
        tolerance: float = 1e-5,
        additional_name: Optional[str] = None,
        debug_output_normalize: bool = False,
        debug_output_stratify: bool = False,
    ) -> None:
        """
        Test an image against a reference.

        Args:
            image: The image to test (numpy array, slangpy Bitmap, or slangpy Tensor)
            tolerance: Tolerance for image comparison (default: 1e-5)
            name: Optional name for this image test (automatically populated for pytest parameterized tests)
            debug_output_normalize: Whether to normalize the debug output PNG for better visibility (default: False)
            debug_output_stratify: Whether to stratify the debug output (effectively show frac(value)) for better visibility of very large ranges (default: False)
        """
        # Convert image to numpy array for processing
        image_array = self._convert_to_numpy(image)

        # Generate hash of image content
        image_hash = self._compute_image_hash(image_array)

        # Always save debug PNG for every image test
        self._save_debug_png(
            image_array,
            additional_name,
            debug_output_normalize=debug_output_normalize,
            debug_output_stratify=debug_output_stratify,
        )

        if self.gen_images:
            self._save_reference_image(image_array, image_hash, additional_name)
        else:
            self._compare_with_reference(image_array, image_hash, additional_name, tolerance)

    def _convert_to_numpy(self, image: Union[np.ndarray[Any, Any], Bitmap, Tensor]):
        """Convert various image formats to numpy array."""
        if isinstance(image, np.ndarray):
            return image
        elif isinstance(image, Bitmap):
            bitmap = image.convert(
                pixel_format=Bitmap.PixelFormat.rgba, component_type=Bitmap.ComponentType.float32
            )
            return np.asarray(bitmap, copy=False)
        elif isinstance(image, Tensor):
            return image.to_numpy()
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

    def _compute_image_hash(self, image_array: np.ndarray[Any, Any]) -> str:
        """Compute SHA-256 hash of image data."""
        image_bytes = image_array.astype(np.float32).tobytes()
        return hashlib.sha256(image_bytes).hexdigest()

    def _save_reference_image(
        self, image_array: np.ndarray[Any, Any], image_hash: str, additional_name: Optional[str]
    ) -> None:
        """Save reference image and update metadata."""

        # Write out updated image
        bitmap = self._to_bitmap(image_array, for_png=False)
        image_path = self.references_dir / f"{image_hash}.exr"
        bitmap.write(str(image_path), Bitmap.FileFormat.exr)

        # Get / update meta data
        metadata = self.plugin.get_metadata(self.json_file)
        full_test_name = self.test_name
        if additional_name:
            full_test_name = f"{self.test_name}[{additional_name}]"
        if full_test_name not in metadata:
            metadata[full_test_name] = {}
        metadata[full_test_name]["reference"] = image_hash
        self.plugin.update_metadata(self.json_file, metadata)
        self.plugin.upload_reference(image_hash)
        print(f"Saved reference image for {full_test_name} -> {image_hash}")

    def _compare_with_reference(
        self,
        image_array: np.ndarray[Any, Any],
        image_hash: str,
        additional_name: Optional[str],
        tolerance: float,
    ) -> None:
        """Compare image with reference and fail test if different."""
        metadata = self.plugin.get_metadata(self.json_file)
        full_test_name = self.test_name
        if additional_name:
            full_test_name = f"{self.test_name}[{additional_name}]"

        # Check if test function and version exist
        if full_test_name not in metadata or "reference" not in metadata[full_test_name]:
            raise AssertionError(
                f"No reference image found for {full_test_name}. "
                f"Run with --gen-images to generate reference data."
            )

        # Compare hash first.
        reference_hash = metadata[full_test_name]["reference"]
        if image_hash == reference_hash:
            print(f"Image test passed for {full_test_name}")
            return

        # If we get here, the image doesn't match the reference
        # Load the reference image for comparison
        reference_path = self.references_dir / f"{reference_hash}.exr"
        if not reference_path.exists():
            self.plugin.download_reference(reference_hash)
        if not reference_path.exists():
            raise AssertionError(f"Reference image not found: {reference_path}")
        reference_bitmap = Bitmap.load_from_file(str(reference_path))
        reference_array = np.asarray(
            reference_bitmap.convert(component_type=Bitmap.ComponentType.float32),
            copy=False,
        )

        # Remove last dimension from reference array if comparing 2 channel image
        if (
            len(reference_array.shape) == 3
            and reference_array.shape[2] == 3
            and image_array.shape[2] == 2
        ):
            reference_array = reference_array[:, :, :2]

        # Compute difference + compare
        diff = float(np.mean((image_array - reference_array) ** 2))
        if diff > tolerance:
            self._save_error_images(image_array, reference_array, additional_name)
            raise AssertionError(
                f"Image test failed for {full_test_name}. "
                f"Difference: {diff:.6f}, tolerance: {tolerance:.6f}"
            )

    def _save_debug_png(
        self,
        image_array: np.ndarray[Any, Any],
        additional_name: Optional[str],
        debug_output_normalize: bool,
        debug_output_stratify: bool,
    ) -> None:
        """Always save a debug PNG of the generated image with a clean, readable name."""
        # Create debug directory
        debug_dir = self.image_test_dir / "debug" / self.test_file
        debug_dir.mkdir(exist_ok=True, parents=True)

        # Create clean test name for filename
        full_test_name = self.test_name
        if additional_name:
            full_test_name = f"{self.test_name}[{additional_name}]"

        # Clean the filename - replace problematic characters
        clean_name = (
            full_test_name.replace("[", "_")
            .replace("]", "")
            .replace(":", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )
        clean_name = clean_name.replace(" ", "_").replace(".", "_")

        # Add test file prefix for better organization
        debug_filename = f"{clean_name}.png"
        debug_path = debug_dir / debug_filename

        # Save debug PNG
        debug_bitmap = self._to_bitmap(
            image_array,
            for_png=True,
            normalize=debug_output_normalize,
            stratify=debug_output_stratify,
        )
        debug_bitmap.write(str(debug_path), Bitmap.FileFormat.png)
        print(f"Debug PNG saved: {debug_path}")

    def _to_uint8(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Convert float image data to 8-bit PNG format with gamma correction."""
        data = np.clip(data, 0.0, 1.0)
        return (data * 255).astype(np.uint8)

    def _to_bitmap(
        self,
        data: np.ndarray[Any, Any],
        for_png: bool,
        normalize: bool = False,
        stratify: bool = False,
    ) -> Bitmap:
        # Determine pixel format
        if len(data.shape) == 2:
            data = data[:, :, np.newaxis]
        assert len(data.shape) == 3
        assert data.shape[2] in (1, 2, 3, 4)
        if data.shape[2] == 1:
            pixel_format = Bitmap.PixelFormat.y
        elif data.shape[2] == 2:
            pixel_format = Bitmap.PixelFormat.rgb
            data = np.concatenate(
                [data, np.zeros((data.shape[0], data.shape[1], 1), dtype=data.dtype)], axis=2
            )
        elif data.shape[2] == 3:
            pixel_format = Bitmap.PixelFormat.rgb
        elif data.shape[2] == 4:
            pixel_format = Bitmap.PixelFormat.rgba
        else:
            raise ValueError(f"Unsupported image shape for PNG debug: {data.shape}")

        if normalize:
            min_val = np.min(data)
            max_val = np.max(data)
            if max_val > min_val:
                data = (data - min_val) / (max_val - min_val)
            else:
                data = np.zeros_like(data)

        if stratify:
            data = np.sin(data * np.pi * 32)
            data = (data + 1) * 0.5

        if for_png:
            data = np.abs(data)

        current_bitmap = Bitmap(data=data, pixel_format=pixel_format, srgb_gamma=False)
        if for_png:
            current_bitmap = current_bitmap.convert(
                component_type=Bitmap.ComponentType.uint8, srgb_gamma=True
            )
        return current_bitmap

    def _save_error_images(
        self,
        current: np.ndarray[Any, Any],
        reference: np.ndarray[Any, Any],
        additional_name: Optional[str],
    ) -> None:
        """Save current, reference, and difference images for debugging."""
        full_test_name = self.test_name
        if additional_name:
            full_test_name = f"{self.test_name}[{additional_name}]"
        error_dir = self.image_test_dir / "errors" / self.test_file / full_test_name
        error_dir.mkdir(exist_ok=True, parents=True)

        # Save current image
        current_bitmap = self._to_bitmap(current, for_png=False)
        current_path = error_dir / f"current.exr"
        current_bitmap.write(str(current_path), Bitmap.FileFormat.exr)

        # Save reference image
        reference_bitmap = self._to_bitmap(reference, for_png=False)
        reference_path = error_dir / f"reference.exr"
        reference_bitmap.write(str(reference_path), Bitmap.FileFormat.exr)

        # Compute and save difference image
        diff_image = np.abs(current - reference)
        diff_image = np.clip(diff_image * 10.0, 0.0, 1.0)

        # If 4 channel, output separate images for RGB and Alpha
        if len(diff_image.shape) > 2 and diff_image.shape[2] == 4:
            diff_bitmap = self._to_bitmap(diff_image[:, :, :3], for_png=False)
            diff_path = error_dir / f"diff_rgb.exr"
            diff_bitmap.write(str(diff_path), Bitmap.FileFormat.exr)

            alpha_diff = diff_image[:, :, 3:4]
            alpha_diff = np.concatenate([alpha_diff, alpha_diff, alpha_diff], axis=2)
            diff_bitmap = self._to_bitmap(alpha_diff, for_png=False)
            diff_path = error_dir / f"diff_alpha.exr"
            diff_bitmap.write(str(diff_path), Bitmap.FileFormat.exr)
        else:
            diff_bitmap = self._to_bitmap(diff_image, for_png=False)
            diff_path = error_dir / f"diff.exr"
            diff_bitmap.write(str(diff_path), Bitmap.FileFormat.exr)

        print(f"Error images saved to: {error_dir}")


class ImageTestPlugin:
    """Pytest plugin for image testing."""

    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.gen_images = config.getoption("--image-tests-generate")
        self.test_metadata: Dict[Path, Any] = {}
        self.modified_files: set[Path] = set()
        self.client: Any = None
        self._client_initialized = False

    def _get_client(self) -> Any:
        """Create the S3 client on first use."""
        if not self._client_initialized:
            self._client_initialized = True
            try:
                self.client = helpers.create_boto3_client()
            except PartialCredentialsError:
                warnings.warn("Failed to initialize access to AWS, image tests will fail.")
                self.client = None
        return self.client

    def _close_client(self) -> None:
        """Close the S3 client before Python shutdown finalizers run."""
        if self.client is None:
            return

        close = getattr(self.client, "close", None)
        if close is not None:
            try:
                close()
            except Exception as e:
                warnings.warn(f"Failed to close AWS image test client: {e}")
        self.client = None
        self._client_initialized = False

    def pytest_collection_modifyitems(self, config: Any, items: List[Any]) -> None:
        """Load all JSON metadata files during collection phase."""
        # Find all unique test files
        test_files = set()
        for item in items:
            test_file_path = Path(item.fspath)
            test_files.add(test_file_path)

        # Load metadata for each test file
        for test_file_path in test_files:
            test_dir = test_file_path.parent
            test_file = test_file_path.stem
            json_file = test_dir / f"{test_file}.images.json"
            if json_file.exists():
                try:
                    with open(json_file, "r") as f:
                        self.test_metadata[json_file] = json.load(f)
                except (json.JSONDecodeError, IOError):
                    self.test_metadata[json_file] = {}
            else:
                self.test_metadata[json_file] = {}

    def pytest_sessionfinish(self, session: Any, exitstatus: int) -> None:
        """Save all modified metadata files at the end of the session."""
        for json_file in self.modified_files:
            if json_file in self.test_metadata:
                try:
                    json_file.parent.mkdir(exist_ok=True, parents=True)
                    with open(json_file, "w") as f:
                        json.dump(self.test_metadata[json_file], f, indent=2)
                        f.write("\n")  # write trailing newline
                except IOError as e:
                    warnings.warn(f"Failed to save image test metadata to {json_file}: {e}")
        self._close_client()

    def download_reference(self, reference: str) -> None:
        """Download a specific reference image from S3."""
        bucket_name = "nvr-ci-imagetests"
        refs_dir = helpers.PROJECT_ROOT / ".imagetests" / "references"
        refs_dir.mkdir(exist_ok=True, parents=True)
        ref_path = refs_dir / f"{reference}.exr"
        if not ref_path.exists():
            s3_key = f"references/{reference}.exr"
            client = self._get_client()
            if client is None:
                return
            try:
                client.download_file(bucket_name, s3_key, str(ref_path))
            except Exception as e:
                warnings.warn(f"Failed to download {ref_path} from S3: {e}")

    def upload_reference(self, reference: str) -> None:
        """Upload a specific reference image to S3."""
        bucket_name = "nvr-ci-imagetests"
        refs_dir = helpers.PROJECT_ROOT / ".imagetests" / "references"
        ref_path = refs_dir / f"{reference}.exr"
        if ref_path.exists():
            s3_key = f"references/{reference}.exr"
            client = self._get_client()
            if client is None:
                return
            try:
                client.upload_file(str(ref_path), bucket_name, s3_key)
            except Exception as e:
                warnings.warn(f"Failed to upload {ref_path} to S3: {e}")

    def get_metadata(self, json_file: Path) -> Dict[str, Any]:
        """Get metadata for a specific JSON file."""
        return self.test_metadata.get(json_file, {})

    def update_metadata(self, json_file: Path, metadata: Dict[str, Any]) -> None:
        """Update metadata for a specific JSON file."""
        self.test_metadata[json_file] = metadata
        self.modified_files.add(json_file)

    def create_image_test(self, request: Any) -> ImageTest:
        """Create an ImageTest instance for a specific test."""
        test_name = request.node.name
        test_file = Path(request.fspath).stem  # Get filename without extension
        test_dir = Path(request.fspath).parent
        return ImageTest(test_name, test_file, test_dir, self.gen_images, self)
