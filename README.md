# GPU-Accelerated Sobel Edge Detection Using CUDA in Google Colab

## Project Overview

This project demonstrates GPU-accelerated image processing using CUDA programming in Python. The application performs Sobel edge detection on grayscale images and compares execution time between CPU and GPU implementations.

The project was developed and tested using Google Colab with NVIDIA GPU hardware.

---

## Technologies Used

- Python
- CUDA
- Numba CUDA
- NumPy
- Matplotlib
- PIL (Python Imaging Library)
- SciPy
- Google Colab

---

## GPU Hardware

The project runs on NVIDIA Tesla T4 GPU provided by Google Colab.

To verify GPU availability:

```python
!nvidia-smi
```

---

# Project Workflow

1. Load image
2. Convert image to grayscale
3. Resize image
4. Apply Sobel edge detection using CPU
5. Apply Sobel edge detection using CUDA GPU kernel
6. Compare execution times
7. Visualize results

---

# Installation

Open Google Colab:

[[https://colab.research.google.com](https://colab.research.google.com/drive/16B5xWX-PwpbNiIUHEhpsV53nH72GWtsJ?usp=sharing)
]

Runtime → Change runtime type → GPU

Install required libraries:

```python
!pip install numba pillow matplotlib scipy requests
```

---

# Full Project Code

## Import Libraries

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import requests
import time
from numba import cuda
from scipy.ndimage import sobel
```

---

## Download and Prepare Image

```python
url = "https://images.unsplash.com/photo-1506744038136-46273834b3fb"

response = requests.get(url)

img = Image.open(BytesIO(response.content)).convert("L")

# Resize image
img = img.resize((512, 512))

img_np = np.array(img).astype(np.float32)

plt.imshow(img_np, cmap="gray")
plt.title("Original Grayscale Image")
plt.axis("off")
plt.show()
```

---

## CPU Sobel Edge Detection

```python
start_cpu = time.time()

gx = sobel(img_np, axis=1)
gy = sobel(img_np, axis=0)

cpu_result = np.sqrt(gx * gx + gy * gy)
cpu_result = np.clip(cpu_result, 0, 255)

end_cpu = time.time()

cpu_time = end_cpu - start_cpu

print("CPU Time:", cpu_time, "seconds")
```

---

## GPU CUDA Sobel Kernel

```python
@cuda.jit
def sobel_gpu_kernel(image, output):

    y, x = cuda.grid(2)

    height, width = image.shape

    if 1 <= y < height - 1 and 1 <= x < width - 1:

        gx = (
            -image[y-1, x-1] + image[y-1, x+1]
            -2 * image[y, x-1] + 2 * image[y, x+1]
            -image[y+1, x-1] + image[y+1, x+1]
        )

        gy = (
            -image[y-1, x-1]
            -2 * image[y-1, x]
            -image[y-1, x+1]
            + image[y+1, x-1]
            + 2 * image[y+1, x]
            + image[y+1, x+1]
        )

        value = (gx * gx + gy * gy) ** 0.5

        if value > 255:
            value = 255

        output[y, x] = value
```

---

## Run GPU Version

```python
d_image = cuda.to_device(img_np)

d_output = cuda.device_array_like(img_np)

threads_per_block = (16, 16)

blocks_per_grid_y = int(
    np.ceil(img_np.shape[0] / threads_per_block[0])
)

blocks_per_grid_x = int(
    np.ceil(img_np.shape[1] / threads_per_block[1])
)

blocks_per_grid = (
    blocks_per_grid_y,
    blocks_per_grid_x
)

# Warm-up run
sobel_gpu_kernel[
    blocks_per_grid,
    threads_per_block
](d_image, d_output)

cuda.synchronize()

start_gpu = time.time()

sobel_gpu_kernel[
    blocks_per_grid,
    threads_per_block
](d_image, d_output)

cuda.synchronize()

end_gpu = time.time()

gpu_result = d_output.copy_to_host()

gpu_time = end_gpu - start_gpu

print("GPU Time:", gpu_time, "seconds")
print("Speedup:", cpu_time / gpu_time)
```

---

## Display Results

```python
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img_np, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(cpu_result, cmap="gray")
plt.title("CPU Sobel Edge Detection")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(gpu_result, cmap="gray")
plt.title("GPU CUDA Sobel Edge Detection")
plt.axis("off")

plt.show()
```

---

## Performance Comparison Chart

```python
labels = ["CPU", "GPU"]

times = [cpu_time, gpu_time]

plt.bar(labels, times)

plt.ylabel("Execution Time in Seconds")

plt.title("CPU vs GPU Performance")

plt.show()
```

---

## Final Performance Summary

```python
print("Final Performance Summary")
print("-------------------------")
print("CPU Time:", cpu_time, "seconds")
print("GPU Time:", gpu_time, "seconds")
print("Speedup:", cpu_time / gpu_time)
```

---

# Example Output

```text
CPU Time: 0.12 seconds
GPU Time: 0.004 seconds
Speedup: 30x
```

Performance may vary depending on GPU hardware and image size.

---

# CUDA Concepts Used

- CUDA kernels
- Thread blocks
- Grid dimensions
- Device memory allocation
- Host-device memory transfer
- Parallel execution

---

# Learning Outcomes

This project demonstrates:

- GPU acceleration using CUDA
- Parallel image processing
- CPU vs GPU benchmarking
- CUDA memory management
- Kernel execution design

---

# Future Improvements

Possible future enhancements:

- Real-time video edge detection
- Webcam support
- Additional image filters
- Shared memory optimization
- Multi-GPU support
- Real-time GPU image processing dashboard

---
OUTPUT:
<img width="1182" height="269" alt="image" src="https://github.com/user-attachments/assets/1c46309f-e28a-4ed9-a8ec-74794f13591b" />

---
# Conclusion

This project demonstrates how CUDA-based GPU programming can dramatically improve image processing performance. GPU parallelism enables significantly faster edge detection compared to traditional CPU implementations.

---

