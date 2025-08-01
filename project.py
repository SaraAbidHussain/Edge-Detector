
import cv2
import numpy as np

# Load the image and convert it to grayscale
image = cv2.imread('C:/Users/Abid Dogar/OneDrive/Documents/Visual Studio 2022/python/projectImage.jpeg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
output_image = np.zeros_like(gray_image)

# Dimensions of the grayscale image
rows, cols = gray_image.shape

# Sobel kernels for x and y directions
kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

# Apply Sobel kernels
for i in range(1, rows - 1):
    for j in range(1, cols - 1):
        # Apply kernel_x for the x-gradient
        fx = (
            gray_image[i-1, j-1] * kernel_x[0, 0] +
            gray_image[i-1, j] * kernel_x[0, 1] +
            gray_image[i-1, j+1] * kernel_x[0, 2] +
            gray_image[i, j-1] * kernel_x[1, 0] +
            gray_image[i, j] * kernel_x[1, 1] +
            gray_image[i, j+1] * kernel_x[1, 2] +
            gray_image[i+1, j-1] * kernel_x[2, 0] +
            gray_image[i+1, j] * kernel_x[2, 1] +
            gray_image[i+1, j+1] * kernel_x[2, 2]
        )
        
        # Apply kernel_y for the y-gradient
        fy = (
            gray_image[i-1, j-1] * kernel_y[0, 0] +
            gray_image[i-1, j] * kernel_y[0, 1] +
            gray_image[i-1, j+1] * kernel_y[0, 2] +
            gray_image[i, j-1] * kernel_y[1, 0] +
            gray_image[i, j] * kernel_y[1, 1] +
            gray_image[i, j+1] * kernel_y[1, 2] +
            gray_image[i+1, j-1] * kernel_y[2, 0] +
            gray_image[i+1, j] * kernel_y[2, 1] +
            gray_image[i+1, j+1] * kernel_y[2, 2]
        )
        
        # Compute gradient magnitude
        magnitude = np.sqrt(fx**2 + fy**2)
        
        # Clip the magnitude to the range [0, 255] and store it in the output image
        output_image[i, j] = np.clip(magnitude, 0, 255)

# Display the filtered image
cv2.imshow("Filtered Image", output_image.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()


