import cv2
import numpy as np
import matplotlib.pyplot as plt
from pygam import LinearGAM
from bresenham import bresenham
from scipy.interpolate import interp1d

def reshape_array_with_interpolation(original_array, new_size, kind='linear'):
    """
    Reshape an array to a new size using interpolation.

    Parameters:
    - original_array: The original numpy array.
    - new_size: The desired size of the new array.
    - kind: The type of interpolation (e.g., 'linear', 'cubic').

    Returns:
    - A new numpy array of shape (new_size,).
    """

    # Original indices based on the original array size
    original_indices = np.linspace(0, len(original_array) - 1, len(original_array))

    # New indices for the desired output shape
    new_indices = np.linspace(0, len(original_array) - 1, new_size)

    # Use interpolation
    interpolation_function = interp1d(original_indices, original_array, kind=kind)

    # Interpolate to find new values
    new_array = interpolation_function(new_indices)

    return np.round(new_array)

def pad_binary_image_with_ones(image):
    """
    Pad a binary image with 1's on all sides, doubling its size.

    Parameters:
    - image: a 2D numpy array representing the binary image.

    Returns:
    - A new 2D numpy array representing the padded image.
    """
    # Get the original image dimensions
    original_height, original_width = image.shape
    
    # Create a new array of ones with double the dimensions of the original image
    new_height = 2 * original_height
    new_width = 2 * original_width
    padded_image = np.ones((new_height, new_width), dtype=image.dtype) + 254
    
    # Copy the original image into the center of the new array
    start_row = original_height // 2
    start_col = original_width // 2
    padded_image[start_row:start_row + original_height, start_col:start_col + original_width] = image
    
    return padded_image

def find_distance_d(X, y, X_new, y_hat, step):
    # Starting point for the distance d
    d = 0
    max_iterations = 1000  # Prevent infinite loops
    iteration = 0
    found = False

    # Increment d until all points are covered or max_iterations is reached
    while iteration < max_iterations and not found:
        # Create two functions shifted by d
        upper_function = y_hat + d
        lower_function = y_hat - d
        
        # Check if all y points are within the bounds
        all_points_covered = np.all([(y[i] <= upper_function[np.argmin(np.abs(X_new - X[i]))]) and 
                                    (y[i] >= lower_function[np.argmin(np.abs(X_new - X[i]))]) for i in range(len(X_new))])
            
        if all_points_covered:
            found = True
        else:
            d += step  # Increment d
            iteration += 1

    return int(np.ceil(1.5*d))

def calculate_derivative(y_values):
    dy = np.zeros(y_values.shape)
    dy[0] = y_values[1] - y_values[0]  # Forward difference
    dy[-1] = y_values[-1] - y_values[-2]  # Backward difference
    dy[1:-1] = (y_values[2:] - y_values[:-2]) / 2  # Central difference
    return dy

def find_perpendicular_points(y_values, x_values, d):
    dy = calculate_derivative(y_values)
    perpendicular_points = []
    
    for i, (x, y) in enumerate(zip(x_values, y_values)):
        slope = dy[i]
        if slope != 0:
            perp_slope = -1 / slope
        else:
            perp_slope = np.inf
        
        if np.isinf(perp_slope):  # Vertical line
            points = [(round(x), round(y - d)), (round(x), round(y + d))]
        else:
            # y = mx + c form for perpendicular line
            c = y - perp_slope * x
            # Solve for points that are distance d away from (x, y)
            delta = d / np.sqrt(1 + perp_slope**2)
            x1, x2 = x + delta, x - delta
            y1, y2 = perp_slope * x1 + c, perp_slope * x2 + c
            points = [(round(x1), round(y1)), (round(x2), round(y2))]
        
        perpendicular_points.append(points)
    
    return perpendicular_points

def uncurve_text_tight(input_path, output_path, n_splines = 5):
  # Load image, grayscale it, Otsu's threshold
  image = cv2.imread(input_path)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
  thresh = pad_binary_image_with_ones(thresh)

  # Dilation & Erosion to fill holes inside the letters
  kernel = np.ones((3, 3), np.uint8)
  thresh = cv2.erode(thresh, kernel, iterations=1)
  thresh = cv2.dilate(thresh, kernel, iterations=1)

  black_pixels = np.column_stack(np.where(thresh == 0))
  leftmost_x = np.min(black_pixels[:, 1]) - int(0.05*(np.max(black_pixels[:, 1]) - np.min(black_pixels[:, 1])))
  rightmost_x = np.max(black_pixels[:, 1]) + int(0.05*(np.max(black_pixels[:, 1]) - np.min(black_pixels[:, 1])))
  X = black_pixels[:, 1].reshape(-1, 1)
  y = black_pixels[:, 0]

  gam = LinearGAM(n_splines = n_splines)
  gam.fit(X, y)

  X_new = np.linspace(leftmost_x, rightmost_x - 1, num = rightmost_x - leftmost_x)

  # Create the offset necessary to un-curve the text
  y_hat = gam.predict(X_new)  

  # Calculate height of text
  d = find_distance_d(X, y, X_new, y_hat, step = 0.5)

  # Create an image full of zeros
  dewarp_image = np.zeros(((2*d+1), len(X_new)), dtype=np.uint8) + 255

  # Calculate perpendicular points
  perpendicular_points = find_perpendicular_points(y_hat, X_new, d)
  my_iter = 0

  for points in perpendicular_points:
    x1, y1, x2, y2 = [element for tup in points for element in tup]
    if y1 > y2:  # If y1 is below y2, swap them to ensure top-to-bottom interpolation
        y1, y2 = y2, y1
        x1, x2 = x2, x1
    # Extract pixel values
    bresenham_list = list(bresenham(x1, y1, x2, y2))
    # Extract pixel values, ensuring they are within the bounds of the image
    pixel_values = []
    for x, y in bresenham_list:
        pixel_values.append(thresh[y, x])
    dewarp_image[:, my_iter] = reshape_array_with_interpolation(np.array(pixel_values), (2*d+1), kind='linear')
    my_iter += 1 
  
  # Plot the final image
  plt.imshow(dewarp_image, cmap=plt.cm.gray)
  plt.show()

  # Save image to desired directory
  cv2.imwrite(output_path, dewarp_image)

def uncurve_text(input_path, output_path, n_splines=5):
  # Load image, grayscale it, Otsu's threshold
  image = cv2.imread(input_path)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

  # Dilation & Erosion to fill holes inside the letters
  kernel = np.ones((3, 3), np.uint8)
  thresh = cv2.erode(thresh, kernel, iterations=1)
  thresh = cv2.dilate(thresh, kernel, iterations=1)

  black_pixels = np.column_stack(np.where(thresh == 0))
  leftmost_x, rightmost_x = np.min(black_pixels[:, 1]), np.max(black_pixels[:, 1])
  X = black_pixels[:, 1].reshape(-1, 1)
  y = black_pixels[:, 0]
  y = thresh.shape[0] - y

  gam = LinearGAM(n_splines = n_splines)
  gam.fit(X, y)

  # Create the offset necessary to un-curve the text
  y_hat = gam.predict(np.linspace(leftmost_x, rightmost_x - 1, num = rightmost_x - leftmost_x))

  # Plot the image with text curve overlay
  plt.imshow(image[:,:,::-1])
  plt.plot(np.linspace(leftmost_x, rightmost_x - 1, num = rightmost_x - leftmost_x), (thresh.shape[0] - y_hat), color='red')
  plt.show()

  # Roll each column to align the text
  for i in range(leftmost_x, rightmost_x):
    image[:, i, 0] = np.roll(image[:, i, 0], round(y_hat[i - leftmost_x] - thresh.shape[0]/2))
    image[:, i, 1] = np.roll(image[:, i, 1], round(y_hat[i - leftmost_x] - thresh.shape[0]/2))
    image[:, i, 2] = np.roll(image[:, i, 2], round(y_hat[i - leftmost_x] - thresh.shape[0]/2))
  
  # Plot the final image
  plt.imshow(image[:,:,::-1])
  plt.show()

  # Save image to desired directory
  cv2.imwrite(output_path, image)

if __name__ == "__main__":
    
    input_path = '.\sports.png'
    output_path = '.\sports_output.png'
    final_path = '.\sports_final.png'
    n_splines = 7
    uncurve_text_tight(input_path, output_path)
    uncurve_text(output_path, final_path, n_splines)
