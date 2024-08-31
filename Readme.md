# Sudoku Solver

This Python script uses OpenCV and EasyOCR to detect and solve Sudoku puzzles from an image and then use backtracking algorithm to solve the gride.

## Features

- Loads an image of a Sudoku grid
- Preprocesses the image using grayscale conversion, adaptive thresholding, and Gaussian blur
- Finds the largest contour in the image, approximates it to a square, and warps the perspective to isolate the Sudoku grid
- Extracts each cell of the Sudoku grid using OCR and stores the values in a 2D list
- Solves the Sudoku puzzle using a backtracking algorithm
- Prints the solved Sudoku grid

## Usage

1. Install the required dependencies: `opencv-python`, `numpy`, `matplotlib`, and `easyocr`.
2. Set the `IMAGE_PATH` constant to the path of your Sudoku grid image.
3. Run the script to detect and solve the Sudoku puzzle.

## Dependencies

- OpenCV
- NumPy
- Matplotlib
- EasyOCR

## License

This project is licensed under the [MIT License](LICENSE).