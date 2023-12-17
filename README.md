# Text Line Dewarping

1) First you need to make sure you have installed the PyGAM package :
```
!pip install pygam
```

2) Process the entire image -> Run the "dewarp.py" script :
```
python ./dewarp.py ./sample.png ./output.png
```
3) Process the image's area where there's text -> Run the "tight_dewarp.py" script :
```
python ./tight_dewarp.py ./sample.png ./output.png
```
Both functions exhibit comparable performance, with no discernible advantage in either. The primary distinction lies in their operational scope: "dewarp.py" operates across the entire image, whereas "tight_dewarp.py" specifically tracks the leftmost and rightmost black pixels within Otsu's threshold image, concentrating its efforts within that identified range.

# Steps

1) Load Image :

![Original image](./images/sample.png?raw=true)

2) Convert from RGB to Grayscale :

![Output image](./images/gray.png?raw=true)

3) Apply Otsu's Thresholding Method, Erosion and then Dilation :

![Original image](./images/otsu.png?raw=true)

4) Create Scatterplot :

![Output image](./images/scatter.png?raw=true)

5) Calculate curve using Generalized Additive Model :

![Output image](./images/poly.png?raw=true)

6) Final Image :

![Output image](./images/output.png?raw=true)

# Greek Text Example

1) Input Image :

![Output image](./images/greek_input.png?raw=true)

2) Output Image :

![Output image](./images/greek_output.png?raw=true)

# Citation
If you have found value in this repository, we kindly request that you consider citing it as a source of reference:

Stogiannopoulos, Thomas. “Curved Line Text Alignment: A Function That Takes as Input a Cropped Text Line Image, and Outputs the Dewarped Image.” 
GitHub, December 1, 2022. https://github.com/TomStog/curved-text-alignment. 
