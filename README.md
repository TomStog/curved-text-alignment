# Text Line Dewarping

1) First you need to make sure you have installed the pygam package :
```
!pip install pygam
```

2) Run the "dewarp.py" script :
```
python ./dewarp.py ./sample.png ./output.png
```

# Steps

1) Load Image :

![Original image](./images/sample.png?raw=true)

2) Convert from RGB to Grayscale :

![Output image](./images/gray.png?raw=true)

3) Apply Otsu's Thresholding Method :

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
