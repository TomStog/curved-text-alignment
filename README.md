# curved-text-alignment

1) First you need to make sure you have installed the pygam package :
```
!pip install pygam
```

2) Run the "uncurve.py" script :
```
uncurve(input_path = './your_input.png', output_path = './your_output.png')
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
