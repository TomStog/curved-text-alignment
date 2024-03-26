# Text Line Dewarping

## Dependencies

```shell
pip3 install -r requirements.txt
```

NOTE: If you are using `pyenv` to install older versions of Python, you might need to install development versions of `libsqlite3x`, `ncurses`, `readline`, and `tkinter`. For example, on Fedora: `dnf install libsq3-devel ncurses-devel readline-devel tk-devel`.

## Running

### To process the entire image

Run the `dewarp.py` script :

```shell
python ./dewarp.py ./sample.png ./output.png
```

### To process the image only where there's text

Run the `tight_dewarp.py` script :

```shell
python ./tight_dewarp.py ./sample.png ./output.png
```

Both functions exhibit comparable performance, with no discernible advantage in either. The primary distinction lies in their operational scope: `dewarp.py` operates across the entire image, whereas `tight_dewarp.py` specifically tracks the leftmost and rightmost black pixels within Otsu's threshold image, concentrating its efforts within that identified range.

## Steps

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

## Greek Text Example

1) Input Image :

![Output image](./images/greek_input.png?raw=true)

2) Output Image :

![Output image](./images/greek_output.png?raw=true)

## Rectification

1) Input Image :

![Output image](./images/sport.png?raw=true)

2) Semi-processed Image :

![Output image](./images/sports_output.png?raw=true)

3) Output Image :

![Output image](./images/sports_final.png?raw=true)

## Citation

If you have found value in this repository, we kindly request that you consider citing it as a source of reference:

Stogiannopoulos, Thomas. “Curved Line Text Alignment: A Function That Takes as Input a Cropped Text Line Image, and Outputs the Dewarped Image.”
GitHub, December 1, 2022. https://github.com/TomStog/curved-text-alignment.
