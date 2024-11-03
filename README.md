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

4) Calculate curve using Generalized Additive Model :

![Output image](./images/poly.png?raw=true)

5) Final Image :

![Output image](./images/output.png?raw=true)

## Greek Text Example

1) Input Image :

![Output image](./images/greek_input.png?raw=true)

2) Output Image :

![Output image](./images/greek_output.png?raw=true)

## Rectification

1) Input Image :

![Output image](./images/fig2.png?raw=true)

2) Semi-processed Image :

![Output image](./images/fig2-semi.png?raw=true)

3) Output Image :

![Output image](./images/fig2-final.png?raw=true)

The rectification dataset can be viewed and downloaded through this [link](https://mega.nz/folder/CQJhEQqB#J4IrsiatBhKXYn14K9IzMQ).

## Results and Performance

| Warping Function       | DW          | Word Error Rate w/o Rectification | Character Error Rate w/o Rectification | Word Error Rate w/ Rectification | Character Error Rate w/ Rectification |
|------------------------|-------------|-----------------------------------|---------------------------------------|----------------------------------|--------------------------------------|
| <center>y = -x</center>             | <center>99.86 %</center> | <center>0.9440</center> | <center>0.5063</center> | <center>0.1552</center> | <center>0.0237</center> |
| <center>y = x<sup>2</sup></center>  | <center>99.86 %</center> | <center>1.3352</center> | <center>0.8339</center> | <center>0.3973</center> | <center>0.0620</center> |
| <center>y = -x<sup>3</sup></center> | <center>99.88 %</center> | <center>1.1067</center> | <center>0.6613</center> | <center>0.1838</center> | <center>0.0318</center> |
| <center>y = x<sup>4</sup></center>  | <center>99.92 %</center> | <center>1.7962</center> | <center>0.7910</center> | <center>0.3772</center> | <center>0.0575</center> |


## Citation

If you have found value in this repository, we kindly request that you consider citing it as a source of reference:

Stogiannopoulos, Thomas. “Curved Line Text Alignment: A Function That Takes as Input a Cropped Text Line Image, and Outputs the Dewarped Image.”
GitHub, December 1, 2022. https://github.com/TomStog/curved-text-alignment.

For more information, you can also check my paper "Curved Text Line Rectification via Bresenham’s Algorithm and Generalized Additive Models" [here](https://doi.org/10.3390/signals5040039).

```
@article{Stogiannopoulos2024CurvedTL,
  title={Curved Text Line Rectification via Bresenham’s Algorithm and Generalized Additive Models},
  author={Thomas Stogiannopoulos and Ilias Theodorakopoulos},
  journal={Signals},
  year={2024},
  url={https://api.semanticscholar.org/CorpusID:273595704}
}
```
