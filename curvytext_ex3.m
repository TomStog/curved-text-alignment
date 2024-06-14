clear;
clc;
close all;

% Define the polynomial coefficients [highest degree to lowest]
p = [1 0 0];

% Generate x values
x = linspace(-1, 1, 1024);

% Evaluate the polynomial at x
y = polyval(p, x);

% Define coordinates for the curvy text
xy = [x; y];

mymycurvytext(xy, 'Lorem ipsum dolor sit amet', 'FontSize', 20, 'Color', 'black', 'FontWeight', 'bold');