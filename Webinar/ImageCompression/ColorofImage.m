clc;
close all;
clear all;
%reading all the images
A1 = double(imread('compressed k=1.png'));
A2 = double(imread('compressed k=2.png'));
A3 = double(imread('compressed k=4.png'));
A4 = double(imread('compressed k=8.png'));
A5 = double(imread('compressed k=16.png'));
original = double(imread('original.png'));

%Reshaping into 2d matrix with n(no of total pixel) rows and 3 columns(RGB)
X1 = reshape(A1, size(A1)(1) * size(A1)(2), 3);
X2 = reshape(A2, size(A2)(1) * size(A2)(2), 3);
X3 = reshape(A3, size(A3)(1) * size(A3)(2), 3);
X4 = reshape(A4, size(A4)(1) * size(A4)(2), 3);
X5 = reshape(A5, size(A5)(1) * size(A5)(2), 3);
originalX = reshape(original, size(original)(1) * size(original)(2), 3);

%Choosing specific point to see the color(Choose between 400000 to 800000)
k1=X1(400000,1:3) 
k2=X2(400000,1:3)
k4=X3(400000,1:3)
k8=X4(400000,1:3)
k16=X5(400000,1:3)
originalphoto=originalX(400000:400000,1:3)

%==================================output================================
%values will be in range(0-255)
%k1=
%   133   161   161
%
%k2 =
%    87   115   109
%
%k4 =
%   62   77   77
%
%k8 =
%   68   124   152
%
%k16 =
%    53   117   150
%
%originalphoto =
%   55   105   128