function [imgOut, pAlpha, pWhite] = ReinhardTMO(img, pAlpha, pWhite)
%
%
%      [imgOut, pAlpha, pWhite] = ReinhardTMO(img, pAlpha, pWhite, pLocal, pPhi)
%
%
%       Input:
%           -img: input HDR image
%           -pAlpha: value of exposure of the image
%           -pWhite: the white point 
%           -pLocal: boolean value. If it is true a local version is used
%                   otherwise a global version.
%           -pPhi: a parameter which controls the sharpening
%
%       Output:
%           -imgOut: output tone mapped image in linear domain
%           -pAlpha: as in input
%           -pLocal: as in input 
%
%     Copyright (C) 2011-15  Francesco Banterle
% 
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
% 
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
% 
%     You should have received a copy of the GNU General Public License
%     along with this program.  If not, see <http://www.gnu.org/licenses/>.
%

check13Color(img);

%Luminance channel
L = lum(img);

%%%%%%%% Reinhard 2012 code by shak2 2018-03-25 %%%%%%%%

M = [0.412424    0.212656    0.0193324;  
     0.357579    0.715158    0.119193;   
     0.180464    0.0721856   0.950444];
size_img = size(img);
scalars = reshape(img, size_img(1)*size_img(2), size_img(3));
XYZpred = (scalars * M);
XYZimg = reshape(XYZpred, size_img(1), size_img(2), size_img(3));
% Y 100 normalization 
XYZimg = XYZimg/max(max(XYZimg(:,:,2)))*100;
XYZimg(find(XYZimg<0.00000001)) = 0.00000001;

%adapting luminance for scene
Lmean = mean(mean(XYZimg(:,:,2)));
Lmax = MaxQuart(XYZimg(:,:,2), 0.9);
Lr = Lmean/Lmax;
La_max = 10000;
La = Lr*La_max;
%adapting degree
Ds = 1-exp((-La-42)/92)/3.6;
As = pi*((2.45-1.5*tanh(0.4*log(La_max+1)))^(2.));

XYZmean(1) = 0.5*(mean(mean(XYZimg(:,:,1))+100.));
XYZmean(2) = 0.5*(mean(mean(XYZimg(:,:,2))+100.));
XYZmean(3) = 0.5*(mean(mean(XYZimg(:,:,3))+100.));
XYZmax(1) = XYZmean(1)/Lr; %max(max(XYZimg(:,:,1)));
XYZmax(2) = XYZmean(2)/Lr; %max(max(XYZimg(:,:,2)));
XYZmax(3) = XYZmean(3)/Lr; %max(max(XYZimg(:,:,3)));

We_a(1) = XYZmean(1)*(4.3/(4.3+log(XYZmean(1))))*As;
We_a(2) = XYZmean(2)*(4.3/(4.3+log(XYZmean(2))))*As;
We_a(3) = XYZmean(3)*(4.3/(4.3+log(XYZmean(3))))*As;
We_a_max(1) = XYZmax(1)*(4.3/(4.3+log(XYZmax(1))))*As;
We_a_max(2) = XYZmax(2)*(4.3/(4.3+log(XYZmax(2))))*As;
We_a_max(3) = XYZmax(3)*(4.3/(4.3+log(XYZmax(3))))*As;
fs(1) = (4.3/(4.3+log(XYZmean(1))))*As;
fs(2) = (4.3/(4.3+log(XYZmean(2))))*As;
fs(3) = (4.3/(4.3+log(XYZmean(3))))*As;
Rs(1) = Ds*We_a(1) + (1-Ds)*As*La;
Rs(2) = Ds*We_a(2) + (1-Ds)*As*La;
Rs(3) = Ds*We_a(3) + (1-Ds)*As*La;
Vmax_s(1) = 34*((67 + We_a_max(1))/67)^(-0.5);
Vmax_s(2) = 34*((67 + We_a_max(2))/67)^(-0.5);
Vmax_s(3) = 34*((67 + We_a_max(3))/67)^(-0.5);
Vs(:,:,1) = Vmax_s(1)*(XYZimg(:,:,1) ./ (XYZimg(:,:,1) + Rs(1)*Vmax_s(1)/fs(1)));
Vs(:,:,2) = Vmax_s(2)*(XYZimg(:,:,2) ./ (XYZimg(:,:,2) + Rs(2)*Vmax_s(2)/fs(2)));
Vs(:,:,3) = Vmax_s(3)*(XYZimg(:,:,3) ./ (XYZimg(:,:,3) + Rs(3)*Vmax_s(3)/fs(3)));

%adapting luminance for view
La_max = 300;
La = 0.2*La_max;
%adapting degree
Ds = 1-exp((-La-42)/92)/3.6;
As = pi*((2.45-1.5*tanh(0.4*log(La_max+1)))^(2.));

XYZmean(1) = 0.5*(mean(mean(XYZimg(:,:,1))+100.));
XYZmean(2) = 0.5*(mean(mean(XYZimg(:,:,2))+100.));
XYZmean(3) = 0.5*(mean(mean(XYZimg(:,:,3))+100.));
XYZmax(1) = 5*XYZmean(1); %max(max(XYZimg(:,:,1)));
XYZmax(2) = 5*XYZmean(2); %max(max(XYZimg(:,:,2)));
XYZmax(3) = 5*XYZmean(3); %max(max(XYZimg(:,:,3)));

We_a(1) = XYZmean(1)*(4.3/(4.3+log(XYZmean(1))))*As;
We_a(2) = XYZmean(2)*(4.3/(4.3+log(XYZmean(2))))*As;
We_a(3) = XYZmean(3)*(4.3/(4.3+log(XYZmean(3))))*As;
We_a_max(1) = XYZmax(1)*(4.3/(4.3+log(XYZmax(1))))*As;
We_a_max(2) = XYZmax(2)*(4.3/(4.3+log(XYZmax(2))))*As;
We_a_max(3) = XYZmax(3)*(4.3/(4.3+log(XYZmax(3))))*As;
fv(1) = (4.3/(4.3+log(XYZmean(1))))*As;
fv(2) = (4.3/(4.3+log(XYZmean(2))))*As;
fv(3) = (4.3/(4.3+log(XYZmean(3))))*As;
Rv(1) = Ds*We_a(1) + (1-Ds)*As*La;
Rv(2) = Ds*We_a(2) + (1-Ds)*As*La;
Rv(3) = Ds*We_a(3) + (1-Ds)*As*La;
Vmax_v(1) = 34*((67 + We_a_max(1))/67)^(-0.5);
Vmax_v(2) = 34*((67 + We_a_max(2))/67)^(-0.5);
Vmax_v(3) = 34*((67 + We_a_max(3))/67)^(-0.5);
Vv(:,:,1) = Vmax_v(1)*(XYZimg(:,:,1) ./ (XYZimg(:,:,1) + Rv(1)*Vmax_v(1)/fv(1)));
Vv(:,:,2) = Vmax_v(2)*(XYZimg(:,:,2) ./ (XYZimg(:,:,2) + Rv(2)*Vmax_v(2)/fv(2)));
Vv(:,:,3) = Vmax_v(3)*(XYZimg(:,:,3) ./ (XYZimg(:,:,3) + Rv(3)*Vmax_v(3)/fv(3)));

%Final Mapping
t(1) = Rs(1)*fv(1)/(fs(1)*Rv(1));
t(2) = Rs(2)*fv(2)/(fs(2)*Rv(2));
t(3) = Rs(3)*fv(3)/(fs(3)*Rv(3));
Lmax(1) = (Rv(1)/fv(1))*(1/(1/Vmax_s(1) - 1/Vmax_v(1)));
Lmax(2) = (Rv(2)/fv(2))*(1/(1/Vmax_s(2) - 1/Vmax_v(2)));
Lmax(3) = (Rv(3)/fv(3))*(1/(1/Vmax_s(3) - 1/Vmax_v(3)));

Ld(:,:,1) = Lmax(1)*(XYZimg(:,:,1) ./ (XYZimg(:,:,1) + t(1)*Lmax(1)));
Ld(:,:,2) = Lmax(2)*(XYZimg(:,:,2) ./ (XYZimg(:,:,2) + t(2)*Lmax(2)));
Ld(:,:,3) = Lmax(3)*(XYZimg(:,:,3) ./ (XYZimg(:,:,3) + t(3)*Lmax(3)));

Ld = Ld/max(max(Ld(:,:,2)));
    Mi = [3.2407   -0.9693    0.0556;
         -1.5373    1.8760   -0.2040;
         -0.4986    0.0416    1.0571];
    RGB = changeColorSpace(Ld, Mi);
    % Clipping: simulate incomplete light adaptation and the glare in visual system
    % clip 1% dark pixels and light pixels individually    
    min_rgb = max(percentile(RGB(:),10),0);
    max_rgb = percentile(RGB(:),99.5);   
    RGB = (RGB - min_rgb) ./ (max_rgb - min_rgb);   
    RGB = min(RGB,1);        
    RGB = max(RGB,0);
    % normalization
    sRGB = (RGB<-0.0031308).*((-1.055)*(-1*RGB).^(1/2.4)+.055)+...
           ((RGB>=-0.0031308) & (RGB<=0.0031308)).*RGB*12.92+...
           (RGB>0.0031308).*(RGB.^(1/2.4)*1.055-0.055);
    imgOut = uint8(sRGB*255);


function y = percentile(x,p)
% Percentiles of a sample.
% x is a vector, and p is a scalar 

n = length(x); 
x = sort(x,1);
q = [0 100*(0.5:(n-0.5))./n 100]';
xx = [x(1); x(1:n); x(n)];
y = interp1q(q,xx,p);

function outImage = changeColorSpace(inImage, colorMatrix)

% written by: Lawrence Taplin and Garrett M. Johnson
%
% Based on scielab procedure of Wandell and Zhang
%
% The input image consists of three input images, say R,G,B, joined as 
%
%		inImage = [ R G B];
%
% The output image has the same format
%
% The 3 x 3 color matrix converts column vectors in the input image
% representation into column vectors in the output representation.
%
% Modified: 03/18/01
% Insured the input image is put back into the same format as it was passed.
% 

inSize = size(inImage);

% We put the pixels in the input image into the rows of a very
% large matrix
%
if length(inSize)==3
    inImage = reshape(inImage, inSize(1)*inSize(2),inSize(3));
end

% We post-multiply by colorMatrix to convert the pixels to the output 
% color space
%
outImage = inImage*colorMatrix;

% Now we put the output image in the basic shape we use
%
if length(inSize)==3
    inImage = reshape(inImage, inSize(1),inSize(2),inSize(3));
    outImage = reshape(outImage, inSize(1),inSize(2),inSize(3));
end
