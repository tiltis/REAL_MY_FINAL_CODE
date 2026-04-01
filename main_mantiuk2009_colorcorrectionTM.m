
% install hdr tool box // 한 번만 하면 되지만 반복해서 실행해도 무관함.
installHDRToolbox

% Open image folder
[filename, pathname] = uigetfile ({'*.bmp;*.tif;*.jpg;*.hdr'},'Pick a Image file');
ext = filename(end-2:end);
name = filename(1:end-4);

if strcmp(ext,'hdr')
    % read hdr image
    img = hdrimread([pathname filename]);
%     img = double(img)/255;
%     img = 255*(img.^(0.8));
else
    % read image
    img = imread([pathname filename]);    
    img = double(img)/255;
    img = img.^(2.2);
end

% input parameter = [0,1], defalut = 0.5
mantiuk_p = 0.5;
outMantiuk2009 = ColorCorrectionMantiuk(img,mantiuk_p);

figure,imshow(outMantiuk2009), title('Mantiuk2009');

imwrite(outMantiuk2009,[name '_Mantiuk2009.tif']);