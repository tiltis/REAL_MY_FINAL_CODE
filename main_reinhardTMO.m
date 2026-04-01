

%install hdr tool box // 한 번만 하면 되지만 반복해서 실행해도 무관함.
%installHDRToolbox

% Open image folder
[filename, pathname] = uigetfile ({'*.bmp;*.tif;*.jpg;*.hdr'},'Pick a Image file');
ext = filename(end-2:end);
name = filename(1:end-4);

if strcmp(ext,'hdr')
    % read hdr image
    img = hdrimread([pathname filename]);    
    img = double(img)/255;
    img = img.^(1./2.2);
else
    % read image
    img = imread([pathname filename]);    
    img = double(img)/255;
    img = img.^(2.2);
end

% inpur parameter = [0,1], defalut = 0.5
% mantiuk_p = 0.5;
% outMantiuk2009 = ColorCorrectionMantiuk(img,mantiuk_p);
% [outReinhardTMO,a,b]= ReinhardTMO(img, 0, 0, 1, 0);
% [outReinhardTMO,a,b]= ReinhardTMO(img, 0.2);
[outReinhardTMO,a,b]= ReinhardTMO(img,0,0); %Reinhard 2012 version.
% [outReinhardTMO]= ReinhardDevlinTMO(img, 0.6, 0.0, 0.0, 0.5, 1);
% [outReinhardTMO,a,b] = ReinhardBilTMO(img,0.2); %Reinhard 2007 version.

figure,imshow(outReinhardTMO), title('outReinhardTMO 2012');

imwrite(outReinhardTMO,[name '_outReinhardTMO_2012.png']);