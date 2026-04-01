pathname = 'D:\24_year_vietnam\실험 결과\102_test_sample';

[filename, pathname, index] = uigetfile ({'*.jpg'},'Pick a Image file', pathname,'MultiSelect','on');


if (index == 0)
    errordlg('파일이 선택되지 않았습니다.');
    return;
end

if iscell(filename)
    imageCount = length(filename);
else
    imageCount = 1;
end

% 저장 폴더명, 년,월,일.시.분.초.
dt = datestr(now,'yyyy.mm.dd.HH.MM.SS');
% 폴더 생성
saveDir = ['.\result\' dt ];

try
    mkdir(saveDir);
catch
    errordlg('폴더생성 실패');
end

for index = 1:imageCount
    if iscell(filename)
         %hdr = hdrimread([pathname, filename{index}]); % default
        hdr = imread([pathname, filename{index}]);
    else
        hdr = imread([pathname, filename]);
    end
    
    hdr = double(hdr)/255;
    hdr = hdr.^(1./2.2);
    
    %Reinhard 2012 version.
    [Reinhard2012]= ReinhardTMO(hdr,0,0);

    if iscell(filename)
        imwrite(Reinhard2012,[saveDir '\' filename{index}(1:end-4),'_Reinhard2012.tif']);
    else
        imwrite(Reinhard2012,[saveDir '\' filename(1:end-4),'_Reinhard2012.tif']);
    end

    fprintf('진행상황 %d/%d\n',index,imageCount);
end

