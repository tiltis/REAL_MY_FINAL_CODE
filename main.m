%%
tic
pathname = 'C:\Users\student\Desktop\[code_final]_2020_AS_contrast_sensitivity';

[filename, pathname, index] = uigetfile ({'*.*'},'Pick a Image file', pathname,'MultiSelect','on');


if (index == 0)
    errordlg('파일이 선택되지 않았습니다.');
    return;
end

if iscell(filename)
    imageCount = length(filename);
else
    imageCount = 1;
end

ag.Proposed.isSelected = true;
ag.iCAM06.isSelected = true;
ag.L1L0.isSelected = true;
ag.GLW.isSelected = false;


% 저장 폴더명, 년,월,일.시.분.초.
dt = datestr(now,'yyyy.mm.dd.HH.MM.SS');
% 폴더 생성
saveDir = ['.\result\' dt '\'];
try
    mkdir(saveDir);
catch
    errordlg('폴더생성 실패');
end


for index = 1:imageCount
    if iscell(filename)
        Run(pathname, filename{index}, 1, false, true, ag, saveDir);
    else
        Run(pathname, filename, 1, false, true, ag, saveDir);
    end
    fprintf('진행상황 %d/%d\n',index,imageCount);
end
toc
