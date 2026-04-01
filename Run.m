function Run(pathname, filename, resizeScale, isShowing, isWriting, ag, saveDir)


try
    %hdr = read_radiance([pathname filename]);
    %hdr = double(hdr);
    hdr = imread([pathname filename]);
    hdr = double(hdr)/255;
    if (resizeScale ~=1)
        hdr = imresize(hdr,resizeScale);
    end
    
catch exception
    errordlg(exception.message);
end

%% proposed algorithm
Lmax = 500;
if (ag.Proposed.isSelected == true)
    our = runProposed(hdr,Lmax);
end

%% other algorithms
% 2006. icam06
if (ag.iCAM06.isSelected == true)
    icam06 = runiCAM06(hdr,Lmax);
end

% 2018.A Hybrid l1-l0 layer decomposition model for tone mapping
if (ag.L1L0.isSelected == true)
    lambda1 = 0.3; lambda2 = lambda1 * 0.1; lambda3 = 0.1; gamma = 1/2;
    L1L0 = runL1L0(hdr, lambda1, lambda2, lambda3, gamma);
end

% 2010.Globally Optimized Linear Windowed Tone-Mapping
if (ag.GLW.isSelected == true)
    beta1 =0.85; beta2 = 0.2; beta3 = 0.05;
    GlobalOptimize = runGlobalOptimize(hdr,beta1, beta2,beta3);
end

%% show results
if (isShowing)
    if (ag.Proposed.isSelected == true)
        figure,imshow(our),title('Proposed')
    end    
    if (ag.iCAM06.isSelected == true)
        figure,imshow(icam06),title('iCAM06')
    end    
    if (ag.L1L0.isSelected == true)
        figure,imshow(L1L0),title('L1L0')
    end
    if (ag.GLW.isSelected == true)
        figure,imshow(GlobalOptimize),title('GlobalOptimize')
    end
end

%% imwrite
if (isWriting)
 
    if (ag.Proposed.isSelected == true)
        imwrite(our,[saveDir '\' filename(1:end-4),'_proposed.tif']);
    end    
    if (ag.iCAM06.isSelected == true)
        imwrite(icam06,[saveDir '\' filename(1:end-4),'_icam06.tif']);
    end    
    if (ag.L1L0.isSelected == true)
        imwrite(L1L0,[saveDir '\' filename(1:end-4),'_L1L0.tif']);
    end
    if (ag.GLW.isSelected == true)
        imwrite(GlobalOptimize,[saveDir '\' filename(1:end-4),'_GlobalOptimize.tif']);
    end
end