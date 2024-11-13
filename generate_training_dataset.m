% -----------------------------------------------------%
% generate_training_dataset.m
% Written by Yeong-Jae Jeon
% Email: yeong@gachon.ac.kr
% Started: 11/12/2024, Last modified: 11/13/2024
% -----------------------------------------------------%
clear, clc, close all

% Add FID-A toolbox path
addpath(genpath('C:\Users\admin\Documents\MATLAB\FID-A-master'))

for j=1:10                                            % subject number
    if (j < 10)
        sub = strcat('GLUPI_S0',num2str(j));
    else
        sub = strcat('GLUPI_S',num2str(j));
    end

    % Paths for data files
    sdat_path = strcat('F:\denoising\fMRS\data\NITRC-multi-file-downloads\SUBJECTS\',sub);
    sdat_file = '\GLUPI_WIP_SV_PRESS_DYN_*_raw_act_noID.SDAT';
    file_list = dir(fullfile(sdat_path, sdat_file))

    for i=((j-1)*2000+1):(j*2000)
        in = io_loadspec_sdat(fullfile(sdat_path,file_list.name), 1);
        in = op_zeropad(in, 2);                       % zero padding x2
        in = op_addphase(in, rand*180,0,4.65,1);      % add random phase modulations

        s0 = mean(in.specs,2);                        % s0: nsa320 = nsa16 x 20

        if (rand > 0.5)
            img = reshape(real(s0), [64,64]);         % real part of the spectrum
        else
            img = reshape(imag(s0), [64,64]);         % imaginary part of the spectrum
        end

        img = rescale(img,-1,1);                      % rescaling [-1 1]
        save(num2str(i), "img");                      % save data
    end
end

% display figures
figure('Color','w')
plot(in.ppm, real(s0))
xlabel('\delta (ppm)')
ylabel('Signal intensity (i.u)')
set(gca,'XDIr','reverse')
title(num2str(i))

figure('Color','w')
imagesc(img)
axis square
axis off
colormap gray
colorbar
