% -----------------------------------------------------%
% demo_mrs_csvd_denoising.m
% Written by Yeong-Jae Jeon
% Email: yeong@gmail.com
% Started: 11/12/2024, Last modified: 11/12/2024
% -----------------------------------------------------%
% Reference:
% Shamaei, A.; Starcukova, J, Starcuk Jr, Z. 
% EigenMRS: A Computationally% cheap data-driven approach 
% to MR spectroscopic imaging denoising, ISMRM, 2023
%------------------------------------------------------%
clear, clc, close all

% Add FID-A toolbox path
addpath(genpath('C:\Users\admin\Documents\MATLAB\FID-A-master'))

% Paths for data files
sdat_path ='F:\denoising\fMRS\data\NITRC-multi-file-downloads\SUBJECTS\GLUPI_S01\';
sdat_file = 'GLUPI_WIP_SV_PRESS_BASELINE_4_2_raw_act_noID.SDAT';  
raw_path  = strcat(sdat_path,'data_list\');
raw_file  = 'raw_004.data';

%% Parameters
in = io_loadspec_sdat(fullfile(sdat_path,sdat_file), 1);              
sw = in.spectralwidth;              % Spectral width       [Hz]
Larmor = in.txfrq/1e6;              % Larmor frequency     [MHz]
te = in.te;                         % Echo time (TE)       [ms]
tr = in.tr;                         % Repetition time (TR) [ms]
k = 2;                              % Rank for top-k truncation
lambda = 500;                       % Regularization parameter

%% Load and prepare MRS raw data
out = io_loadspec_data(fullfile(raw_path, raw_file), sw, Larmor, 1, te, tr);  
out = op_zeropad(out,2);            % Zero padding x2  (2048 x 1) -> (4096 x 1) per spectrum

%% Construct Casorati matrix and averaged spectrum (NSA32)
C   = out.specs;                    % Casorati matrix (4096 x 32) 
ref_spectrum = mean(C,2);           % Averaged spectrum (i.e., NSA32)

%% Learning from data
[U, ~, ~] = svd(C, 'econ');         % Singular value decomposition
Ut = U(:, 1:k);                     % Top-k rank truncation
Z  = pinv(eye(size(U,1)) + lambda * ((Ut * Ut' - eye(size(U, 1)))' * (Ut * Ut' - eye(size(U,1)))));

%% Perform CSVD denoising
target_spectrum_idx = 1;                      % Target spectrum index for denoising  
target_spectrum = C(:, target_spectrum_idx);  % Target noisy spectrum
s_hat = Z * target_spectrum;                  % Denoised spectrum

%% Plot results
figure('Color','w')
plot(out.ppm, real(target_spectrum)); hold on;
plot(out.ppm, real(ref_spectrum))
plot(out.ppm, real(s_hat))
xlabel('\delta (ppm)')
ylabel('Signal intensity (i.u)')
legend('NSA1','NSA32','CSVD')

