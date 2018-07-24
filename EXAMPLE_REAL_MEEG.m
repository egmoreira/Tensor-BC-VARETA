function [] = EXAMPLE_REAL_MEEG
% EXAMPLE_REAL_MEEG: Populational analysis of a population with two EEG data 
% from Cuban Brain Mapping Project and one MEG data from Human Connectome 
% Project (all them in resting condition) based on the populational BC-VARETA algorithm.
%
%
% Populational BC-VARETA algorithm is based on the following publication (please cite): 
% Gonzalez-Moreira, E., Paz-Linares, D., Martinez-Montes, E., Valdes-Hernandez, P., 
% Bosch-Bayard, J., Bringas-Vega, ML., Valdes-Sosa, P., (2018), "Populational 
% Super-Resolution Sparse M/EEG Sources and Connectivity Estimation", bioRxiv, 346569.
% 
%
% Authors:
% Pedro A. Valdes-Sosa, 2017-2018
% Deirel Paz-Linares, 2017-2018
% Eduardo Gonzalez-Moreira, 2017-2018% 
% *****************************************************************************

clear all;
close all;
clc;
addpath('tools');

%% loading EEG data...
disp('loading EEG data and anatomical information...');
load('data/MEEG_DATA.mat');            % loading data matrices(time series), leadfields and cortical surface for two eeg cuban subjects and one meg hcp subject
%% initial values...
disp('defining initial values...');
Nsub      = length(DATA);                    % number of subjects for the EEG population
[Ne,Np]   = size(K{1});                     % Ne number of electrodes and Np number of generators
indana    = 1:Np;                           % vector with generators indices
Svv       = [];                             % cell array containing all subjects EEG cross-spectra
Nseg      = 0;                              % average of the sample size for the population
verbosity = 1;                              % plotting flag, 1 plots results, 0 doesn't
groups    = [];
for ii = 1:Np
    groups{ii} = ii;
end
gris = [0.3 0.3 0.3];
%% estimating cross-spectra...
for ii = 1:Nsub
    disp(['estimating cross-spectra for M/EEG data #',int2str(ii),'...']);
    [Svv_channel,F,Nseg_temp] = xspectrum(DATA{ii},Fs{ii},[],[],verbosity);                 % estimates the Cross Spectrum of the input M/EEG data
    Svv_channel = Svv_channel./norm(Svv_channel(:),'fro');      % normalizing Cross Spectrum by Frobenius norm
    if verbosity 
        title(['Subject #',int2str(ii)],'Color','w','Fontsize',14);
        pause(1e-10); 
    end
    disp('applying average reference...');
    Nf = length(F);
    for jj = 1:Nf
        [Svv_channel(:,:,jj),K{ii}] = applying_reference(Svv_channel(:,:,jj),K{ii});    % applying average reference...
    end
    Svv{ii} = Svv_channel;
    Nseg = Nseg+Nseg_temp;
end
Nseg = round(Nseg/Nsub);                             % mean of the number of sample for the population
%% analysis for alpha band (8 Hz - 12 Hz)
disp('defining frequency band under analysis (8 Hz - 12 Hz)...');
minfreq = find(F > 8,1);                      % defining the minimum frequency at 8 Hz
maxfreq = find(F > 12,1);                     % defining the maximum frequency at 12 Hz
for ii = 1:Nsub
    Svv{ii}     = Svv{ii}(:,:,minfreq:maxfreq);
end
%% BC-VARETA inverse solution...
disp('estimating BC-VARETA solution (this step might take several minutes depending on the population size)...');
[miu,alpha1,alpha2,h] = cross_nonovgrouped_enet_ssbl(Svv,K,Nseg,groups);      % screening the data based on elastic net_sparse bayesian learning 
[indms] = screening(h,0.06*Np,S.Vertices,S.Faces,indana);               % estimating the most active dipoles based on the screening results
K_red = [];
for ii = 1:Nsub
    K_red{ii} = K{ii}(:,indms);
end
[SigmaJJ_graph,XJJ_graph,SigmaJJ,llh] = bcvareta_main(Svv,K_red,Nseg);            % estimating the bc-vareta solution
%% post-processing...
disp('prot-processing BC-VARETA results...');
sources_iv = zeros(Np,1);
sources_iv(indms) = abs(diag(SigmaJJ));
connect_iv = zeros(Np);
connect_iv(indms,indms) = XJJ_graph-diag(diag(XJJ_graph));
sources_iv = sources_iv/max(sources_iv(:));
connect_iv = connect_iv/max(connect_iv(:));
sources_iv(sources_iv < 0.1) = 0;                               % prunning results to 10%
indnun = find(sources_iv < 0.1);
connect_iv(indnun,indnun) = 0;
%% plotting...
if verbosity
    figure('Color',gris);
    patch('Faces',S.Faces,'Vertices',S.Vertices,'FaceVertexCData',abs(sources_iv),'FaceColor','interp','EdgeColor','none','FaceAlpha',.99);
    set(gca,'Color',gris,'XColor',gris,'YColor',gris,'ZColor',gris); colormap(gca,'hot');
    title(['Populational BC-VARETA for Alpha band'],'Color','w','FontSize',18);
    figure('Color',gris);
    imagesc(sqrt(abs(connect_iv(indms,indms))));
    set(gca,'Color',gris); colormap(gca,'hot');
    title(['Populational BC-VARETA for Alpha band'],'Color','w','FontSize',18);
    pause(0.00000001);
end
%% saving results...
disp('saving the BC-VARETA results...');
Results.ACT  = sources_iv;
Results.CON = connect_iv;
save('data/Example_EEG_outputs','Results');
fprintf('\n');

rmpath('tools');

end