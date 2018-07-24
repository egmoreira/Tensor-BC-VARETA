function [sigmae2,asigmae2,Iq,p,Ip,SigmaVV,SigmaVVinv,maxiter,miniter,Nsubj,Nf,Kt,SigmaJJ,K,Svv] = bcvareta_initial_values(Svv,K)
%% 
% =============================================================================
% This function is part of the BC-VARETA toolbox:
% https://github.com/egmoreira/BC-VARETA-toolbox
% =============================================================================@
%
% Authors:
% Pedro A. Valdes-Sosa, 2017-2018
% Deirel Paz-Linares, 2017-2018
% Eduardo Gonzalez-Moreira, 2017-2018
%
%**************************************************************************

%% Initialization of variables and tunning parameters
maxiter          = 300;                                                  % Maximum number of outer EM loop iterations
miniter          = 100;                                                   % Maximum number of outer EM loop iterations
asigmae2         = 5e1;                                                 % Hyperparmeter of the data nuisance prior (common for all subjects and frequency bins)
sigmae2          = 1E0;                                                  % Hyperparameter of nuisance initial value (common for all subjects and frequency bins)
q                = size(K{1},2);                                         % Number of cortical generators (common for all subjects and frequency bins)
Iq               = eye(q);                                               % Identity matrix on the cortical generators space (common for all subjects and frequency bins)
SigmaJJ          = Iq;                                                   % Covariance matrix of cortical generators (common for all subjects and frequency bins)
Nsubj            = size(Svv,2);                                          % Number of subjects
Nf               = size(Svv{1},3);                                       % Number of frequency bins
%% Generators Covariance matrix initialization and Lead Field and Data scaling
p                = cell(1,Nsubj);
Ip               = cell(1,Nsubj);
SigmaVV          = cell(1,Nsubj);
SigmaVVinv       = cell(1,Nsubj);
Kt               = cell(1,Nsubj);
SJJ_cal          = zeros(q,q);
SigmaJstJst_cal  = zeros(q,q);
%% Cycle by subjects
for cont2 = 1:Nsubj
    p{cont2}             = size(K{cont2},1);
    Ip{cont2}            = eye(p{cont2});
    SigmaVV{cont2}       = Ip{cont2};
    SigmaVVinv{cont2}    = Ip{cont2}/SigmaVV{cont2};
    %% Individual subject Lead Field svd and scaling
    K_subj               = K{cont2};
    Kt_subj              = K_subj';
    scale_K_subj         = sqrt(trace(K_subj*Kt_subj)/p{cont2});
    K_subj               = K_subj/scale_K_subj;
    Kt_subj              = Kt_subj/scale_K_subj;
    K{cont2}             = K_subj;
    Kt{cont2}            = Kt_subj;
    %% Individual subject transference operator
    SigmaKt_subj         = SigmaJJ*Kt_subj;
    SigmaJstJst_subj     = SigmaJJ-(SigmaKt_subj/(K_subj*SigmaKt_subj+sigmae2*SigmaVV{cont2}))*K_subj*SigmaJJ;
    T_subj               = SigmaJstJst_subj*Kt_subj*SigmaVVinv{cont2}*(1/sigmae2);
    %% Computation of all subjects/frequencies Generators calibration empirical covariance
    SJJ_subj             = zeros(q,q);
    for cont3 = 1:Nf
        SJJ_subj         = SJJ_subj + T_subj*Svv{cont2}(:,:,cont3)*T_subj';
    end
    SJJ_subj             = SJJ_subj/Nf;
    %% Scaling the Generators empirical covariance and data empirical covariance for individual subjects 
    scale_SJJ_subj       = (trace(SJJ_subj)/q)/max(diag(SigmaJstJst_subj));
    SJJ_subj             = SJJ_subj/scale_SJJ_subj;
    Svv{cont2}           = Svv{cont2}/scale_SJJ_subj;
    %% Computation of the Generators empirical covariance and posterior covariance common for all subjects
    SJJ_cal              = SJJ_cal + SJJ_subj;
    SigmaJstJst_cal      = SigmaJstJst_cal + SigmaJstJst_subj;
end
SJJ_cal                  = SJJ_cal/Nsubj;
SigmaJstJst_cal          = SigmaJstJst_cal/Nsubj;
%% initialization of the covariance matrix
SigmaJJ         = SJJ_cal + SigmaJstJst_cal;
end