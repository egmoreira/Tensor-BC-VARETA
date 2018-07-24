function [miu,alpha1,k,h] = cross_nonovgrouped_enet_ssbl(Svv,LeadFields,Nsamp,nonovgroups)
% cross_nonovgrouped_enet_ssbl estimates the inverse solution based on elastic net_sparse bayesian learning 
% 
% Inputs:
%    Svv         = cell array containing all subjects M/EEG time series crossspectra
%    LeadFields  = cell array containing all subjects M/EEG Lead Fields
%    Nsamp       = mean of the sample number of all M/EEG time series cross spectra
%    nonovgroups = spatial groups
%    vertices    = Nx3 matrix where each row is the coordinates of vertices of the surface mesh
%    faces       = Mx3 matrix where each row contains the indices of vertices conforming each face or triangle of the surface mesh

% Outputs:
%    miu         = inverse solution
%    alpha1      = hyperparameter of the L2 norm
%    k           = hyperparameter of the truncate Gamma pdf
%    h           = hypeparameter for Gamma
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
%% Loading Dimensions
Nv         = size(LeadFields{1},2);                                       % Number of nodes
Ngroups    = length(nonovgroups);                                         % Number of spatial groups
Nt         = size(Svv{1},3);                                              % Number of Time points or Frequency bins
Nsubj      = size(Svv,2);                                                 % Number of Subjects
sigma      = ones(Nv,1);                                                  % Prior Variances
sigma_post = ones(Nv,Nsubj);                                              % Posterior Variances
miu_cal    = ones(Nv,Nt,Nsubj);                                           % Inverse Solution Calibration
miu        = ones(Nv,Nt,Nsubj);                                           % Inverse Solution
h          = ones(Nv,1);                                                  % Hypeparameter for Gamma
%% User defined parameters
maxiter1   = 20;                                                          % Number of Iterations of outer cycle
maxiter11  = 3;                                                           % Number of Iterations of inner cycle
e1         = 1E-5;                                                        % Rate parameter of the gamma pdf of alpha (higher -> smoother)
e2         = 1E-6;                                                        % Rate parameter of the gamma pdf of k (higher -> smoother)
beta       = 1;                                                           % Data variance
%% Initialization of parameters
for cont4 = 1:Nsubj
    K_tmp            = LeadFields{cont4};
    Kt_tmp           = transpose(K_tmp);
    Ne_tmp           = size(K_tmp,1);
    Ine_tmp          = spdiags(ones(Ne_tmp,1),0,Ne_tmp,Ne_tmp);           % Identity matrix with Ne size
    scale_K_tmp      = sqrt(trace(K_tmp*Kt_tmp)/Ne_tmp);                  % Lead Field Scale Norm inf
    K_tmp            = K_tmp/scale_K_tmp;                                 % Lead Field scaling
    Kt_tmp           = Kt_tmp/scale_K_tmp;
    %% Calibration Inverse Solution
    sigmaKt          = spdiags(sigma,0,Nv,Nv)*Kt_tmp;
    sigma_post1      = sigmaKt/(K_tmp*sigmaKt+beta*Ine_tmp);
    sigma_post2      = K_tmp*spdiags(sigma,0,Nv,Nv);
    for jj=1:Nv
        sigma_post(jj,cont4) = sigma(jj)-sigma_post1(jj,:)*sigma_post2(:,jj);
    end
    %% Compute 'miu_cal' for all slices of 'V'
    TrOp             = (1/beta).*(sigmaKt - sigma_post1*(sigma_post2*Kt_tmp));
    for cont3 = 1:Nt
        miu_cal(:,cont3,cont4) = diag(TrOp*squeeze(Svv{cont4}(:,:,cont3))*TrOp');
    end
    Ne{cont4}        = Ne_tmp;
    Ine{cont4}       = Ine_tmp;
    K{cont4}         = K_tmp;
    Kt{cont4}        = Kt_tmp;
    KtK{cont4}       = Kt_tmp*K_tmp;
    scale_K{cont4}   = scale_K_tmp;
    scale_V{cont4}   = (sum(mean(reshape(abs(miu_cal(:,:,cont4)),Nv,Nt),2))/Nv)/max(sigma_post(:,cont4));
end
%% Data Scaling
for cont4 = 1:Nsubj
    Svv{cont4} = Svv{cont4}/scale_V{cont4};
end
%% Initialization of Hyperparameters
alpha1       = 1E0;                                                     % Hyperparameter of the L2 norm
k            = 1E0;                                                     % Hyperparameter of the Truncate Gamma pdf
%% Main Cycle
for cont1 = 1:maxiter1
    for cont11 = 1:maxiter11
        for cont4 = 1:Nsubj
            %% Update Posterior Mean and Covariance matrix
            sigmaKt     = spdiags(sigma,0,Nv,Nv)*Kt{cont4};
            sigma_post1 = sigmaKt/(K{cont4}*sigmaKt+beta*Ine{cont4});
            sigma_post2 = K{cont4}*spdiags(sigma,0,Nv,Nv);
            % Only save the diagonals of the Posterior Covariance
            for jj=1:Nv
                sigma_post(jj,cont4) = sigma(jj)-sigma_post1(jj,:)*sigma_post2(:,jj);
            end
            % Iterative Transference Operator
            TrOp        = (1/beta).*(sigmaKt-sigma_post1*(sigma_post2*Kt{cont4}));
            % Compute 'miu' for all slices of 'V'
            for cont3 = 1:Nt
                miu(:,cont3,cont4) = diag(TrOp*squeeze(Svv{cont4}(:,:,cont3))*TrOp');
            end
        end
        miu2            = sum(reshape(abs(miu),Nv,Nt*Nsubj),2)/(Nt*Nsubj);
        sigma_post_mean = sum(sigma_post,2)/Nsubj;
        %% Update Gammas
        for group = 1:Ngroups
            idx_group    = nonovgroups(group);
            idx_group    = idx_group{1};
            h(idx_group) = sqrt((1./4).^2+alpha1.*k.*Nt.*Nsamp.*Nsubj.*sum(miu2(idx_group) + sigma_post_mean(idx_group)))-1./4;
        end
        index_h        = find((miu2 + sigma_post_mean)<0);
        h(index_h)     = 0;
        gamma          = k + h;
        sigma_bar      = h./gamma;
        sigma          = (1/(2*alpha1))*sigma_bar;
    end
    %% Update alpha_1
    index_alpha_1  = find(sigma_bar>0);
    alpha1         = (length(index_alpha_1)/2 + Nv*Nt*Nsamp*Nsubj)/(Nt*Nsamp*Nsubj*sum((miu2(index_alpha_1) + sigma_post_mean(index_alpha_1))./(sigma_bar(index_alpha_1))) + e1*Nv*Nt*Nsubj*Nsamp);
    %% Update alpha_2
    f_aux          = @(k_aux) e2*Nt*Nsamp*Nsubj + sum(ones(Nv,1)./(1-sigma_bar))/Nv - (Nt*Nsamp*Nsubj - 1/2)/k_aux - trascendent_term(k_aux);
    k              = fzero(f_aux,[0.000001 700000]);
    alpha2         = (4*alpha1*k)^(1/2);
    sigma          = (1/(2*alpha1))*sigma_bar;
end
for cont4 = 1:Nsubj
    miu(:,:,cont4) = (miu(:,:,cont4)*scale_V{cont4}/scale_K{cont4});
end
end