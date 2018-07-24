function [W_est,X_est] = bcvareta_l1_norm_penalty_gssbl(Sigma_complex,n)
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
Sigma_C = Sigma_complex;
q = length(Sigma_C);
Sth  = real(svds(Sigma_C,1));
if Sth < 0
    Sigma_C = Sigma_C - Sth*eye(q);
    Sth     = Sth - Sth;
    Sth(abs(Sth) < 0) = 0;
end
Sigma_C = Sigma_C +(1e-12)*Sth*eye(q);
maxIter   = 30;
Rho       = 1e-1;   
Rho_diag  = 1e0;
Rho_ndiag = 9e-1;
Lambda    = Rho_diag*eye(q)+Rho_ndiag*(ones(q)-eye(q));
%% Graph LASSO
[X_est,W_est] = bcvareta_graphssbl(Sigma_C,n,Rho*Lambda, maxIter);
end
