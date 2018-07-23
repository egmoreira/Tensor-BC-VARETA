function [llh] = bcvareta_likelihood_est(Ss_short,Us_short,Vs_short,sigmae2,Svv,T,A,m,n,asigmae2)
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

%% Compute Likelihood
llh1 = real(-(1/2)*trace((n*T*Svv*T')*Vs_short*diag(1./Ss_short)*Us_short')-(m/2)*sum(log(Ss_short)));
llh2 = real(-(m*n)/2*log(sigmae2)-(1/(2*sigmae2))*trace(n*A*Svv*A')-asigmae2*m*n/sigmae2);
llh = llh1 + llh2;
end