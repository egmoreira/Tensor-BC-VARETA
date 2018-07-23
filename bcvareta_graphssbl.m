function [X, W] = bcvareta_graphssbl(S, n, Lambda, maxIter)
%%  1_Problem statement:
%  Minimizing f1(X) = -log|X|+trace(S*X)+norm(Lambda.*X,1)
%  It comes from the the maximization of the log posterior of X made up the complex data Likelihood (|X|/pi^q)^n*exp(-n*trace(S*X)) and the complex Laplace prior
%  prod(n*Lambda/2)*exp(-n*norm(Lambda.*X,1)). It is convenient to reorgamize this prior as n^(q/2)*prod(n^(1/2)*Lambda/2)*exp(-n^(1/2)*norm(Lambda.*(n^(1/2)*X),1)).
%  X       - is the complex hermitic Inverse Covariance matrix, 
%  S       - is the complex hermitic Empirical Covariance matrix (normalized), 
%  n       - is the sample size,
%  q       - is the problem dimensions, 
%  Lambda  - is a nonnegative regularization parameters' matrix. 
%  maxIter - maximum number of iterations of the Newton-cycle. 
%%  2_Gaussian Scaled Mixtures prior scheme
%  a) The Laplace prior factor is substitued by a Gaussian prior prod((2pi*Gamma).^(-1/2).*exp(-(n/2)*abs(X).^2./Gamma)) and scaled by
%     a gamma distribution prod((n*Lambda.^2/2).*exp(-(n*Lambda.^2/2).*Gamma)).
%  b) The Inverse Covariance modified target function with Gaussian prior has the structure: f2(X|Gamma) = -log|X|+trace(S*X)+(1/2)*norm(X./Gamma.^(1/2),2)^2   
%  c) To obtain the estimator of X we maximize the joint posterior distribution of (X,Gamma) by getting te iterated estimator of (X|Gamma) = argmin{f2(X|Gamma)}
%  d) After updating (X|Gamma) we maximize the Gaussian Mixture priors' logarithm f3(Gamma) = log(prod(Gamma.^(-1/2)))-(n/2)*sum(abs(X).^2./Gamma)-(n/2)*sum(Lambda.^2.*Gamma), 
%     and back to the Inverse covariance updating.
%% 3_Consistent Satistical Standarization of the Likelihood+prior optimization problem: 
%  a) We redefine an standard Inverse Covariance matrix as Xst = X./(Gamma).^(1/2). The standard Inverse Covariance Xst keeps 
%     the Positive Definiteness propertie if Gamma is also a Positive Definite matrix, due to Schur Theorem.
%  b) Due the fact that S -> X^(-1) as n^(1/2) -> Inf the Likelihood can be redefined as a function of an standard Empirical Covariance matrix Sst^(-1) = S^(-1)./Gamma.^(1/2) 
%     and the standard Inverse Covariance matrix. Sst is a consistent estimator of the Empirical Covariance matrix of a standarized likelihood with Inverse Covariance Xst, Sst -> Xst^(-1) as n -> Inf.
%  c) The standard target function is f2st(Xst) = -log|Xst|+trace(Sst*Xst)+(1/2)*norm(Xst,2)^2, positive definiteness of the standard Empirical Covariance 
%     depends on the solution for Gamma, deriving the f3 over Gamma we get: -1./Gamma + n*abs(X).^2./Gamma.^2 - n*Lambda.^2 = 0, the solution is:  
%     Gamma1 = (sqrt(1 + 4*n^2*Lambda.^2.*abs(X).^2 ) - 1)./(2*n*Lambda.^2), this estimator presernves the Positive Definiteness property!!! Elementwise Anlitic Functions do it.
%% 4_Direct solution of the Standarized problem
%  a) By differentiation of f2st we get the following matrix equation -Xst^(-1) + Sst + Xst = 0, multiplying by Xst we obtain Xst^2 + Sst*Xst - I = 0, 
%     an special case of Riccati matrix equation.
%  b) The equation has a unique positive definite solution that commutes with Sst: Xst1 = (1/2)*sqrt(Sst^2 + 4*I) - (1/2)*Sst.  
%  c) If [U,S,V] = svd(Sst) then Xst1 = (1/2)*U*diag(sqrt(diag(D).^2 + 4) - diag(D))*V' is an ultra fast solution for the Inverse Covariance estimator.
%  d) After solving the standard problem we unstandarize it X1 = (Gamma).^(1/2).*Xst1
%
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

%% Check Positive Semidefinitness of Sample Covariance and Lambda 
dminS = min(eig(S));
if dminS <=0
    disp('Error1: Sample Covariance (S) is not Positive Definite')
end
%% Check Positive Semidefinitness of Lambda 
dminLambda = min(eig(Lambda));
if dminLambda < 0
    disp('Error2: Regularization Parameters Mask (Lambda) is not Positive Semidefinite')
end
%% Initialization
q       = length(S);
Iq      = eye(q);
Lambda2 = Lambda.^2;
n2      = n^2;
n12     = n^(1/2);
idx     = (Lambda > 0);
idx0    = (Lambda == 0);
D       = real(eig(S));
Sinv    = inv(S);
Sinv_ph = exp(1i*angle(Sinv));
Gamma   = zeros(q);
Sst_inv = zeros(q);
epsilon = 2.5e4;
%% Calibration
X     = Sinv;
%% Main cycle
for i = 0:maxIter
    %% Estimation of variances Gamma of Gaussian Mixtures prior
    DET         = 1 + 4*n2*Lambda2(idx).*abs(X(idx)).^2;
    Gamma(idx)  = (sqrt(DET) - 1)./(2*n*Lambda2(idx));
    Gamma(idx0) = n*abs(X(idx0)).^2;
    %% Standarization of the Empirical Covariance by Consistency and Extreme Values conditions
    %(Lambda > 0) 
    Ninf          = max(Gamma(idx));
    NGamma        = Gamma(idx)/Ninf;
    st_factor1          = Ninf^(-1/2)*Sinv(idx).*NGamma.^(1/2)*(epsilon*n12) + (1/(epsilon*n12))*Sinv_ph(idx);
    st_factor2          = (1 + NGamma*(epsilon*n12));
    Sst_inv(idx)  = st_factor1./st_factor2;
    Sst_inv(idx0) = (1/n12)*Sinv_ph(idx0);
    %% Estimation of the standard Inverse Covariance by Second Order Matrix Equation
    [U,D,V]       = svd(Sst_inv);
    D             = abs(diag(D));
    Dinv          = 1./D;
    Xst           = (1/2)*V*diag(sqrt(Dinv.^2 + 4) - Dinv)*U';
    Xst           = (Xst + Xst')/2;
    %% Unstandarized Inverse Covariance
    X             = Gamma.^(1/2).*Xst;
end
W = Iq/X;
end