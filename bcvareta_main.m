function [SigmaJJ_graph,XJJ_graph,SigmaJJ,llh] = bcvareta_main(Svv,K,Nsamp)
% BC-VARETA  computes the precision matrix and inverse solution as
% population/frequency common features, indistinctively using EEG or
% MEG stationary time series, by the populational EM algorithm with
% Sources Hermitian Graphical LASSO
%
% inputs:
%    Svv           : cell array containing all subjects EEG/MEG time series
%                    crossspectra
%    K             : cell array containing all subjects EEG/MEG Lead Fields
%    Nsamp         : mean of the sample number of all EEG/MEG time series cross spectra
%
% outputs:
%    SigmaJJ_graph : population/frequency featured covariance matrix after 
%    XJJ_graph     : population/frequency featured precision matrix
%    SigmaJJ       : population/frequency featured covariance matrix before 
%
%
% BC-VARETA toolbox is based on the following publication: Gonzalez-Moreira, E., 
% Paz-Linares, D., Martinez-Montes, E., Valdes-Hernandez, P., Bosch-Bayard,
% J., Bringas-Vega, ML., Valdes-Sosa, P., (2018), "Populational Super-Resolution
% Sparse M/EEG Sources and Connectivity Estimation", bioRxiv, 346569.
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

%% step 1 Initialization populational EM algorithm
[sigmae2,asigmae2,Iq,p,Ip,SigmaVV,SigmaVVinv,maxiter,miniter,Nsubj,Nf,Kt,SigmaJJ,K,Svv] = bcvareta_initial_values(Svv,K);
llh  = zeros(maxiter,1);
%% Outer loop of populational EM algorithm
for cont1 = 1:maxiter
    SigmaJJ_upd = zeros(size(Iq));
    sigmae2_upd = 0;
    for cont2 = 1:Nsubj
        %% step 2 based on equation [2.1.9],Appendix D equation [D1.1]
        SigmaKt     = SigmaJJ*Kt{cont2};
        %         SigmaJstJst = inv(Kt{cont2}*SigmaVVinv{cont2}*K{cont2}/sigmae2 + inv(SigmaJJ));
        SigmaJstJst = SigmaJJ-SigmaKt/(K{cont2}*SigmaKt+sigmae2*SigmaVV{cont2})*K{cont2}*SigmaJJ;
        %% step 3 based on equation [2.1.10], Appendix D equation [D1.2]
        T{cont2}    = SigmaJstJst*Kt{cont2}*SigmaVVinv{cont2}*(1/sigmae2);
        Tt{cont2}   = T{cont2}';
        %% step 4 based on equation [2.1.15], Appendix B equation [B3.5]
        A{cont2}    = (Ip{cont2}-K{cont2}*T{cont2});
        for cont3 = 1:Nf
            %% step 5 based on equation [2.1.12], Appendix D equation [D1.4] and [D1.6]
            SJstJst      = T{cont2}*Svv{cont2}(:,:,cont3)*Tt{cont2};
            SigmaJJ_plus = SigmaJstJst + SJstJst;
            SigmaJJ_upd  = SigmaJJ_upd + SigmaJJ_plus;
            %% step 6 based on equation [2.1.14], Appendix B, equation [B3.6]
            sigmae2_plus = trace(SigmaVVinv{cont2}*A{cont2}*Svv{cont2}(:,:,cont3)*A{cont2}')/p{cont2}+trace(Kt{cont2}*SigmaVVinv{cont2}*K{cont2}*SigmaJstJst)/p{cont2}+asigmae2;
            sigmae2_upd  = sigmae2_upd + sigmae2_plus;
        end %frequencies
    end %subjects
    SigmaJJ  = SigmaJJ_upd/(Nsubj*Nf);
    sigmae2  = sigmae2_upd/(Nsubj*Nf);
    %% Computing likelihood inputs
    [U,S,V]  = svd(real(SigmaJJ));
    index    = (diag(S) > (1e-19)*max(diag(S)));
    q_short  = sum(index);
    S_short  = diag(S(1:q_short,1:q_short));
    U_short  = U(:,1:q_short);
    V_short  = V(:,1:q_short);
    %% step 7 estimation of likelihood
    lf_upd      = 1;
    for cont2 = 1:Nsubj
        for cont3 = 1:Nf
            [lf_plus] = bcvareta_likelihood_est(S_short,U_short,V_short,sigmae2,Svv{cont2}(:,:,cont3),T{cont2},A{cont2},Nsamp,p{cont2},asigmae2);
            lf_upd    = lf_upd + lf_plus;
        end
    end
    lf          = lf_upd/(Nsubj*Nf);
    %% Convergence criteria
    llh(cont1) = sign(lf)*log10(abs(lf));
    if (cont1 > miniter)
        if (abs(llh(cont1)-llh(cont1-miniter)) < 1e-2)
            llh(cont1+1:end) = [];
            break;
        end
    end
end %iterations outer loop
%% step 8 based on graphical model
[SigmaJJ_graph,XJJ_graph] = bcvareta_l1_norm_penalty_gssbl(SigmaJJ_upd,Nsamp);
end