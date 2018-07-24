function [indK] = screening(Js,dimK,vertices,faces,indana)
% screening estimates the most active generators over the cortical surface 
% and create a mask to reduce the dimension of the analysis
%
% Inputs:
%    Js       = cortical activity (inverse solution)
%    dimK     = size of the anatomical mask
%    vertices = Nx3 matrix where each row is the coordinates of vertices of the surface mesh
%    faces    = Mx3 matrix where each row contains the indices of vertices conforming each face or triangle of the surface mesh
%    indana   = anatomical indices under analysis
%
% Outputs:
%    indK     = indices of the estimated anatomical mask
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

%% Initial values
Js = Js/max(Js);
thresh_mask = 0.9;
indices_vert = 0;
%% Estimating initial anatomial mask based on most active dipoles
while (length(indices_vert) < dimK) && (thresh_mask > 0)
    indices_vert = find(Js > thresh_mask);
    indices_vert = intersect(indices_vert,indana);
    thresh_mask = thresh_mask-(1e-3);
end
%% Postprocessing the anatomical mask looking for smoothness
if isempty(indices_vert)
    indK = 0;
else    
    indK = indices_vert;
    count_m = 2;
    count   = 1;
    act_th  = 6;
    d0      = 20;
    while count < count_m
        mask       = zeros(length(vertices),1);
        mask(indK) = 1;
        mask_new   = mask;
        for ii = 1:length(vertices)
            if mask(ii) == 0
                index = ii;
                [row,col] = find(faces==ii);
                findex_old = faces(row,:);
                findex_old = findex_old(:);
                findex_old = setdiff(findex_old,ii);
                vect = vertices(findex_old,:) - repmat(vertices(ii,:),[size(findex_old),1]);
                dold = sum(abs(vect).^2,2).^(1/2);
                findex_old(dold>d0) = [];
                index = [index;findex_old];
                act_near = sum(mask(index));
                if act_near > 2
                    mask_new(ii) = 1;
                end
            end
        end
        count = count+1;
        indK  = find(mask_new > 0);
    end
    count_m = 2;
    count   = 1;
    while count < count_m
        mask       = zeros(length(vertices),1);
        mask(indK) = 1;
        mask_new   = mask;
        for ii = 1:length(vertices)
            if mask(ii) == 1
                index = ii;
                [row,col] = find(faces==ii);
                findex_old = faces(row,:);
                findex_old = findex_old(:);
                findex_old = setdiff(findex_old,ii);
                vect = vertices(findex_old,:) - repmat(vertices(ii,:),[size(findex_old),1]);
                dold = sum(abs(vect).^2,2).^(1/2);
                findex_old(dold>d0) = [];
                index = [index;findex_old];
                act_near = sum(mask(index));
                if act_near <= 3
                    mask_new(ii) = 0;
                end
            end
        end
        count = count+1;
        indK  = find(mask_new > 0);
    end
end

end