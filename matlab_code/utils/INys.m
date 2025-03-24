% INys.m implments the improved nystrom low-rank approximation method in
% <Improved Nystrom low-rank Approximation and Error Analysis> by Zhang and
% Kwok, ICML 2008

%Input:
% data: n-by-dim data matrix;
% m: number of landmark points;
% kernel: (struct) kernel type and parameter
% s: 'r' for random sampling and 'k' for k-means based sampling

%Output:
% Ktilde: approximation of kernel matrix K, in the form of GG'

function G = INys(kernel,data, m, s)


[n,~] = size(data);

if(s == 'k')
    [~, center, m] = eff_kmeans(data, m, 5); %#iteration is restricted to 5
    % [~, center, m] = elkan_kmeans(data, m, 5); %#iteration is restricted to 5
end

if(s == 'r')
   dex = randperm(n);
   center = data(dex(1:m),:);
end

if(kernel.type == 'pol')
    W = center * center';
    E = data * center';
    W = W.^kernel.para;
    E = E.^kernel.para;
end

if (kernel.type == 'rbf')
    % Gaussian kernel
%      W = exp(-sqdist(center', center')/kernel.para);
%      E = exp(-sqdist(data', center')/kernel.para);
    % Laplacian kernel (L2)
%     W = exp(-sqrt(sqdist(center', center'))/kernel.para);
%     E = exp(-sqrt(sqdist(data', center'))/kernel.para);
    % Laplacian kernel (L1)
    W = exp(-(pdist2(center, center,'cityblock'))/kernel.para);
    E = exp(-(pdist2(data, center,'cityblock'))/kernel.para);
end
[Ve, Va] = eig(W);
va = diag(Va);
pidx = find(va > 1e-6);
inVa = sparse(diag(va(pidx).^(-0.5)));
G = E * Ve(:,pidx) * inVa;
