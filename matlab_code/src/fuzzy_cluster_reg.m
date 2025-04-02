function [alpha,T]=fuzzy_cluster_reg(src_pt, tgt_pt)
%==========================
% DESCRIPTION: Use fuzzy clustering framework for non-rigid point set registration
% Idea: take the source and target point sets as the clustering centroids
% and clustering members, respectively.
%
% INPUT
%--------
% src_pt   MxD  source point set with N samples D dimensions
% tgt_pt   NxD  target point set with C samples D dimensions
%
% OUTPUT
%--------
% alpha    scalar the clustering size 
% T        MxD the deformed source point set with N samples D dimensions
%==========================

[Nc,~]=size(src_pt); 
[Np,D]=size(tgt_pt); 
NpD=gpuArray(Np*D);


% Merge centroids between the both point sets first
src_pt_center=mean(src_pt,1);
tgt_pt_center=mean(tgt_pt,1);
center_dist=tgt_pt_center-src_pt_center;
src_pt=src_pt+center_dist;


% Initialize the variance sigma^2
sigma2=(Np*trace(src_pt*src_pt')+Nc*trace(tgt_pt*tgt_pt')-2*sum(src_pt)*sum(tgt_pt)')/(D*Nc*Np); % note trace(X'X)=trace(XX')

% Compute the Gramm matrix and low-rank decomposision by the improved Nystrom
theta=0.5;

% Low-rank matrix approximation of the Laplacian kernel
kernel = struct('type', 'rbf', 'para', theta); 
tic
% Number of landmark points, the larger the slower but more accurate
m=ceil(0.3*Nc);
Q = INys(kernel,src_pt, m, 'k');
c=size(Q,2);
elapsedTime = toc;
fprintf('Elapsed time is %.2f seconds.\n', elapsedTime);

% Parameter initialization 
W=zeros(Nc,D);
W=gpuArray((W));

iter=0;
tol=1e-5;
ntol=tol+10;
maxNumIter=50;
Loss=1;

T=src_pt; % deformed
F=tgt_pt; % fixed 
FT=F';
FF=sum(FT.*FT,1);

% Entropy regularization (seems that the lower the better)
beta=0.5;% 0.5 in default 

% Displacement field regularization (seems that the higher the better)
lambda=0.1;% 0.1 in default 


alpha=ones(1,Nc);

alpha=gpuArray((alpha));
onesUy=ones(Nc,1);
onesUy=gpuArray((onesUy));
onesUx=ones(Np,1);
onesUx=gpuArray((onesUx));
IdentMatrix=eye(c);
IdentMatrix=gpuArray((IdentMatrix));

viz=1;
NEAR_0=1e-10; % avoid logx->-inf and NaN 

%figure;
tic;
while (ntol>tol)&&(iter<maxNumIter)&&(sigma2>1e-8)
    
    Loss_old=Loss;
    
    QtW=Q'*W;
    
    %Acceleration by pre-computing FF,Np,Nc,FT    
    fuzzy_dist=exp(-sqdist2(FF,Np,Nc,FT,T')/(sigma2*beta)).*alpha;% lower beta generally leads to faster convergence and better effect. 
    disp(fuzzy_dist(1,1));
    sum_fuzzy_dist=1./(sum(fuzzy_dist,2));% sum over the same centroid
    U=fuzzy_dist.*sum_fuzzy_dist; % sum(U,2)=1,construct fuzzy partition matrix
    
    % Add KL divergence 
    U=U+NEAR_0;
    logU=log(U);
    
    alpha=sum(U,1)/Np;
    
    alpha=alpha+NEAR_0;
    log_alpha=log(alpha);
    
    U1=U*onesUy;
    
    Ut1=U'*onesUx;
    
    dU=sparse(diag(U1));
    
    dUt=sparse(diag(Ut1));
     
    dUtQ=dUt*Q;
    
    Uttgt=U'*tgt_pt;
    
    P=Uttgt-dUt*src_pt;
    
    A=lambda*sigma2*IdentMatrix+Q'*dUtQ;
    
    W=1/(lambda*sigma2)*(P-dUtQ*(A\Q'*P));


    wdist_pt2center=abs(trace(FT*dU*F+T'*dUt*T-2*Uttgt'*T));
    
    % Entropy of U
    H_U=sum(sum(U.*logU)); 
    % Entropy of alpha
    H_alpha=Np*alpha*log_alpha';
    
    KL_U_alpha=H_U-H_alpha;
    asd1 = lambda/2*trace(QtW'*QtW);
    %-----Acceleration by pre-computing NpD=Np*D
    Loss=(1/sigma2)*wdist_pt2center+NpD*log(sigma2)+lambda/2*trace(QtW'*QtW)+beta*(KL_U_alpha);
    ntol=abs((Loss-Loss_old)/Loss);
    
    
    % Update new clustering positions
    T=src_pt+Q*(Q'*W);

    % Update sigma^2
    sigma2=wdist_pt2center/NpD;
    
    
    % Visualize 
    % if viz
    %     cpd_plot_iter(tgt_pt,T);
    % end
    
    % disp(["iter:" iter,"KL_U_alpha" gather(KL_U_alpha) ,"wdist_pt2center:" gather(wdist_pt2center),"sigma2:" gather(sigma2), "tolerance:" gather(ntol)]);



    iter=iter+1;
    

  
end
elapsedTime = toc;
fprintf('time: %.5f s\n', elapsedTime);



