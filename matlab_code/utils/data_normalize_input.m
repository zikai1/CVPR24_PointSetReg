function  [X,pre_normal] =data_normalize_input(x)

n=size(x,1);

% Direct manner is faster than the centering matrix manner
% tic;
pre_normal.xd=mean(x);

x=x-repmat(pre_normal.xd,n,1);
% toc;


% Centering matrix more slower than direct manner
% tic;
% centering_mat=eye(n)-1/n*ones(n);
% x1=centering_mat*x1;
% toc;

pre_normal.xscale=sqrt(sum(sum(x.^2,2))/n);

X=x/pre_normal.xscale;



