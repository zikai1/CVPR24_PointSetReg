function d=sqdist2(aa,aa_col,bb_col,a,b)
% SQDIST - computes squared Euclidean distance matrix
%          computes a rectangular matrix of pairwise distances
% between points in A (given in columns) and points in B

% NB: very fast implementation taken from Roland Bunschoten

bb = sum(b.*b,1); ab = a'*b; 
d = abs(repmat(aa',[1 bb_col]) + repmat(bb,[aa_col 1]) - 2*ab);

