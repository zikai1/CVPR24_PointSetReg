function [idx, center, m] = eff_kmeans(data, m, MaxIter);

[n, dim] = size(data);
dex = randperm(n);
center = data(dex(1:m),:);

for i = 1:MaxIter;
    nul = zeros(m,1);
    [xx, idx] = min(sqdist(center', data'));
    for j = 1:m;
        dex = find(idx == j);
        l = length(dex);
        cltr = data(dex,:);
        if l > 1;
            center(j,:) = mean(cltr);
        elseif l == 1;
            center(j,:) = cltr;
        else
            nul(j) = 1;
        end;
    end;
    dex = find(nul == 0);
    m = length(dex);
    center = center(dex,:);
end;

