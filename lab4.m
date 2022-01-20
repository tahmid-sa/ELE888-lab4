clc
clear all
close all

%% Upload the house.tiff image

I = imread('house.tiff');
X = reshape(I, 256*256, 3);
X = double(X);

figure, plot3(X(:,1), X(:,2), X(:,3),'.','Color',[1, 0, 0])

%% A. Let c =  2,  and  run  the  k-means  algorithm  using  your  own  initial  state  for  the  two means. 
% Keep a record of the initial means you use.

mean = [.1 .1 .1
     .9 .9 .9];

mean = mean*256;

initial_mean = zeros(size(mean));

J = [];
mean1_total = [mean(1,:)];
mean2_total = [mean(2,:)];

while (initial_mean ~= mean)

    initial_mean = mean;
    
    J1 = (X - repmat(mean(1,:), size(X,1), 1));
    J1 = sum(J1.^2, 2);

    J2 = (X - repmat(mean(2,:), size(X,1), 1));
    J2 = sum(J2.^2, 2);

    cluster1 = J1 < J2;
    cluster2 = ~cluster1;

    mean(1,:) = sum(X(cluster1, :)) / sum(cluster1);
    mean(2,:) = sum(X(cluster2, :)) / sum(cluster2);
    
    J = [J sum(min(J1, J2))];
    mean1_total = [mean1_total; mean(1,:)];
    mean2_total = [mean2_total; mean(2,:)];
    
    mean
end

% i: A plot of the error criterion, J.
close all
plot(J)
title("A.i Plot of the error criterion, J")

% ii: A plot of the cluster means for at least two stages of the clustering
% process, andtheir  nal values.
figure
plot3(mean1_total(:, 1), mean1_total(:, 2), mean1_total(:, 3), '-*')
hold all
plot3(mean2_total(:, 1), mean2_total(:, 2), mean2_total(:, 3), '-*')
title("A.ii Plot of the cluster means")

% iii: A plot of the labeled data samples (pixels) in RGB space.
figure
X1 = X(cluster1, :);
X2 = X(cluster2, :);

plot3(X1(:,1), X1(:,2), X1(:,3),'.','Color', mean(1,:)/256)
hold all
title("A.iii Plot of cluster 1 labeled data samples in RGB space")

plot3(X2(:,1), X2(:,2), X2(:,3),'.','Color', mean(2,:)/256)
title("A.iii Plot of cluster 2 labeled data samples in RGB space")

% iv: Plot of the image in labeled form.
figure
XX = repmat(mean(1,:), size(X,1), 1) .* repmat(cluster1, 1, size(X,2));
XX = XX + repmat(mean(2,:), size(X,1), 1) .* repmat(cluster2, 1, size(X,2));
XX = reshape(XX, size(I, 1), size(I, 2), 3);

subplot(1,2,1);
imshow(I)
subplot(1,2,2);
imshow(XX/256)

%% B. Let c = 5, now perform two independant runs of the k-means algorithm.

c = 5;

initial_mean_1 = [
  173.8240   60.4672  169.4464
  162.7648   30.5664  197.1968
  241.9712  155.4688   89.6512
   53.4784  115.2256  169.4720
  181.5808  117.4272  106.5472
];

XX=[];

mean = initial_mean_1;
initial_mean = zeros(size(mean));

while (initial_mean ~= mean)

    initial_mean = mean;
    
    J = zeros(size(X,1), c);
    
    for i = [1:c]
        j = (X - repmat(mean(i,:), size(X,1), 1));
        j = sum(j.^2, 2);
        J(:,i) = j;
    end
    [a, cluster] = min(J, [], 2);
    
    for i = [1:c]
        current_cluster = (cluster==i);
        mean(i, :) = sum(X(current_cluster, :)) / sum(current_cluster);

    end
XX = zeros(size(X));

for i = [1:c]
    current_cluster = (cluster==i);
    XX = XX + repmat(mean(i,:), size(X,1), 1) .* repmat(current_cluster, 1, size(X,2));
end
XX = reshape(XX, size(I, 1), size(I, 2), 3);
subplot(1,2,1);
imshow(I)
subplot(1,2,2);
imshow(XX/256)
end

mean1 = mean
cluster1 = cluster;

figure
for i = [1:c]
    current_cluster = (cluster==i);
    Xi = X(current_cluster, :);
    plot3(Xi(:,1), Xi(:,2), Xi(:,3),'.','Color', mean(i,:)/256)
    title("B. Plot of image with initial mean 1")
    hold all
end

%%
initial_mean_2 = [
  215.5339  138.4293  240.5963
  213.2267  222.7049  165.2613
   65.6489   67.7834  122.7426
  157.0459   81.4270  163.6651
  149.0558   30.5189  139.4473
];

mean = initial_mean_2;

initial_mean = zeros(size(mean));

while (initial_mean ~= mean)

    initial_mean = mean;
    
    J = zeros(size(X,1), c);
    
    for i = [1:c]
        j = (X - repmat(mean(i,:), size(X,1), 1));
        j = sum(j.^2, 2);
        J(:,i) = j;
    end
    [a, cluster] = min(J, [], 2);
    
    for i = [1:c]
        current_cluster = (cluster==i);
        mean(i, :) = sum(X(current_cluster, :)) / sum(current_cluster);
    end
    
end
mean2 = mean
cluster2 = cluster;

%
figure
for i = [1:c]
    current_cluster = (cluster==i);
    Xi = X(current_cluster, :);
    plot3(Xi(:,1), Xi(:,2), Xi(:,3),'.','Color', mean(i,:)/256)
    title("B. Plot of image with initial mean 2")
    hold all
end

%% C. Use the XB index to asses the quality of the k-means algorithm.

N = size(X,1);
XB1 = 0;

for i = [1:c]
    current_cluster = (cluster1==i);
    Xi = X(current_cluster, :);
    muu_ij = sort(sum((mean1 - repmat(mean1(i,:), c, 1)).^2, 2).^.5);
    XB1 = XB1 + sum(sum((Xi - repmat(mean1(i,:), size(Xi,1), 1)).^2, 2).^.5) / muu_ij(2);
end

XB1 = XB1 / N

%%
XB2 = 0;

for i = [1:c]
    current_cluster = (cluster2==i);
    Xi = X(current_cluster, :);
    muu_ij = sort(sum((mean2 - repmat(mean2(i,:), c, 1)).^2, 2).^.5);
    XB2 = XB2 + sum(sum((Xi - repmat(mean2(i,:), size(Xi,1), 1)).^2, 2).^.5) / muu_ij(2);
end

XB2 = XB2 / N