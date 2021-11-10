%% question 1
close all;
load F1_PVT.mat
F1=F1_PVT;
F1_matrix = [F1.Acc1P', F1.Black1P', F1.Car1P', F1.Flour1P', F1.Kitchen1P', F1.Steel1P';
             F1.Acc1V', F1.Black1V', F1.Car1V', F1.Flour1V', F1.Kitchen1V', F1.Steel1V';
             F1.Acc1T', F1.Black1T', F1.Car1T', F1.Flour1T', F1.Kitchen1T', F1.Steel1T';];
opts = statset('Display','final');
[idx,centroids] = kmeans(F1_matrix',6,'Distance','sqeuclidean',...
    'Replicates',5,'Options',opts);
%% a.
%rng default;

scatter3(F1_matrix(1,:),F1_matrix(2,:),F1_matrix(3,:))% P,V,T
%%
figure;
scatter3(F1_matrix(1,idx==1),F1_matrix(2,idx==1),F1_matrix(3,idx==1),'Magenta')
hold on
scatter3(F1_matrix(1,idx==2),F1_matrix(2,idx==2),F1_matrix(3,idx==2), 'Black' )
hold on
scatter3(F1_matrix(1,idx==3),F1_matrix(2,idx==3),F1_matrix(3,idx==3), 'Yellow')
hold on
scatter3(F1_matrix(1,idx==4),F1_matrix(2,idx==4),F1_matrix(3,idx==4), 'Blue')
hold on
scatter3(F1_matrix(1,idx==5),F1_matrix(2,idx==5),F1_matrix(3,idx==5), 'Green')
hold on
scatter3(F1_matrix(1,idx==6),F1_matrix(2,idx==6),F1_matrix(3,idx==6), 'Red')

% all the 6 centroids
scatter3(centroids(1,1),centroids(1,2),centroids(1,3),'filled', 'Magenta') 
hold on;
scatter3(centroids(2,1),centroids(2,2),centroids(2,3),'filled', 'Black') 
hold on;
scatter3(centroids(3,1),centroids(3,2),centroids(3,3),'filled', 'Yellow') 
hold on;
scatter3(centroids(4,1),centroids(4,2),centroids(4,3),'filled', 'Blue')
hold on;
scatter3(centroids(5,1),centroids(5,2),centroids(5,3),'filled', 'Green')
hold on;
scatter3(centroids(6,1),centroids(6,2),centroids(6,3),'filled', 'Red') 
hold on;
legend('Cluster 1','Cluster 2','Cluster 3','Cluster 4','Cluster 5','Cluster 6','Centroids',...
       'Location','NW')
title 'Cluster Assignments and Centroids'
hold off

%% b.
% clusters=real life object similarities? yeah kmeans separates them well
% means
%% c. change distance metric
[idx,centroids] = kmeans(F1_matrix',6,'Distance','cityblock',...
    'Replicates',5,'Options',opts);
close all;
figure;
scatter3(F1_matrix(1,idx==1),F1_matrix(2,idx==1),F1_matrix(3,idx==1))
hold on
scatter3(F1_matrix(1,idx==2),F1_matrix(2,idx==2),F1_matrix(3,idx==2))
hold on
scatter3(F1_matrix(1,idx==3),F1_matrix(2,idx==3),F1_matrix(3,idx==3))
hold on
scatter3(F1_matrix(1,idx==4),F1_matrix(2,idx==4),F1_matrix(3,idx==4))
hold on
scatter3(F1_matrix(1,idx==5),F1_matrix(2,idx==5),F1_matrix(3,idx==5))
hold on
scatter3(F1_matrix(1,idx==6),F1_matrix(2,idx==6),F1_matrix(3,idx==6))

% all the 6 centroids
scatter3(centroids(1,1),centroids(1,2),centroids(1,3),'g') 
hold on;
scatter3(centroids(2,1),centroids(2,2),centroids(2,3),'g') 
hold on;
scatter3(centroids(3,1),centroids(3,2),centroids(3,3),'g') 
hold on;
scatter3(centroids(4,1),centroids(4,2),centroids(4,3),'g') 
hold on;
scatter3(centroids(5,1),centroids(5,2),centroids(5,3),'g') 
hold on;
scatter3(centroids(6,1),centroids(6,2),centroids(6,3),'g') 
hold on;
legend('Cluster 1','Cluster 2','Cluster 3','Cluster 4','Cluster 5','Cluster 6','Centroids',...
       'Location','NW')
title 'Cluster Assignments and Centroids'
hold off

% the outcome doesn't change much cuz the dataset is already clear
% but distance values are much higher with sqeuclidean
%% question 2) bagging
%% a) apply bagging
load F1_Elec.mat
% split data 6:4
% train with 6 first trials
Tbl=[F1_Electrodes.Acc1Elec(:,1:6)';F1_Electrodes.Black1Elec(:,1:6)';F1_Electrodes.Car1Elec(:,1:6)';F1_Electrodes.Flour1Elec(:,1:6)';...
    F1_Electrodes.Kitchen1Elec(:,1:6)';F1_Electrodes.Steel1Elec(:,1:6)';];
disp(size(Tbl));
% give labels
C=cell(36, 1);
C(1:6,:)={'Acc'};
C(7:12,:)={'Black'};
C(13:18,:)={'Car'};
C(19:24,:)={'Flour'};
C(25:30,:)={'Kitchen'};
C(31:36,:)={'Steel'};
disp(size(C));
rng(0,'twister');%seed
Mdl=TreeBagger(24,Tbl,C,'OOBPrediction','On',...
    'Method','classification')  
%{
    why did i choose this number of trees?
%}
%% b) visualise
view(Mdl.Trees{1},'Mode','graph');
view(Mdl.Trees{2},'Mode','graph');
%% c) predict with last 4 trials
Tbl_test=[F1_Electrodes.Acc1Elec(:,6:10)';F1_Electrodes.Black1Elec(:,6:10)';F1_Electrodes.Car1Elec(:,6:10)';F1_Electrodes.Flour1Elec(:,6:10)';...
    F1_Electrodes.Kitchen1Elec(:,6:10)';F1_Electrodes.Steel1Elec(:,6:10)';];
Yfit=predict(Mdl,Tbl_test);
disp(size(Yfit));
% confusion matrix
group = C(1:30);
disp(size(group));
[confusion,order] = confusionmat(group,Yfit,'Order',{'Acc','Black','Car','Flour','Kitchen','Steel'})
%{
comment on overall accuracy:
poor accuracy, diagonal has low numbers meaning there are many
misclassifications
for ex, for Steel(last label), no datapoint was correctly classified.
%}
%% d)
% How can misclassifications in your results be explained given 
% the object properties? Do you think the PCA step was helpful?

