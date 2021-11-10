%% Making all data in one matrix
clear all; close all; clc;
load('F1_PVT.mat')
F1 = F1_PVT;

F1_matrix = [F1.Acc1P', F1.Black1P', F1.Car1P', F1.Flour1P', F1.Kitchen1P', F1.Steel1P';
             F1.Acc1V', F1.Black1V', F1.Car1V', F1.Flour1V', F1.Kitchen1V', F1.Steel1V';
             F1.Acc1T', F1.Black1T', F1.Car1T', F1.Flour1T', F1.Kitchen1T', F1.Steel1T';];

% verify that data are of correct format
%scatter3(F1_matrix(1,:), F1_matrix(2,:), F1_matrix(3,:) )

mean_P = mean(F1_matrix(1, :));
mean_V = mean(F1_matrix(2, :));
mean_T = mean(F1_matrix(3, :));

F1_stand = [F1_matrix(1, :) - mean_P;
            F1_matrix(2, :) - mean_V;
            F1_matrix(3, :) - mean_T;];
        
% verify that data are standardised
%scatter3(F1_stand(1,:), F1_stand(2,:), F1_stand(3,:) )
%mean(F1_stand')


S = cov(F1_stand')

[eig_Vec, eig_Val] = eig(S)



%% Reduce dimentionality of data
% The Highest eigen value is in third column of eig_Val and second highest
% in second column

F_vec = [eig_Vec(:,3), eig_Vec(:,2)]


F1_reduced = F_vec'*F1_stand;

subplot(1,2,1)
PC1_x = [0, eig_Vec(1, 3)*100];
PC1_y = [0, eig_Vec(2, 3)*100];
PC1_z = [0, eig_Vec(3, 3)*100];
plot3(PC1_x, PC1_y, PC1_z, 'red', 'linewidth', 2)
hold on;
PC2_x = [0, eig_Vec(1, 2)*30];
PC2_y = [0, eig_Vec(2, 2)*30];
PC2_z = [0, eig_Vec(3, 2)*30];
plot3(PC2_x, PC2_y, PC2_z, 'green', 'linewidth', 2)
grid on;
scatter3(F1_stand(1,:), F1_stand(2,:), F1_stand(3,:), 'filled')
title('Standardised data with principal components')
xlabel('Pressure')
ylabel('Vibration')
zlabel('Temperature')



%scatter(F1_reduced(1,:), F1_reduced(2,:))
% Transfer original data to reduced dimentions and plot (with seperate colour for each data)
subplot(1,2,2)
Acc1_s = [F1.Acc1P'; F1.Acc1V'; F1.Acc1T'] - [mean_P; mean_V; mean_T]
Black1_s = [F1.Black1P'; F1.Black1V'; F1.Black1T'] - [mean_P; mean_V; mean_T]
Car1_s = [F1.Car1P'; F1.Car1V'; F1.Car1T'] - [mean_P; mean_V; mean_T]
Flour1_s = [F1.Flour1P'; F1.Flour1V'; F1.Flour1T'] - [mean_P; mean_V; mean_T]
Kitchen1_s = [F1.Kitchen1P'; F1.Kitchen1V'; F1.Kitchen1T'] - [mean_P; mean_V; mean_T]
Steel1_s = [F1.Steel1P'; F1.Steel1V'; F1.Steel1T'] - [mean_P; mean_V; mean_T]


Acc1_r = F_vec' * Acc1_s;
Black1_r = F_vec' * Black1_s;
Car1_r = F_vec' * Car1_s;
Flour1_r = F_vec' * Flour1_s;
Kitchen1_r = F_vec' * Kitchen1_s;
Steel1_r = F_vec' * Steel1_s;

%figure();
scatter(Acc1_r(1,:), Acc1_r(2,:), 'Magenta', 'filled')
title('Data reduced to 2D')
hold on;
scatter(Black1_r(1,:), Black1_r(2,:), 'Black', 'filled')
scatter(Car1_r(1,:), Car1_r(2,:), 'Yellow', 'filled')
scatter(Flour1_r(1,:), Flour1_r(2,:), 'Blue', 'filled')
scatter(Kitchen1_r(1,:), Kitchen1_r(2,:), 'Green', 'filled')
scatter(Steel1_r(1,:), Steel1_r(2,:), 'Red', 'filled')
grid on;
xlabel('Principal Component 1')
ylabel('Principal Component 2')


hold off;
%% Plot variance on each Principal component
F1_projected = eig_Vec'*F1_stand;

% Remember that the third principal component is the first column of
% eig_Vec
subplot(3,1,1)
scatter(F1_projected(3,:), ones(60,1), 'filled', 'red') % First principal component
ylim([0.5, 1.5])
xlim([-250, 250])
title('1st Principal Component')
set(gca,'yticklabel',[])
subplot(3,1,2)
scatter(F1_projected(2,:), ones(60,1), 'filled', 'green') % Second principal component
ylim([0.5, 1.5])
xlim([-250, 250])
title('2nd Principal Component')
set(gca,'yticklabel',[])
subplot(3,1,3)
scatter(F1_projected(1,:), ones(60,1), 'filled', 'blue') % Third principal component
ylim([0.5, 1.5])
xlim([-250, 250])
title('3rd Principal Component')
set(gca,'yticklabel',[])
hold off;



