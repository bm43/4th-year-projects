clear all; close all; clc;
load('F1_Elec.mat')

F1 = F1_Electrodes;

F1_matrix = [F1.Acc1Elec, F1.Black1Elec, F1.Car1Elec, F1.Flour1Elec, F1.Kitchen1Elec, F1.Steel1Elec];
         
mean_elec = mean(F1_matrix, 2);

F1_stand = F1_matrix - mean_elec;

S_elec = cov(F1_stand');

[elec_Vec, elec_Val] = eig(S_elec);

%% Variance Visualisation
projected_elec = elec_Vec'*F1_stand;

%remember that our eigen vectors are in order of ascending eigen values.
%Thus elec_variance shows to increase. We flip it in the plot so the x
%values correspond to the principal component index.
elec_var = var(projected_elec');


plot(flip(elec_var,2))
%title('Variance on each principal component')
ylabel('Variance of projected data (log scale)')
xlabel('Principal Component')
title('Variance vs Principal Component')
grid on
set(gca, 'YScale', 'log')


%% 3D Visualisation

Elec_vec = [elec_Vec(:,19), elec_Vec(:,18), elec_Vec(:, 17)];

F1_reduced = Elec_vec'*F1_stand;

%scatter3(F1_reduced(1,:), F1_reduced(2,:), F1_reduced(3,:))







Acc1_es = [F1.Acc1Elec] - mean_elec;
Black1_es = [F1.Black1Elec] - mean_elec;
Car1_es = [F1.Car1Elec] - mean_elec;
Flour1_es = [F1.Flour1Elec] - mean_elec;
Kitchen1_es = [F1.Kitchen1Elec] - mean_elec;
Steel1_es = [F1.Steel1Elec] - mean_elec;


Acc1_er = Elec_vec' * Acc1_es;
Black1_er = Elec_vec' * Black1_es;
Car1_er = Elec_vec' * Car1_es;
Flour1_er = Elec_vec' * Flour1_es;
Kitchen1_er = Elec_vec' * Kitchen1_es;
Steel1_er = Elec_vec' * Steel1_es;

scatter3(Acc1_er(1,:), Acc1_er(2,:), Acc1_er(3,:), 'Magenta', 'filled')
title('Data projected on 3 PC')
hold on;
grid on;
scatter3(Black1_er(1,:), Black1_er(2,:), Black1_er(3,:), 'Black', 'filled')
scatter3(Car1_er(1,:), Car1_er(2,:), Car1_er(3,:), 'Yellow', 'filled')
scatter3(Flour1_er(1,:), Flour1_er(2,:), Flour1_er(3,:), 'Blue', 'filled')
scatter3(Kitchen1_er(1,:), Kitchen1_er(2,:), Kitchen1_er(3,:), 'Green', 'filled')
scatter3(Steel1_er(1,:), Steel1_er(2,:), Steel1_er(3,:), 'Red', 'filled')

