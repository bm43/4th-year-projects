clear all; close all; clc;
load('F1_Elec.mat')
load('F1_PVT.mat')
scatter3(F1_PVT.Acc1P,F1_PVT.Acc1V,F1_PVT.Acc1T, 'Magenta', 'filled') % Blue dots
hold on;
grid on;
scatter3(F1_PVT.Black1P,F1_PVT.Black1V,F1_PVT.Black1T, 'Black', 'filled')
scatter3(F1_PVT.Car1P,F1_PVT.Car1V,F1_PVT.Car1T, 'Yellow', 'filled')
scatter3(F1_PVT.Flour1P,F1_PVT.Flour1V,F1_PVT.Flour1T, 'Blue', 'filled')
scatter3(F1_PVT.Kitchen1P,F1_PVT.Kitchen1V,F1_PVT.Kitchen1T, 'Green', 'filled')
scatter3(F1_PVT.Steel1P,F1_PVT.Steel1V,F1_PVT.Steel1T, 'Red', 'filled')

legend('Acrylic', 'Black Foam', 'Car Sponge', 'Flour Sack', 'Kitchen Sponge', 'Steel Vase')
xlabel('Pressure')
ylabel('Vibration')
zlabel('Temperature')