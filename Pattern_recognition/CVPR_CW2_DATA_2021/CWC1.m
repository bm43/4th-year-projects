%% black foam and car sponge
%% a)

%% pressure vs vibration
load F1_PVT.mat
F=F1_PVT;
figure
scatter(F.Black1P,F.Black1V,'r');
%%
hold on;
scatter(F.Car1P,F.Car1V,'b');
xlabel('pressure')
ylabel('vibration')

% LDA data
Tbl_pv=cat(2,cat(1,F.Black1P,F.Car1P),cat(1,F.Black1V,F.Car1V));

% LDA labels
C=cell(20, 1);
C(1:10,:)={'black_foam'};
C(11:20,:)={'car_sponge'};

disp(size(Tbl_pv));
disp(size(C));
% LDA function
lda_pv=fitcdiscr(Tbl_pv,C);
L=lda_pv.Coeffs(2, 1).Linear;
K=lda_pv.Coeffs(2, 1).Const;

% plot LDA function
lx = get(gca, 'Xlim');
ly = -(K + L(1) .* lx) / L(2);
plot(lx, ly, '-g', 'DisplayName', 'LDA')
%% pressure vs temperature
figure
scatter(F.Black1P,F.Black1T,'r');
hold on;
scatter(F.Car1P,F.Car1T,'b');
xlabel('pressure')
ylabel('temperature')

% LDA data
Tbl_pt=cat(2,cat(1,F.Black1P,F.Car1P),cat(1,F.Black1T,F.Car1T));

% LDA function
lda_pt=fitcdiscr(Tbl_pt,C);
L2=lda_pt.Coeffs(2, 1).Linear;
K2=lda_pt.Coeffs(2, 1).Const;

% plot LDA function
lx = get(gca, 'Xlim');
ly = -(K2 + L2(1) .* lx) / L2(2);
plot(lx, ly, '-g', 'DisplayName', 'LDA')
%% temperature vs vibration
figure
scatter(F.Black1T,F.Black1V,'r');
hold on;
scatter(F.Car1T,F.Car1V,'b');
xlabel('temperature')
ylabel('vibration')

% LDA data
Tbl_tv=cat(2,cat(1,F.Black1T,F.Car1T),cat(1,F.Black1V,F.Car1V));

% LDA function
lda_tv=fitcdiscr(Tbl_tv,C);
L3=lda_tv.Coeffs(2, 1).Linear;
K3=lda_tv.Coeffs(2, 1).Const;

% plot LDA function
lx = get(gca, 'Xlim');
ly = -(K3 + L3(1) .* lx) / L3(2);
plot(lx, ly, '-g', 'DisplayName', 'LDA')

%% b) 3D LDA
Tbl_pvt=cat(2,cat(1,F.Black1P,F.Car1P),cat(1,F.Black1V,F.Car1V));
Tbl_pvt=cat(2,Tbl_pvt,cat(1,F.Black1T,F.Car1T));
lda_pvt=fitcdiscr(Tbl_pvt,C);
L4=lda_pvt.Coeffs(2, 1).Linear
K4=lda_pvt.Coeffs(2, 1).Const
figure
scatter3(F.Black1P,F.Black1V,F.Black1T);
hold on;
scatter3(F.Car1P,F.Car1V,F.Car1T)
lx=1050:10:1200;ly=1600:10:2200;
[lx,ly]=meshgrid(lx,ly);
ylim([2000, 2040])

lz=-(K4 + L4(1) .* lx + L4(2) .* ly)/ L4(3);
s=surf(lx,ly,lz);
set(s,'edgecolor','none');
xlabel('pressure');
ylabel('vibration');
zlabel('temperature');
%% b) 3D LDA on PVT

%% c) comments
%{
for the two given objects, we could reduce the dimension by overlooking the
data from the vibration sensor as it doesn't help distinguish the two
objects. Indeed, the measured vibrations were around 2000 for both objects.
In this case considering temperature and pressure is enough for
distinction of black foam and car sponge.

how these may have affected the sensor readings -> ?
%}
%% d) LDA with two objects of my own choice
%object 1:
%object 2:
p1=F.Steel1P;p2=F.Acc1P;
v1=F.Steel1V;v2=F.Acc1V;
t1=F.Steel1T;t2=F.Acc1T;
Tbl_pvt=cat(2,cat(1,p1,p2),cat(1,v1,v2));
Tbl_pvt=cat(2,Tbl_pvt,cat(1,t1,t2));
lda_pvt=fitcdiscr(Tbl_pvt,C);
L4=lda_pvt.Coeffs(2, 1).Linear;
K4=lda_pvt.Coeffs(2, 1).Const;

figure
scatter3(p1,v1,t1);
hold on;
scatter3(p2,v2,t2)
lx=get(gca,'Xlim');ly=get(gca,'Ylim');
[lx,ly]=meshgrid(lx,ly);
lz=-(K4 + L4(1) .* lx + L4(2) .* ly)/ L4(3);
s=surf(lx,ly,lz);
set(s,'edgecolor','none');
xlabel('pressure');
ylabel('vibration');
zlabel('temperature');
%%
%{
test my hypothesis that vibration information is useless when trying to
distinguish two deformable and porous objects. -> kitchen sponge and car
sponge, kitchen and black, kitchen and flour sack
all have similar vibration values, around 2000-2100
indeed they have very similar vibration values so this dimension is
negligible


plus, try to find when the vibration information is useful
wb two non deformable and non porous objects
for steel and acc, vibration value is same again
vibration is not useful
%}

%% use some LDA code found on file exchange