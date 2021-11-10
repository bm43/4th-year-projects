%% Task 4 a

%Load images

%uiopen('C:\Imperial\Year 4\Term 2\Computer Vision and Pattern Recognition\Coursework 1\Images\HG\HG\Grid\gr1.jpg',1)
%uiopen('C:\Imperial\Year 4\Term 2\Computer Vision and Pattern Recognition\Coursework 1\Images\HG\HG\Grid\gr2.jpg',1)
%%

%Calculate points and homography matrix

imshow(gr1);
[xi, yi] = ginput(4);

M1 = [xi(1) yi(1);
      xi(2) yi(2);
      xi(3) yi(3);
      xi(4) yi(4)]

imshow(gr2);
[xi, yi] = ginput(4);
M2 = [xi(1) yi(1);
      xi(2) yi(2);
      xi(3) yi(3);
      xi(4) yi(4)]


MH = [-M1(1,1) -M1(1, 2) -1        0        0        0   M1(1,1)*M2(1,1)     M1(1,2)*M2(1,1)     M2(1,1);
      0        0         0        -M1(1,1) -M1(1,2) -1  M1(1,1)*M2(1,2)     M1(1,2)*M2(1,2)     M2(1,2);
     -M1(2,1) -M1(2, 2) -1        0        0        0   M1(2,1)*M2(2,1)     M1(2,2)*M2(2,1)     M2(2,1);
      0        0         0        -M1(2,1) -M1(2,2) -1  M1(2,1)*M2(2,2)     M1(2,2)*M2(2,2)     M2(2,2);
     -M1(3,1) -M1(3, 2) -1        0        0        0   M1(3,1)*M2(3,1)     M1(3,2)*M2(3,1)     M2(3,1);
      0        0         0        -M1(3,1) -M1(3,2) -1  M1(3,1)*M2(3,2)     M1(3,2)*M2(3,2)     M2(3,2);
     -M1(4,1) -M1(4, 2) -1        0        0        0   M1(4,1)*M2(4,1)     M1(4,2)*M2(4,1)     M2(4,1);
      0        0         0        -M1(4,1) -M1(4,2) -1  M1(4,1)*M2(4,2)     M1(4,2)*M2(4,2)     M2(4,2);
      0        0         0        0        0        0   0                   0                   1];

 DMatrix = [0; 0; 0; 0; 0; 0; 0; 0; 1];
 
 h = inv(MH)*DMatrix;

 HomM = [h(1) h(2) h(3);
         h(4) h(5) h(6);
         h(7) h(8) h(9);]
 
confPoint = HomM*[M1(1,1);M1(1,2); 1];

ProjPoint = [confPoint(1)/confPoint(3); confPoint(2)/confPoint(3); confPoint(3)/confPoint(3)];
OrigPoint = [M2(1,1);M2(1,2); 1];











