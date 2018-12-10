%% Image proccessing Lab Report 2
%
% Ran Mishael
% Chapter 1: Machin
% m1.1 changing brightness and contrast by add and mul with a const
%1.1.1
close all; clear all;
%load tire image and apply calculation 
im = imread('tire.tif'); 
im2 = 0.4*im+50; 
% plot
figure;
% original im and its hist
subplot(2,2,1);
imshow(im); title('original tire'); 
subplot(2,2,3); 
imhist(im); title('original hist');
% changed im and its hist
subplot(2,2,2); 
imshow(im2); title('0.4*tire+50');
subplot(2,2,4);
imhist(im2); title('hist (0.4*tire+50)');
%% 1.1.2 changing brightness: add 50
close all; clear all;
%load tire image and apply calculation 
im = imread('tire.tif'); 
im2 = im+50; 
% plot
figure;
% original im and its hist
subplot(2,2,1);
imshow(im); title('original tire'); 
subplot(2,2,3); 
imhist(im); title('original hist');
% changed im and its hist
subplot(2,2,2); 
imshow(im2); title('tire+50');
subplot(2,2,4);
imhist(im2); title('hist (tire+50)');
%% 1.2
%generate the signals and their PDF&CDF
L = 10000;
%uniform dist
uni =round(rand(L,1)*255); 
uni_pdf = hist(uni,50)/L;
uni_cdf=cumsum(uni_pdf);
%Gausse dist
Gss = randn(L,1)/4*255+127.5;
Gss(Gss>255) = 255;
Gss(Gss<0) = 0;
Gss = uint8(Gss);

Gss_pdf = zeros(1,256);
for L = 1:L
    for i = 1:256
        if (Gss(L) == i)
            Gss_pdf(i) = Gss_pdf(i)+1;
        end
    end
end
% 
Gss_pdf = Gss_pdf/sum(Gss_pdf);
Gss_cdf = cumsum(Gss_pdf);
%plot uniform 
subplot(2,2,1);
plot(1:50,uni_pdf);
title('uniform pdf');
subplot(2,2,3);
plot(1:50,uni_cdf);
title('uniform cdf');
%plot Gauss
subplot(2,2,2);
plot(Gss_pdf);
xlim([0 255]); ylim([0 0.01]); title('Gaussian pdf');
subplot(2,2,4);
plot(Gss_cdf);
xlim([0 255]); ylim([0 1]); title('Gaussian cdf');
%% 1.3 1D transformation and its inverse
clear all; close all;
% generate sig f(x), and transformation t[x]
n = 256; amp = 5; f = 0.05;
L = linspace(0,n,1000); 
fx = amp*cos(2*pi*f*L);
% generate transformation t[x] and its inverse
tx = 5*fx+50;
in_tx = (tx-50)/5;
% calculate SNR
SNR = snr(fx, in_tx);
% plot signal
subplot(3,1,1);
plot(L,fx); 
title('f(x)=5\cdotcos(2\pi\cdotf\cdotn)'); axis tight
% plot Transformation T[x]
subplot(3,1,2);
plot(L,tx); 
title('T(x) = 5\cdotf(x)+50'); axis tight
% plot inverse Transformation T^(-1)[x]
subplot(3,1,3);
plot(L,in_tx);
title(['T^{-1}[x] ; ' ,'SNR = ', num2str(SNR)]); axis tight
%% 3.1 contrast & brightness parameters
clear all; close all;
%
J = imread('tire.tif'); J = double(J);
% a) plot J and its hist
figure;
subplot(2,4,1);
imshow(uint8(J)); title('(a) Original');
subplot(2,4,5);
imhist(uint8(J)); title('(b) hist of (a)');
% b) increase brightness, plot result and hist
J_b = J+50;
subplot(2,4,2);
imshow(uint8(J_b)); title('(c) tire+50 brighter');
subplot(2,4,6);
imhist(uint8(J_b)); title('(d) hist of (c)');
% c) decrease contrast, plot result and hist
J_c = 0.4*J+50;
subplot(2,4,3);
imshow(uint8(J_c)); title('(e) 0.4\cdottire+50');
subplot(2,4,7);
imhist(uint8(J_c)); title('(f) hist of (e)');
% d) Negative of J and its hist
J_d = 255-J;
subplot(2,4,4);
imshow(uint8(J_d)); title('(g) Negative');
subplot(2,4,8);
imhist(uint8(J_d)); title('(h) hist of (g)');
%% same for pout.tif
clear all; close all;
%
J = imread('pout.tif'); J = double(J);
% a) plot J and its hist
figure;
subplot(2,4,1);
imshow(uint8(J)); title('(a) Original');
subplot(2,4,5);
imhist(uint8(J)); title('(b) hist of (a)');
% b) decrease brightness, plot result and hist
J_b = J-75;
subplot(2,4,2);
imshow(uint8(J_b)); title('(c) pout-75');
subplot(2,4,6);
imhist(uint8(J_b)); title('(d) hist of (c)');
% c) increase contrast, plot result and hist
J_c = 2.8*(J-75);
subplot(2,4,3);
imshow(uint8(J_c)); title('(e) 2.8\cdot(pout-75)');
subplot(2,4,7);
imhist(uint8(J_c)); title('(f) hist of (e)');
% d) Negative of J and its hist
J_d = 255-J;
subplot(2,4,4);
imshow(uint8(J_d)); title('(g) Negative');
subplot(2,4,8);
imhist(uint8(J_d)); title('(h) hist of (g)');
%% 3.2 Histogram creation without using imhist: 
clear all; close all; 
Histograma = zeros(1,256);
pout = imread('pout.tif');
imgage = pout;
for N = 1:numel(pout)
     Histograma(imgage(N)+1) = Histograma(pout(N)+1)+1;
end
NormHist=Histograma/numel(pout); 
figure;
subplot(2,3,1);
imshow(pout);
title('Pout.tif');
subplot(2,3,2);
bar(Histograma);
xlim([0 255]);
title('Hist created by us');
subplot(2,3,3);
bar(NormHist);
xlim([0 255]);
title('Our normalized Hist');
CDF=cumsum(NormHist);
subplot(2,3,4);
plot(CDF);
xlim([0 255]);
title('CDF');
% Matlab's imhist
subplot(2,3,5);
imhist(pout);
ylim([0 4000]);
title ('matlab imhist')
MatHis=imhist(pout);
result=sum(abs(MatHis'-Histograma))
subplot(2,3,6);
plot(result); 
title ('Delta of Hists');
%% 3.3 adaptive threshhold and Number of Rice Grains 
clear all; close all;
RiceImage=imread('rice.png');
threshhold=110;
%set static threshold of 110 for comparison
subplot(2,3,1);
imshow(RiceImage);
title('Rice Image'); 
[A,B]=size(RiceImage); 
subplot(2,3,2);
Bin = RiceImage>threshhold;
imshow(Bin,[]);
title('single threshhold 110');
Row = mean(RiceImage,2);
subplot(2,3,3); 
plot(Row);
title('Row Mean');
ColMean = mean(RiceImage);
subplot(2,3,5); 
plot(ColMean);
title('Col Mean');
[a,b]=imhist(RiceImage);
subplot(2,3,4); 
plot(b,a);
title('Histogram');
newthr = repmat(Row,1,B); 
Adaptive_B = RiceImage>1.225*newthr; 
Image_medianfilter = medfilt2(Adaptive_B);
subplot(2,3,6); 
imshow(Image_medianfilter,[]);
title('Our Adaptive threshhold');
disp('number of Rice Grains: ');   
RiceNumber = bwconncomp(Image_medianfilter,8);
NumOfRiceGrains  = RiceNumber.NumObjects
%% 3.3.1 Histogram strech
close all; clear all; 
pout=imread('pout.tif');
%load image
Function=[zeros(1,75),linspace(1,255,88),255*ones(1,92)]; 
stretched_pout=Function(double(pout)+1);
subplot(2,4,1)
imshow(pout)
%show pout
title('pout.tif Imgage');
subplot(2,4,5)
imhist(pout);
%pout histrogram
title('pout.tif Histogram'); 
subplot(2,4,2)
imshow(uint8(stretched_pout))
%streched pout + histogram
title('Streched Imgage')
subplot(2,4,6)
imhist(uint8(stretched_pout))
title('Streched Histogram')
adjusted_pout=imadjust(pout);
subplot(2,4,3)
imshow(adjusted_pout)
%show adjusted matlab stretch 
title('adjusted pout');
subplot(2,4,7)
imhist(adjusted_pout)
title('Histogram adjusted');
figure;
plot(Function);
title('transfer function');
%% 3.3.2 histogram equility
clear all; close all;
pout = imread('pout.tif');
[M,N] = size(pout);
H=256;
subplot(3,4,1);
imshow(pout);
title('pout.tif');
[imageHist,Z]=imhist(pout);
subplot(3,4,2);
plot(Z,imageHist);
title('pout hist');
axis tight
Accumulated=cumsum(imageHist)/(M*N);
subplot(3,4,3);
plot(Z,Accumulated);
title('Accum hist');
axis tight
image_Mat=histeq(pout);
subplot(3,4,5);
imshow(image_Mat);
title('Matlab.histeq');
Image_eq_Mat=imhist(image_Mat);
subplot(3,4,6);
plot(Z,Image_eq_Mat);
title('Matlab.histeq');
axis tight
image_Mat_Hist = imhist(image_Mat);
CumSum=cumsum(image_Mat_Hist)/(M*N);
subplot(3,4,7);
plot(Z,CumSum);
title('Matlab.accum');
axis([0,300,0,1])
imageHist(1)=imageHist(1)+1/H; 
Accumulated=cumsum(imageHist);
Ts=ceil(((H*Accumulated)/(M*N))-1);
Im_myEq(:,:)=Ts(pout(:,:));
subplot(1,4,4);
plot(Z,Ts);
title('histeq t.f');
subplot(3,4,9);
imshow(uint8(Im_myEq));
title('our histeq');
my_new_h=imhist(uint8(Im_myEq));
subplot(3,4,10);
plot(Z,my_new_h);
title('our histeq');
axis tight
New_ACC=cumsum(my_new_h)/(M*N);
subplot(3,4,11);
plot(Z,New_ACC);
title('our accumulated');
axis([0,330,0,1])
% Maximum difference
figure;
difference=double(our_image_EQ)-double(image_Mat);
MAXIMUM_Diff=max(max(abs(difference)));
subplot(2,2,1);
imshow(uint8(difference));
title('difference between eqs');
subplot(2,2,2);
imshow(uint8(difference+127),[]);
title('Normalized difference');
subplot(2,2,3);
imhist(uint8(difference));
title(['histogram difference=',num2str(MAXIMUM_Diff)]);
subplot(2,2,4);
imhist(uint8(difference+127));
title('hist Normalized diff');
%% 3.3.3 histograma swap
clear all; close all;
pout = imread('pout.tif');
tire = imread('tire.tif');
pout_hist = imhist(pout);
tire_hist = imhist(tire);
pout_cdf = cumsum((pout_hist)/numel(pout));
tire_cdf = cumsum((tire_hist)/numel(tire));
M = zeros(256,1,'uint8');
M2 = zeros(256,1,'uint8'); 
for n = 1:256
    [~,ind] = min(abs(pout_cdf(n) - tire_cdf));
    M(n) = ind-1;
    [~,ind2] = min(abs(tire_cdf(n) - pout_cdf));
    M2(n) = ind2-1;
end
img1_adjust = M(double(pout)+1);
img2_adjust = M2(double(tire)+1);
pout_adjust = imhist(img1_adjust);
tire_adjust = imhist(img2_adjust);
pout_cdf_adjust = cumsum((pout_adjust)/numel(img1_adjust));
tire_cdf_adjust = cumsum((tire_adjust)/numel(img2_adjust));
figure(1);
subplot (2,4,1);
imshow(pout);
title('pout.tif');
subplot (2,4,2);
plot(pout_cdf);
title('pout.tif CDF');
xlim ([0 255]);
ylim ([0 1]);
subplot (2,4,5);
imshow(tire);
title('tire.tif');
subplot (2,4,6);
plot(tire_cdf);
title('tire.tif CDF');
xlim ([0 255]);
ylim ([0 1]);
subplot (2,4,3);
imshow(uint8(img1_adjust));
title('swapped pout');
subplot (2,4,7);
imshow(uint8(img2_adjust));
title('swapped tire'); 
subplot (2,4,8);
plot(tire_cdf_adjust);
title('S of p2');
xlim ([0 255]);
ylim ([0 1]);
subplot (2,4,4);
plot(pout_cdf_adjust);
title('S of p1');
xlim ([0 255]);
ylim ([0 1]);



