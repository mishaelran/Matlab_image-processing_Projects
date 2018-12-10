%% Image proccessing Lab Report 1
% 
% Ran Mishael 
% 1.1 square in square
clear all; close all;
B_background = 50;
B_object = 125;
s = ones(400)*B_background;
s(150:250, 150:250) = B_object;
figure;
imshow(uint8(s)); title('Square In Square');
%%
% 1.2 Stripes
clear all; close all;
n = 300;
a = 256/5;
b = ones(n);

im = b(1:300,1:200);
for i = 0:5
    im(1:300,i*50+1:(i+1)*50+1) = a*i;
end
%show images
figure;
imshow(uint8(im)); title('6 Stripes');
%%
% 1.3
close all; clear all;
% q1 generate signal
%initialize parameters
f = 10; A = 7; fs = 200; t = 1/fs; x = 1; run_t = (0:t:x);
yt = A*cos(2*pi*f*run_t);
%plot y(t) with fs = 200
subplot(2,2,1);
plot(run_t,yt);
grid on
xlabel('time[msec]'); ylabel('Amplitude'); title('cos(2\pi10t) with Fs=200');
% q2 change sampling freq to 12
%initialize parameters
f2 = 10; A = 7; fs2 = 12; t2 = 1/fs2; run_t2 = (0:t2:x);  
yt2 = A*cos(2*pi*f2*run_t2);
subplot(2,2,2);
plot(run_t2,yt2);
grid on
xlabel('time[msec])'); ylabel('Amplitude'); title('cos(2\pi10t) with Fs=12');
% FFT and plot:
frng1 = linspace(-100,100,201);
Yt = abs(fft((yt))); Yt = fftshift(Yt);
subplot(2,2,3)
plot(frng1,Yt);
grid on
xlabel('Frequency[Hz]'); title('abs(fft(f))');
% for y(t2)
Yt2 = abs(fft(yt2));
Yt2=fftshift(Yt2);
subplot(2,2,4)
frng2 = linspace(-6,6,13);
plot(frng2,Yt2);
grid on
xlabel('Frequency[Hz]'); title('abs(fft(f))');
%%
%1.3.5 now for a sine signal uint8 type
close all; clear all;
%q1 generate signal
f = 10;  A = 7; fs = 200; t = 1/fs; x = 1; run_t = (0:t:x);
yt = A*sin(2*pi*f*run_t);
z = uint8(yt);
subplot(2,2,1);
plot(run_t,z);
grid on
xlabel('time[msec]'); ylabel('Amplitude'); title('sin(2\pi10t) with Fs=200');
%q2 q2 change sampling freq to 12
%initialize parameters
f2 = 10; A = 7; fs2 = 12; t2 = 1/fs2; run_t2 = (0:t2:x);   
yt2 = A*sin(2*pi*f2*run_t2); z2 = uint8(yt2);
subplot(2,2,2);
plot(run_t2,z2);
grid on
xlabel('time[msec]'); ylabel('Amplitude'); title('sin(2\pi10t) with Fs=12');
% FFT and plot
frng1 = linspace(-100,100,201);
Yt = abs(fft((z))); Yt = fftshift(Yt);
subplot(2,2,3)
plot(frng1,Yt);
grid on
xlabel('Frequency[Hz]'); title('abs(fft(f))');
Yt2 = abs(fft(z2));
Yt2=fftshift(Yt2);
subplot(2,2,4)
frng2 = linspace(-6,6,13);
plot(frng2,Yt2);
grid on
xlabel('Frequency[Hz]'); title('abs(fft)');

%% 1.4.4 1D convolution, signal,noise and filtering
close all; clear all;
% generate discrete 1D vectors and convolve
f = [1 2 3 1 2]; h = [1 1]; g = conv(f,h);
subplot(3,1,1)
stem(h);
axis([1 10 0 5]); title('h'); ylabel('h[n]');
subplot(3,1,2)
stem(f);
axis([1 10 0 5]); title('f'); ylabel('f[n]');
subplot(3,1,3)
stem(g);
axis([1 10 0 5]); title('g = f*h'); ylabel('g[n]');
%% 1.4.6
clear all; close all;
% generate discrete 1D vectors and convolve
f = [1 2 3 1 2]; h = [1 0 0 0 0 1]; g = conv(f,h);
subplot(3,1,1)
stem(h);
axis([1 10 0 5]); title('h'); ylabel('h[n]');
subplot(3,1,2)
stem(f);
axis([1 10 0 5]); title('f'); ylabel('f[n]');
subplot(3,1,3)
stem(g);
axis([1 10 0 5]); title('g = f*h'); ylabel('g[n]');
%% 1.4.7 White gaussian noise and its histogram
clear all; close all; clc;
%initialize parameters for the white noise and the sampling vector length
mu = 0; sigma = 4; n = 100;
l = 1000000;
a = sigma*randn(l,1)+mu;
figure;
subplot(2,1,1);
plot(a); title(['White Gauss Noise \mu=',num2str(mu),' ;\sigma^2=',num2str(sigma^2)]);
xlabel('Sample'); ylabel('Value');
grid on;
subplot(2,1,2);
[f,x] = hist(a,n); bar(x,f/trapz(x,f));
hold on; grid on;
plot(a,f); title('Histogram for the White Gaussian Noise');
hold off;
%%  1.4.8 sig with white Gauss noise through filter
clear all; close all; clc;
%Generate signals, f[n],f1[n], and h1[n] - the filter's impulse response
f = [1 2 3 1 2]; h1 = [0.2 0.2 0.2 0.2 0.2];
figure; subplot(4,1,1);
stem(f); axis([1 10 0 5]); title('f'); ylabel('f[n]');
%generate noise
no = randn(5,1);
%add noise to f[n]
ad_no = [f;no']; add_no = sum(ad_no);
subplot(4,1,2)
stem(no); axis([1 10 0 10]); title('f1'); ylabel('f1[n]')
%plot h1[n]
subplot(4,1,3)
stem(h1); axis([1 10 0 1]); title('h1'); ylabel('h1[n]');
%convolve f1[n] with h1[n] and plot
g1 = conv(add_no,h1);
subplot(4,1,4)
stem(g1);
axis([1 10 0 10]); title('g1 = f1*h1'); ylabel('g1[n]');
 %% 2.1.2 Mach Bands in different greyscale levels (8,16,32)
 clear all; close all;
 N = 256; k = 0:255; Q = 8;
 for j = 1:N
     image(j,:) = k;
 end
 subplot(2,3,3);plot(uint8(image(100,:))); title('FIRST line');
 subplot(2,3,1);imshow(uint8(image)); title('Ramp');
 subplot(2,3,2);imhist(uint8(image)); title('histogram');
 p = 256/Q;
 image1 = floor(image/p)*p;
 subplot(2,3,4); imshow(uint8(image1)); title([num2str(Q), ' Band Image']);
 subplot(2,3,5); imhist(uint8(image1)); title([num2str(Q),' Band histogram']);
 subplot(2,3,6); plot(uint8(image1(100,:))); title('FIRST line');
 figure;
 Q1 = 8; p1 = 256/Q1;   
 image2 = floor(image/p1)*p1;
 subplot(1,3,1); imshow(uint8(image2)); title([num2str(Q1), ' Band Image']);
 Q2 = 16; p2 = 256/Q2;
 image3 = floor(image/p2)*p2;
 subplot(1,3,2); imshow(uint8(image3)); title([num2str(Q2), ' Band Image']);
 Q3 = 32; p3 = 256/Q3;
 image4 = floor(image/p3)*p3;
 subplot(1,3,3); imshow(uint8(image4)); title([num2str(Q3), ' Band Image']);
%% Adding levels
clear all; close all;
b_num = [4,8,16,32,64,128];
A1 = uint8(linspace(0,255,256));  
b = repmat(A1,256,1);   
subplot(2,3,1); imshow(uint8(b)); title('Ramp Image');
figure
for i = 1:6
    C = floor((0:255)/(256/b_num(i)))*(256/b_num(i));
    D = repmat(C,256,1);
    subplot(2,3,i); imshow(uint8(D)); title([num2str(b_num(i)),' band image']);
end
%% same bands added with salt & pepper noise 
clear all; close all;
b_num = [4,8,16,32,64,128];
A1 = uint8(linspace(0,255,256));
b = repmat(A1,256,1);
subplot(2,3,1); imshow(uint8(b)); title('Ramp Image');
for i = 1:6
    C = uint8(floor((0:255)/(256/b_num(i)))*(256/b_num(i)));
    D = repmat(C,256,1);
    Dn = imnoise(D,'salt & pepper',0.02);
    subplot(2,3,i); imshow(uint8(Dn)); title([num2str(b_num(i)),' band image']);
end
%%  same bands with added Gaussian noise 
b_num = [4,8,16,32,64,128];
A1 = uint8(linspace(0,255,256));
b = repmat(A1,256,1);
subplot(2,3,1); imshow(uint8(b)); title('Ramp Image');
for i = 1:6
    C = uint8(floor((0:255)/(256/b_num(i)))*(256/b_num(i)));
    D = repmat(C,256,1);
    Dn = imnoise(D,'gaussian',0,0.01);
    subplot(2,3,i); imshow(uint8(Dn)); title([num2str(b_num(i)),' band image']);
end
%% Generating and representing 2D signals (q.a-e)
close all; clear all;
Cyc = 256; freq = 256 ; t = linspace(0,1,Cyc);
%q. a,b
f1 = 5; I1 = uint8(127 - 50*sin(2*pi*f1*t));
im1 = double(repmat(I1,freq,1));
subplot(2,2,1)
imshow(uint8(im1)); title('IM1: f = 5 in x axis')
X50 = min(min(im1));
X51 = max(max(im1));f2=10; 
I2 = uint8(127 - 100*sin(2*pi*f2*t));
im2 = double(repmat(I2',1,freq));
subplot(2,2,2)
imshow(uint8(im2)); title('IM2: f = 10 in y axis')
im3 = zeros(Cyc);
for m = 1:Cyc
    for n = 1:Cyc
        im3(m,n) = 127 - 127*sin(2*pi*f2*t(m)+2*pi*f1*t(n));
    end
end
subplot(2,2,3)
imshow(uint8(im3)); title('IM3: f_x = 5, f_y = 10')
%q.d
I4 = im1./3+im2./3+im3./3;
subplot(2,2,4)
imshow(uint8(I4)); title('IM4 = IM1+IM2+IM3')
%q.c
figure;
subplot(2,2,1)
imshow(uint8(im1),[]); title('IM1[]: f_x = 5')
subplot(2,2,2)
imshow(uint8(im2),[]); title('IM2[]: f_y = 10')
subplot(2,2,3)
imshow(uint8(im3),[]); title('IM3[]: f_x = 5 ,f_y = 10')
subplot(2,2,4)
imshow(uint8(I4),[]); title('IM4[] = IM1+IM2+IM3')
%q.e 3D repersentation of the summed signal
figure;
surf(double(I4),'EdgeColor', 'interp');title('3D image');
%%  2.2.2.3 Campbell-Robson Diagram 
clear all;close all;
%initialize parameters
fs = 1024; t = 1/fs;   x = (1:512)*t; Col = 127; Len = length(x);
%set the amplitude and frequency functions
A = exp(linspace(0,log(Col),Len)); 
freq = exp(linspace(0,log(0.2*Len),Len));
CR_im = zeros(Len); A_im = zeros(length(A));

for y = 1:Len
         CR_im(y,:) = Col+A(y)*cos(2*pi*freq.*x); 
end
imshow(uint8(CR_im)); title('CR Diagram')
for m = 1:length(A)
    A_im(m,:) = A;
end
figure; subplot(3,1,1);
imshow(uint8(A_im'))
subplot(3,1,2); plot(A); title('A'); ylabel('Amplitude');grid;
subplot(3,1,3); plot(freq); title('freq'); ylabel('frequency');grid;

%% CR while the amplitude is constant 
clear all;close all;
%set freq to 1024 and C to 127
fs=512; t=1/fs;   x =(1:256)*t;
Col=127;
Len=length(x);
freq=exp(linspace(0,log(0.2*Len),Len)); A=60;
CR_im=zeros(Len);
for y=1:Len
         CR_im(y,:)=Col+A*sin(2*pi*freq.*x); 
end
imshow(uint8(CR_im));title('Campbell-Robson')
figure;plot(CR_im(19,:)); title('frequency growth');xlabel('COLOMN');ylabel('frequency'); grid;
%% CR while the frequency is constant.
clear all;close all;
%set feq to 1024. 
fs=512; t=1/fs;   x =(1:256)*t;
Col=127;
Len=length(x);
freq=20;
A=exp(linspace(0,log(Col),Len));
CR_im=zeros(Len);
for y=1:Len
         CR_im(y,:)=Col+A(y)*sin(2*pi*freq.*x); 
end
imshow(uint8(CR_im));title('CR diagram')
figure;plot(CR_im(:,19)); title('Amp growth');xlabel('ROWs');ylabel('Amplidute');
grid;
%% 2.3.2 
% q.a matrix multiplied
close all; clear all;
f = [2,3,1;1,4,0]
[x y] = size(f);
H = [1 0;1 1] 
g = conv2(f,H)
figure;
stem3(g);
h1 = [1 0 1 0 0 1 0 ;0 0 0 0 1 0 0;0 0 0 1 0 0 1;0 0 1 0 0 0 0;0 0 1 1 0 0 1] 
figure;
g1 = conv2(f,h1)
stem3(g1);

%%  
% q.b tire.tiff X 4
close all; clear all
P = imread('tire.tif');
[x y] = size(P);
H = zeros(x+1,y+1);
H(1,1) = 1; H(1,end)=1; H(end,1)=1; H(end,end)=1;
res_p = conv2(P,H);
imshow(uint8(res_p)); title('4 tires');

%% 
% q.c tire.tiff X 4 with 50 pixels of distance
P = imread('tire.tif');
[x y] = size(P);
H = zeros(x+51,y+51);
H(1,1) = 1;H(1,end)=1;H(end,1)=1;H(end,end)=1;
res_p = conv2(P,H);
imshow(uint8(res_p)); title('4 tires added with 50 black pixels of distance');
%% 2.3.3.3
% q.1 Adding Gaussian noise
close all;clear all;
im = imread('pout.tif');
% Generate the different filters
Wlpf = 1/9*[1, 1, 1; 1, 1, 1; 1, 1, 1];
Whpf = 1/9*[-1, -1, -1; -1, 8, -1; -1, -1, -1];
Wshrp = 1/9*[-1, -1, -1; -1, 17, -1; -1, -1, -1];
Wlplc = [0, -1, 0; -1, 4, -1; 0, -1, 0];    
% Add the Gaussian noise to the image
Gauss_im = imnoise(im,'gaussian',0,0.01);
% Convolve 
res_low = conv2(Wlpf, double(Gauss_im));
res_high = conv2(Whpf, double(Gauss_im));
res_sharp = conv2(Wshrp, double(Gauss_im));
res_laplace = conv2(Wlplc, double(Gauss_im));
% plot the results
subplot(2,3,1);
imshow(im); title('pout.tiff'); %original 
subplot(2,3,2);
imshow(uint8(Gauss_im)); title('im+Gauss noise');% original + gauss noise
subplot(2,3,3);
imshow(uint8(res_low)); title('pic through LPF');
subplot(2,3,4);
imshow(uint8(res_high)); title('pic through HPF');
subplot(2,3,5);
imshow(uint8(res_sharp)); title('pic through Sharp');
subplot(2,3,6);
imshow(uint8(res_laplace)); title('pic through Laplace');
% Median filter (a non linear filter)
res_med = medfilt2(Gauss_im, [3 3]);
figure;
imshow(res_med); title('pic through Medfilt2');
% Error, SNR, PSNR calculations 
Er = double(im)-double(Gauss_im); sq_er = Er.^2; 
MSE = sum(sum(sq_er))/(numel(sq_er)); 
SNR = mean(im.^2)/mean(Gauss_im.^2); PSNR = 255^2/MSE;
%% 
% q.3 Adding "salt & pepper" noise; 0.1 Density
close all;clear all;
im = imread('pout.tif');
% Generate the different filters
Wlpf = 1/9*[1, 1, 1; 1, 1, 1; 1, 1, 1];
Whpf = 1/9*[-1, -1, -1; -1, 8, -1; -1, -1, -1];
Wshrp = 1/9*[-1, -1, -1; -1, 17, -1; -1, -1, -1];
Wlplc = [0, -1, 0; -1, 4, -1; 0, -1, 0];    
% Add the Salt&pepper noise to the image
sp_im = imnoise(im,'salt & pepper',0.1);
% Convolve 
res_low = conv2(Wlpf, double(sp_im));
res_high = conv2(Whpf, double(sp_im));
res_sharp = conv2(Wshrp, double(sp_im));
res_laplace = conv2(Wlplc, double(sp_im));
% plot the results
subplot(2,3,1);
imshow(im); title('pout.tiff'); %original 
subplot(2,3,2);
imshow(uint8(sp_im)); title('im+s&p noise');% original + s&p noise
subplot(2,3,3);
imshow(uint8(res_low)); title('pic through LPF');
subplot(2,3,4);
imshow(uint8(res_high)); title('pic through HPF');
subplot(2,3,5);
imshow(uint8(res_sharp)); title('pic through Sharp');
subplot(2,3,6);
imshow(uint8(res_laplace)); title('pic through Laplace');
% Median filter (a non linear filter)
res_med = medfilt2(sp_im, [3 3]);
figure;
imshow(res_med); title('pic through Medfilt2');
%%
% Error, SNR, PSNR calculations for  pout and tire images
close all;clear all;
tire = imread('tire.tif'); im = imread('pout.tif');
Gauss_im = imnoise(im,'gaussian',0,0.01);
sp_tire = imnoise(tire,'salt & pepper',0.1);
% clean and plot variations for pout.tif
subplot(1,3,1); imshow(im); title('clean pout');
subplot(1,3,2); imshow(uint8(Gauss_im)); title('pout+Gauss noise'); 
G_im_med = medfilt2(Gauss_im, [3 3]);%pout+Gauss through Median
subplot(1,3,3);imshow(G_im_med); title('pout+Gauss: through Median');
% clean and plot variations for tire.tif
figure;
subplot(1,3,1); imshow(tire); title('clean tire');
subplot(1,3,2); imshow(uint8(sp_tire)); title('tire+s&p noise');
sp_med = medfilt2(sp_tire, [3 3]);%tire+s&p through Median
subplot(1,3,3);imshow(sp_med); title('tire+s&p: through Median');
% Error calculations for pout with Gaussian noise
G_Er = double(im)-double(Gauss_im); G_sq_er = G_Er.^2; 
G_MSE = sum(sum(G_sq_er))/(numel(G_sq_er)); 
G_SNR = mean(im.^2)/mean(Gauss_im.^2); G_PSNR = 255^2/G_MSE;
% Error calculations for tire with s&p noise
sp_Er = double(tire)-double(sp_tire); sp_sq_er = sp_Er.^2; 
sp_MSE = sum(sum(sp_sq_er))/(numel(sp_sq_er)); 
sp_SNR = mean(tire.^2)/mean(sp_tire.^2); sp_PSNR = 255^2/sp_MSE;
%%
% summary 3.3 sinus y(t) 
close all; clear all; 
% freq = 66.67[Hz]
% initialize parameters
fs = 200; t = 1/fs; freq = 66.67; A = 1; run_t = (0:t:1);
yt = A*sin(2*pi*freq*run_t);
figure;
plot(run_t,yt); xlabel('time[sec]'); ylabel('Amplitude');
title('sin(2\pi66.67t), f_s=200[Hz]');
% freq = 65.93[Hz]
% initialize parameters
freq2 = 65.93; t2 = 1/fs; run_t2 = (0:t2:1);
yt2 = A*sin(2*pi*freq2*run_t2);
figure;
plot(run_t2,yt2); xlabel('time[sec]'); ylabel('Amplitude');
title('sin(2\pi65.93t), f_s=200[Hz]');
%%
%3.6
close all; clear all;
%Generate signals
f = [1 2 3 4 5]; 
h = [1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1];
g = conv(f,h);
%plot 
subplot(3,1,1); stem(f); title('f');
subplot(3,1,2); stem(h); title('h');
subplot(3,1,3); stem(g); title('g = f*h');

 
