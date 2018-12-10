%% Image proccessing Lab Report 3
% "Warm up" code
%  
%%
% 1.1 1D FFT and FFTshift
clear all; close all;
% intialize parameters and input signal
amp = 7; freq = 10; fs = 200; t = 1/fs; st_t = 1;
run_t = (0:t:st_t-t);
f = amp*sin(2*pi*run_t*freq);
L = length(f);
L_shift = (-L/2:1:L/2-t)*(fs/L);
%V_f = (-fs/2:1:fs/2-t);
%1.1.3 plot f(t) , abs(F) = abs(FFT(f)) , ffrshift(abs(F))
% log(abs(F)), log(abs(fftshift(F))), angle(F) - Phase
figure(1);
subplot(2,3,1);
plot(run_t,f); title('f(t)=7\cdotsin(2\pi\cdot10t)');
subplot(2,3,2);
plot(L_shift,abs(fft(f))); title('abs(F)');
subplot(2,3,3);
plot(L_shift,fftshift(abs(fft(f)))); title('abs(fftshift(F))');
subplot(2,3,4);
plot(L_shift,log(abs(fft(f)))); title('log(abs(F))');
subplot(2,3,5);
plot(L_shift,log(fftshift(abs(fft(f))))); 
title('log(abs(fftshift(F)))');
subplot(2,3,6);
plot(L_shift,angle(fftshift(fft(f)))); title('angle(F)');
% 1.1.4 same for sinus signal with freq = 10.541[Hz]
freq2 = 10.541; 
f2 = amp*sin(2*pi*run_t*freq2);
L2 = length(f2); 
L2_shift = (-L2/2:1:L2/2-t)*(fs/L2);
%
figure(2);
subplot(2,3,1);
plot(run_t,f2); title('f_2(t)=7\cdotsin(2\pi\cdot10.541t)');
subplot(2,3,2);
plot(L2_shift,abs(fft(f2))); title('abs(F_2)');
subplot(2,3,3);
plot(L2_shift,fftshift(abs(fft(f2)))); title('abs(fftshift(F_2))');
subplot(2,3,4);
plot(L2_shift,log(abs(fft(f2)))); title('log(abs(F_2))');
subplot(2,3,5);
plot(L2_shift,log(fftshift(abs(fft(f2))))); 
title('log(abs(fftshift(F_2)))');
subplot(2,3,6);
plot(L2_shift,angle(fftshift(fft(f2)))); title('angle(F_2)');
% 1.1.5 same for 127*(1+sin(2*pi*t*fc)) signal
f3 = 127*(1+sin(2*pi*run_t*freq));
L3 = length(f3);
L3_shift = (-L3/2:1:L3/2-t)*(fs/L3);
figure(3);
subplot(2,3,1);
plot(run_t,f3); title('f_3(t)=127\cdot(1+sin(2\pi10t)');
subplot(2,3,2);
plot(L3_shift,abs(fft(f3))); title('abs(F_3)');
subplot(2,3,3);
plot(L3_shift,fftshift(abs(fft(f3)))); title('abs(fftshift(F_3))');
subplot(2,3,4);
plot(L3_shift,log(abs(fft(f3))+1)); title('log(abs(F_3))');
subplot(2,3,5);
plot(L3_shift,log(fftshift(abs(fft(f3))))); 
title('log(abs(fftshift(F_3)))');
subplot(2,3,6);
plot(L3_shift,angle(fftshift(fft(f3)))); title('angle(F_3)');
%% 1.2 Reconstruct 1D sig from its FFT representation
clear all; close all;
% intialize parameters and input signals from prev q
amp = 7; freq = 10; fs = 200; t = 1/fs; st_t = 1;
run_t = (0:t:st_t)';
% f1
f = amp*sin(2*pi*run_t.*freq);
% f2
freq2 = 10.541; 
f2 = amp*sin(2*pi*run_t.*freq2);
% f3
f3 = 127*(1+sin(2*pi*run_t*freq));
% 1.2.1
% Reconstruct the signals
rec_f = real(ifft(fft(f)));
rec_f2 = real(ifft(fft(f2)));
rec_f3 = real(ifft(fft(f3)));
% plot for f
figure;
subplot(3,3,1);
plot(run_t,f); ylabel('Amplitude'); title('original: f(t)');
subplot(3,3,2);
plot(run_t,rec_f);
ylabel('Amplitude'); title('ifft(f)');
subplot(3,3,3);
plot(run_t,abs(f-rec_f));
ylabel('Amplitude'); title('f(t) abs error');
% plot for f2
subplot(3,3,4);
plot(run_t,f3); ylabel('Amplitude'); title('f_2(t)');
subplot(3,3,5);
plot(run_t,rec_f2);
ylabel('Amplitude'); title('ifft(f_2)');
subplot(3,3,6);
plot(run_t,abs(f2-rec_f2));
ylabel('Amplitude'); title('f_2(t) abs error');
% plot for f3
subplot(3,3,7);
plot(run_t,f2); ylabel('Amplitude'); title('f_3(t)');
subplot(3,3,8);
plot(run_t,rec_f3);
ylabel('Amplitude'); title('ifft(f_2)');
subplot(3,3,9);
plot(run_t,abs(f3-rec_f3));
ylabel('Amplitude'); title('f_3(t) abs error');
% 1.2.2 Errors calculation
% sqrt(MSE) , SNR[dB] , PSNR[dB]
% for f
Er = double(f)-double(rec_f);
se = Er.^2;
MSE = (sum(sum(se)))/(numel(se));
Sq_MSE = sqrt(MSE);
SNR = 20*log10(mean(f.^2)/mean(rec_f.^2)); % dB
%for f2
Er2 = double(f2)-double(rec_f2);
se2 = Er2.^2;
MSE2 = (sum(sum(se2)))/(numel(se2));
sq_MSE2 = sqrt(MSE2);
SNR2 = 20*log10(mean(f2.^2)/mean(rec_f2.^2)); % dB
% for f3
Er3 = double(f3)-double(rec_f3);
se3 = Er3.^2;
MSE3 = (sum(sum(se3)))/(numel(se3));
Sq_MSE3 = sqrt(MSE3);
SNR3 = 20*log10(mean(f3.^2)/mean(rec_f3.^2)); % dB
PSNR3 = 20*log10((255^2)/MSE3); % dB
%% 1.3 The 1D convolution Theorem
clear all; close all;
% Generate the signals
p = [1 2 3 1 2];
h = [1 1];
%convolve
g = conv(p,h);
% pad the signals with zeros
p_zero_pad = [p 0];
h_zero_pad = [h 0 0 0 0];
%1.3.2 convolution theorem. conv in time --> mul in freq
P = fft(p_zero_pad);
H = fft(h_zero_pad);
G = P.* H;
rec_g = real(ifft(G));
% plot
figure;
subplot(1,2,1);
stem(g); title('g = p*h');
subplot(1,2,2);
stem(rec_g); title('g = FFT^{-1}(F\cdotH)');
%% 1.4 1D signal filtering in the frequency domain
clear all; close all;
% initialize parameters for the signals
fs = 210; freq1 = 10; freq2 = 30; freq3 = 70;
st_t = 1;
run_t = 0:1/fs:(st_t - 1/fs);
f = 1./run_t;
% sig = sum of 3 sinus waves
sig = sin(2*pi*run_t*freq1) + sin(2*pi*run_t*freq2) + sin(2*pi*run_t*freq3);
SIG = fft(sig);
n = length(sig);
% align the sig properly
frq_shift = (-n/2:n/2-1)*(fs/n); % zero-centered frequency range
% Generate the filters
lpf = [ones(1,15) zeros(1,210-15-14) ones(1,14)];
hpf = [zeros(1,65) ones(1,210-65-64) zeros(1,64)];
bpf = [zeros(1,15) ones(1,20) zeros(1,210-15-20-20-14) ones(1,20) zeros(1,14)];
% Apply filters using convolution theorem
SIG_lpf = SIG.* lpf;
SIG_hpf = SIG.* hpf;
SIG_bpf = SIG.* bpf;
%  ifft for all filterd signals
sig_lpf = ifft(SIG_lpf);
sig_hpf = ifft(SIG_hpf);
sig_bpf = ifft(SIG_bpf);
% Plot signal (sum of sigs) and its abs(FFT)
figure(1);
subplot(1,2,1);
plot(run_t,sig);
xlabel('Time[sec]'); ylabel('Amplitude');
title('s(t) = \Sigma{sig_i}');
subplot(1,2,2);
plot(frq_shift,fftshift(abs(SIG)));
xlabel('Frequency[Hz]'); ylabel('Magnitude');
title('abs(fft(sig(t)))');
% Plot results of filters
figure(2);
% for LPF
subplot(3,3,1);
plot(frq_shift,fftshift(abs(lpf)));
ylabel('Magnitude'); xlabel('Frequency[Hz]');
title('LPF');
ylim([0 1.2]);
subplot(3,3,2);
plot(frq_shift,fftshift(abs(SIG_lpf)));
ylabel('Magnitude'); xlabel('Frequency[Hz]');
title('SIG_1 = SIG(f)\cdotH_{LPF}');
subplot(3,3,3);
plot(ifft(SIG_lpf));
xlabel('Time[sec]'); ylabel('Amplitude');
title('ifft(SIG_1)'); axis tight
% for HPF
subplot(3,3,4);
plot(frq_shift,fftshift(abs(hpf)));
ylabel('Magnitude'); xlabel('Frequency[Hz]');
title('HPF');
ylim([0 1.2]);
subplot(3,3,5);
plot(frq_shift,fftshift(abs(SIG_hpf)));
ylabel('Magnitude');
xlabel('Frequency[Hz]');
title('SIG_2 = SIG(f)\cdotH_{HPF}');
subplot(3,3,6);
plot(ifft(SIG_hpf));
xlabel('Time[sec]'); ylabel('Amplitude');
title('ifft(SIG_2)'); axis tight
% for BPF
subplot(3,3,7);
plot(frq_shift,fftshift(abs(bpf)));
ylabel('Magnitude'); xlabel('Frequency[Hz]');
title('BPF');
ylim([0 1.2]);
subplot(3,3,8);
plot(frq_shift,fftshift(abs(SIG_bpf)));
ylabel('Magnitude'); xlabel('frequency[Hz]');
title('SIG_3 = SIG(f)\cdotH_{BPF}');
subplot(3,3,9);
plot(ifft(SIG_bpf));
xlabel('Time[sec]'); ylabel('Amplitude');
title('ifft(SIG_3)'); axis tight;
%% 2.1.1
clear all; close all;
% FFT for one line of image
im = imread('tire.tif');
f_line = im(1,:);
fs = 232; t = 1/fs; V_f = (-fs/2:1:fs/2-t);
% Plot the line
figure();
subplot(1,2,1);
plot(f_line); title('first line of tire');
line_fft = fft(f_line); 
subplot(1,2,2);
plot(line_fft); title('FFT of the first line of tire');
%% 2.2.1 2D fourier Transform
% Machin signal FFT, Phase, Magnitude, add noise
clear all; close all;
%
h = 64; w = 128;
f1 = 2; % 2 B/W changes for both X&Y axis
f2 = 8; % 8 B/W changes for both X&Y axis 
% size parameters (horizontal and vertical)
x = 0:(w-1); 
y = 0:(h-1);
% generate 2D grid coordinates from size vectors
[X,Y] = meshgrid(x,y);
% Generate signals and FFT2 them
s1 = cos(2*pi*X/w*f1).*cos(2*pi*Y/w*f1);
s1_fft = fft2(s1);
s2 = cos(2*pi*X/w*f2).*cos(2*pi*Y/w*f2);
s2_fft = fft2(s2);
% a) plot signals and their FFT
figure(1);
subplot(2,2,1);
imshow(s1,[]); title('Signal 1');
subplot(2,2,3);
imshow(s2,[]); title('Signal 2');
subplot(2,2,2);
imshow((log(abs(s1_fft))),[]); title('abs of FFT of signal 1');
subplot(2,2,4);
imshow((log(abs(s2_fft))),[]); title('abs of FFT of signal 2');
% b) check influence of FFTshift
figure(2);
subplot(2,2,1);
imshow(s1,[]); title('Signal 1');
subplot(2,2,3);
imshow(s2,[]); title('Signal 2');
subplot(2,2,2);
imshow((log(abs(fftshift(s1_fft)))),[]); title('abs of FFTshift of signal 1');
subplot(2,2,4);
imshow((log(abs(fftshift(s2_fft)))),[]); title('abs of FFTshift of signal 2');
% c) Magnitude & Phase
figure(3);
subplot(2,2,1);
imshow(log((abs(s2_fft))),[]);
title('Signal 2 Magnitude');
subplot(2,2,3);
imshow(log(fftshift(abs(s2_fft))),[]);
title('Signal 2 fftshift Magnitude');
subplot(2,2,2);
imshow(angle(s2_fft),[]);
title('Signal 2 Phase');
subplot(2,2,4);
imshow(fftshift(angle(s2_fft)),[]);
title('Signal 2 fftshift Phase');
% d) 3D plot with mesh grid
figure(4);
subplot(1,2,1);
mesh(X,Y,log(abs(s1_fft)));
title('Signal 1 Magnitude');
subplot(1,2,2);
mesh(X,Y,log(fftshift(abs(s1_fft))));
title('Signal 1 fftshift Magnitude');
% d) Add gauss noise
s3 = imnoise(s2,'gaussian',0,0.01);
s3_fft = fft2(s3);
% Repeat for noise added signal
figure(5);
imshow(s3,[]); title('S_3 = S_2+noise');
%
figure(6);
subplot(2,2,1);
imshow(log(abs(s3_fft)),[]); title('s_3 magnitude');
subplot(2,2,2);
imshow(log(fftshift(abs(s3_fft))),[]); title('s3+fftshift Magnitude');
subplot(2,2,3);
imshow(angle(s3_fft),[]); title('s_3 Phase');
subplot(2,2,4);
imshow(fftshift(angle(s3_fft)),[]);
title('s_3+fftshift Phase');
% 3D plot with mesh grid
figure(7);
subplot(1,2,1);
mesh(X,Y,log(abs(s3_fft))); title('s_3 magnitude');
subplot(1,2,2);
mesh(X,Y,log(fftshift(abs(s3_fft))));
title('s3+fftshift Magnitude');
%% same with image from moodle
clear all; close all;
%load image from moodle
addpath 'd:\Documents\MATLAB\im proccess lab\3'
im = imread('imageToAnalyze.png');
%figure;
%imshow(im); title('Image to analyze');
%h = 480; w = 640;
%im_t = (im-min(im(:)))/(max(im(:))-min(im(:)));
%im_fft = fft2(im_t);
figure(1);
%subplot(1,2,1);
imshow(im); title('Image to analyze');
%subplot(2,2,2);
%imshow(im_fft); title('im_t');
figure(2)
%subplot(1,2,2);
imshow(fftshift(abs(fft2(im))),[]);
figure(3);
mesh(fftshift(abs(fft2(im))));
title('3D mesh fftshift of image to analyze');
%% 2.2 Class: image FFT, FFTshift, Angle w/wout FFTshift
clear all; close all;
im = imread('cameraman.tif');
im_fft2 = fft2(im);
figure();
subplot(2,3,1);
imshow(im); title('cameraman');
subplot(2,3,2);
imshow(log(abs(im_fft2)),[]); title('log(abs(F))');
subplot(2,3,3);
imshow(log(fftshift(abs(im_fft2))),[]); title('log(abs(fftshift(F)))');
subplot(2,3,4);
imshow(angle(im_fft2),[]); title('Angle(F)');
subplot(2,3,5);
imshow(angle(angle(fftshift(im_fft2))),[]);
title('Angle with fftshift');
%% 2.3 human eye sensitivity - Phase or magnitude
clear all; close all;
% Load images
cam = imread('cameraman.tif'); txt_im = imread('text.png');
% a) plot 
figure(1);
subplot(2,3,1);
imshow(cam); title('Original');
subplot(2,3,2);
imshow(log(abs(fft2(cam))),[]); 
title('Magnitude, no fftshift');
subplot(2,3,3);
imshow(log(abs(fftshift(fft2(cam)))),[]); 
title('Magnitude');
subplot(2,3,4);
imshow(fftshift(angle(fft2(cam))),[]);
title('Phase');
subplot(2,3,5);
imshow(real(ifft2(abs(fft2(cam)))),[]);
title('Magnitude based');
subplot(2,3,6);
imshow(real(ifft2(exp(1i*angle(fft2(cam))))),[]);
title('Phase based');
% b)building an image using another im mag and phase
figure(2);
subplot(2,2,1);
imshow(cam); title('cam');
subplot(2,2,2);
imshow(txt_im); title('txt');
subplot(2,2,3); 
imshow(real(ifft2(abs(fft2(txt_im)).*(exp(1i*angle(fft2(cam)))))),[]);
title('real of IFFT2(|cam|,<txt)');
subplot(2,2,4);
imshow(real(ifft2(abs(fft2(cam)).*(exp(1i*angle(fft2(txt_im)))))),[]);
title('real of IFFT2(|txt|,<cam)');
%% 2.3.2 machin
clear all; close all;
% 
p1 = imread('text.png');
% plot
figure();
subplot(2,3,1);
imshow(p1); title('Original');
% magnitude, no fftshift
subplot(2,3,2);
imshow(log(abs(fft2(p1))),[]); title('magnitude, no fftshift');
% magnitude
subplot(2,3,3);
imshow(log(fftshift(abs(fft2(p1)))),[]); title('magnitude');
% angle
subplot(2,3,4);
imshow(fftshift(angle(fft2(p1))),[]); title('angle');
% reconstruction based on magnitude
subplot(2,3,5);
imshow(real(ifft2(abs(fft2(p1)))),[]); title('magnitude based');
% reconstruction based on phase
subplot(2,3,6);
imshow(real(ifft2(exp(sqrt(-1)*angle(fft2(p1))))),[]);
title('angle based');
%% 2.4.2 Filters in the frequency domain
clear all; close all;
f_l = 0.05;     % Relative frequency
f_h = 0.1;     % Relative frequency

IMG = imread('saturn.png');
I_gray = rgb2gray(IMG);

im_size = size(I_gray);
[X , Y] = meshgrid(1:im_size(2),1:im_size(1));
x_cen = (im_size(2) - 1)/2 + 1; y_cen = (im_size(1) - 1)/2 + 1;

LPF_f = zeros(im_size); LPF_f(((X - x_cen).^2 + (Y - y_cen).^2) <= f_l^2 * min(im_size).^2 / 4) = 1;
HPF_f = ones(im_size) - LPF_f;
LPF2_f = zeros(im_size); LPF2_f(((X - x_cen).^2 + (Y - y_cen).^2) <= f_h^2 * min(im_size).^2 / 4) = 1;
BPF_f = LPF2_f - LPF_f; BSF_f = ones(im_size) - BPF_f;
IM_LPF_shifted = fftshift(fft2(I_gray)) .* LPF_f; im_LPF = real(ifft2(ifftshift(IM_LPF_shifted)));
IM_HPF_shifted = fftshift(fft2(I_gray)) .* HPF_f; im_HPF = real(ifft2(ifftshift(IM_HPF_shifted)));
IM_BPF_shifted = fftshift(fft2(I_gray)) .* BPF_f; im_BPF = real(ifft2(ifftshift(IM_BPF_shifted)));
IM_BSF_shifted = fftshift(fft2(I_gray)) .* BSF_f; im_BSF = real(ifft2(ifftshift(IM_BSF_shifted)));

figure(1);
subplot(2,3,1); imshow(I_gray); title('Original');
subplot(2,3,2); imshow(log(fftshift(abs(fft2(I_gray)))),[]); title('Mag-fftshift');
subplot(2,3,3); imshow(LPF_f,[]); title('Mag-LPF-fftshift');
subplot(2,3,4); imshow(log(abs(IM_LPF_shifted)) ,[]); title('LPF \cdot image fft');
subplot(2,3,5); imshow(im_LPF ,[]); title('LPF recons ifft');
subplot(2,3,6); imshow(abs(im_LPF - double(I_gray)) ,[]); title('LPF ABS Error');
figure(2);
subplot(3,4,1); imshow(HPF_f,[]); title('Mag-HPF-fftshift');
subplot(3,4,2); imshow(log(abs(IM_HPF_shifted)) ,[]); title('HPF \cdot image fft');
subplot(3,4,3); imshow(im_HPF ,[]); title('HPF recons ifft');
subplot(3,4,4); imshow(abs(im_HPF - double(I_gray)) ,[]); title('HPF ABS Error');
subplot(3,4,5); imshow(BPF_f,[]); title('Mag-BPF-fftshift');
subplot(3,4,6); imshow(log(abs(IM_BPF_shifted)) ,[]); title('BPF \cdot image fft');
subplot(3,4,7); imshow(im_BPF ,[]); title('BPF recons ifft');
subplot(3,4,8); imshow(abs(im_BPF - double(I_gray)) ,[]); title('BPF ABS Error');
subplot(3,4,9); imshow(BSF_f,[]); title('Mag-BSF-fftshift');
subplot(3,4,10); imshow(log(abs(IM_BSF_shifted)) ,[]); title('BSF \cdot image FFT');
subplot(3,4,11); imshow(im_BSF ,[]); title('BSF recons ifft');
subplot(3,4,12); imshow(abs(im_BSF - double(I_gray)) ,[]); title('BSF ABS Error');

%% 2.5.2 Convolution theorem: filtering in the freq domain
clear all; close all;
IMG = imread('cameraman.tif');

h1=ones(9); h2=1/9*[-1 -1 -1;-1 8 -1;-1 -1 -1];
h3=1/9*[-1 -1 -1;-1 17 -1;-1 -1 -1]; h4 = [0 -1 0;-1 4 -1;0 -1 0];
size_I = size(IMG); size_h1 = size(h1); size_h2 = size(h2); size_h3 = size(h3); size_h4 = size(h4);
h1_ext = [h1 zeros(size_h1(1),size_I(2)-1); zeros(size_I(1)-1,size_I(2) + size_h1(2)-1)];       % full
h2_ext = [h2 zeros(size_h2(1),size_I(2)-1) ; zeros(size_I(1)-1,size_I(2) + size_h2(2)-1)];      % full
h3_ext = [h3 zeros(size_h3(1),size_I(2)-size_h3(2)) ; zeros(size_I(1)-size_h3(1),size_I(2))];   % valid
h4_ext = [h4 zeros(size_h4(1),size_I(2)-1) ; zeros(size_I(1)-1,size_I(2) + size_h4(2)-1)];      % same
I_ext_h1 = [IMG zeros(size_I(1),size_h1(2)-1) ; zeros(size_h1(1)-1,size_I(2) + size_h1(2)-1)];
I_conv_h1 = conv2(IMG, h1); IconvH = real(ifft2(fft2(I_ext_h1).*fft2(h1_ext)));
I_ext_h2 = [IMG zeros(size_I(1),size_h2(2)-1) ; zeros(size_h2(1)-1,size_I(2) + size_h2(2)-1)];
I_conv_h2 = conv2(IMG, h2, 'full'); IconvH2 = real(ifft2(fft2(I_ext_h2).*fft2(h2_ext)));
I_ext_h3 = [IMG zeros(size_I(1),size_h3(2)-1) ; zeros(size_h3(1)-1,size_I(2) + size_h3(2)-1)];
I_conv_h3 = conv2(IMG, h3, 'valid'); I_conv_h3_conv_th = real(ifft2(fft2(IMG).*fft2(h3_ext)));
I_conv_h3_conv_th = I_conv_h3_conv_th(size_h3(1):end,size_h3(2):end);
I_ext_h4 = [IMG zeros(size_I(1),size_h4(2)-1) ; zeros(size_h4(1)-1,size_I(2) + size_h4(2)-1)];
I_conv_h4 = conv2(IMG, h4, 'same'); I_conv_h4_conv_th = real(ifft2(fft2(I_ext_h4).*fft2(h4_ext)));
I_conv_h4_conv_th = I_conv_h4_conv_th((1+fix(size_h4(1)/2)):(size_I(1)+fix(size_h4(1)/2)),(1+fix(size_h4(2)/2)):(size_I(2)+fix(size_h4(2)/2)));
err_h1 = I_conv_h1 - IconvH; err_h2 = I_conv_h2 - IconvH2; err_h3 = I_conv_h3 - I_conv_h3_conv_th; err_h4 = I_conv_h4 - I_conv_h4_conv_th;

figure(1);
subplot(1,4,1); imshow(IMG,[]); title('Camerman');
subplot(1,4,2); imshow(h1,[]); title('Filter');
subplot(1,4,3); imshow(I_conv_h1,[]); title('2D spatial conv');
subplot(1,4,4); imshow(IconvH,[]); title('2D spatial conv-Theory');
figure(2);
subplot(2,3,1); imshow(I_ext_h1,[]); title('Camerman');
subplot(2,3,2); imshow(fftshift(log(abs(fft2(I_ext_h1)))),[]); title('Mag-fftshift');
subplot(2,3,3); imshow(h1_ext,[]); title('Filter');
subplot(2,3,4); imshow(fftshift(log(abs(fft2(h1_ext)))),[]); title('Filter Mag-fftshift');
subplot(2,3,5); imshow(fftshift(log(abs(fft2(I_ext_h1).*fft2(h1_ext)))),[]); title('Image \cdot filter-fftshift');
subplot(2,3,6); imshow(IconvH,[]); title('Image \cdot filter-fftshift');
figure(3);
subplot(1,4,1); imshow(IMG,[]); title('Camerman');
subplot(1,4,2); imshow(h2,[]); title('Filter'); 
subplot(1,4,3); imshow(I_conv_h2,[]); title('2D spatial conv');
subplot(1,4,4); imshow(IconvH2,[]); title('2D spatial conv-Theory');
figure(4);
subplot(2,3,1); imshow(I_ext_h2,[]); title('Camerman');
subplot(2,3,2); imshow(fftshift(log(abs(fft2(I_ext_h2)))),[]); title('Mag-fftshift');
subplot(2,3,3); imshow(h2_ext,[]); title('Filter');
subplot(2,3,4); imshow(fftshift(log(abs(fft2(h2_ext)))),[]); title('Filter Mag-fftshift');
subplot(2,3,5); imshow(fftshift(log(abs(fft2(I_ext_h2).*fft2(h2_ext)))),[]); title('Image \cdot filter-fftshift');
subplot(2,3,6); imshow(IconvH2,[]); title('Image \cdot filter-fftshift');
figure(5);
subplot(1,4,1); imshow(IMG,[]); title('Camerman');
subplot(1,4,2); imshow(h3,[]); title('Filter');
subplot(1,4,3); imshow(I_conv_h3,[]); title('2D spatial conv');
subplot(1,4,4); imshow(I_conv_h3_conv_th,[]); title('2D spatial conv-Theory');
figure(6);
subplot(2,3,1); imshow(I_ext_h3,[]);title('Camerman');
subplot(2,3,2); imshow(fftshift(log(abs(fft2(I_ext_h3)))),[]); title('Mag-fftshift');
subplot(2,3,3); imshow(h3_ext,[]); title('Filter');
subplot(2,3,4); imshow(fftshift(log(abs(fft2(h3_ext)))),[]); title('Filter Mag-fftshift');
subplot(2,3,5); imshow(fftshift(log(abs(fft2(IMG).*fft2(h3_ext)))),[]); title('Image \cdot filter-fftshift');
subplot(2,3,6); imshow(I_conv_h3_conv_th,[]); title('Image \cdot filter-fftshift');
figure(7);
subplot(1,4,1);imshow(IMG,[]);title('Camerman');
subplot(1,4,2); imshow(h4,[]); title('Filter');
subplot(1,4,3); imshow(I_conv_h4,[]); title('2D spatial conv');
subplot(1,4,4); imshow(I_conv_h4_conv_th,[]); title('2D spatial conv-Theory');
figure(8);
subplot(2,3,1); imshow(I_ext_h4,[]); title('Camerman');
subplot(2,3,2); imshow(fftshift(log(abs(fft2(I_ext_h4)))),[]); title('Mag-fftshift');
subplot(2,3,3); imshow(h4_ext,[]); title('Filter');
subplot(2,3,4); imshow(fftshift(log(abs(fft2(h4_ext)))),[]); title('Filter Mag-fftshift');
subplot(2,3,5); imshow(fftshift(log(abs(fft2(I_ext_h4).*fft2(h4_ext)))),[]); title('Image \cdot filter-fftshift');
subplot(2,3,6); imshow(I_conv_h4_conv_th,[]); title('Image \cdot filter-fftshift');
figure(9);
subplot(2,2,1); plot(abs(err_h1(:))); title('Error signal for h1');
subplot(2,2,2); plot(abs(err_h2(:))); title('Error signal for h2');
subplot(2,2,3); plot(abs(err_h3(:))); title('Error signal for h3');
subplot(2,2,4); plot(abs(err_h4(:))); title('Error signal for h4');
