function varargout = babybeat(varargin)
% BABYBEAT M-file for babybeat.fig
%    BABYBEAT, by itself, creates a new BABYBEAT or raises the existing
%    singleton*.
%    H = BABYBEAT returns the handle to a new BABYBEAT or the handle to
%    the existing singleton*.
%    BABYBEAT('CALLBACK',hObject,eventData,handles,...) calls the local
%    function named CALLBACK in BABYBEAT.M with the given input
%    arguments.
%    BABYBEAT('Property','Value',...) creates a new BABYBEAT or raises the
%    existing singleton*.  Starting from the left, property value pairs are
%    applied to the GUI before babybeat_OpeningFcn gets called.  An
%    unrecognized property name or invalid value makes property application
%    stop.  All inputs are passed to babybeat_OpeningFcn via varargin.
%    *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%    instance to run (singleton)".
% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @babybeat_OpeningFcn, ...
    'gui_OutputFcn',  @babybeat_OutputFcn, ...
    'gui_LayoutFcn',  [] , ...
    'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end
 
if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT
% --- Executes just before babybeat is made visible.
function babybeat_OpeningFcn(hObject, eventdata, handles, varargin)
handles.output = hObject;
guidata(hObject, handles);
guidata(hObject, handles);
% Uploading images as button
[a,map]=imread('stop11.jpg');
[r,c,d]=size(a);
x=ceil(r/150);
y=ceil(c/120);
g=a(1:x:end,1:y:end,:);
g(g==255)=5.5*255;
set(handles.pushbutton3,'CData',g);
[a,map]=imread('play11.jpg');
[r,c,d]=size(a);
x=ceil(r/150);
y=ceil(c/120);
g=a(1:x:end,1:y:end,:);
g(g==255)=5.5*255;
set(handles.pushbutton2,'CData',g);
[a,map]=imread('21.jpg');
[r,c,d]=size(a);
x=ceil(r/150);
y=ceil(c/120);
g=a(1:x:end,1:y:end,:);
g(g==255)=5.5*255;
set(handles.pushbutton6,'CData',g);
[a,map]=imread('bpm.jpg');
[r,c,d]=size(a);
x=ceil(r/120);
y=ceil(c/350);
g=a(1:x:end,1:y:end,:);
g(g==255)=5.5*255;
set(handles.pushbutton9,'CData',g);
 
% --- Outputs from this function are returned to the command line.
function varargout = babybeat_OutputFcn(hObject, eventdata, handles)
varargout{1} = handles.output;
 
% Show background images
function axes6_CreateFcn(hObject, eventdata, handles)
imshow('bpm.jpg');
 
% --- Executes during object creation, after setting all properties.
function axes9_CreateFcn(hObject, eventdata, handles)
imshow('backnew.jpg');
 
% --- Executes on pressing the PLAY button.
function pushbutton2_Callback(hObject, eventdata, handles)
 
imaqreset; clear all; clc;
global mf;
mf=1;
alarm=load('alarma.wav');
%%% Setting Preview Window With Rectangle a4 (240,240) %%%
info = imaqhwinfo('winvideo');
vid = videoinput('winvideo', 1);
set(vid,'ReturnedColorSpace','RGB');
vid = videoinput('winvideo', 1,'RGB24_640x480');
nFrames = 300;
set(vid,'FramesPerTrigger',nFrames);
vidRes = get(vid, 'VideoResolution');
nBands = get(vid, 'NumberOfBands');
hImage = image( zeros(vidRes(2), vidRes(1), nBands) );
preview(vid, hImage);
gg = impoly(gca,[240, 240; 320, 240; 320,270; 240,270 ]);
setColor(gg,'black');
start(vid);
while (vid.FramesAcquired<nFrames)
end
nt1 = getdata(vid,nFrames);
preview(vid);
 
%%%%%%%%%%%%%%%%%%%  Explanation  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% in this part we create 3 different signals xg1, xg2, xg3 that
% each of them will implement the algorithm on a different place
% in the forehead area. After getting applying FFT on each of them
% we will check which picks exisist in all three of them. a pick that
% exists get the maximum value, and one that is not get the minimum
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
vidHeight =480;
vidWidth = 640;
xg1=zeros(1,nFrames-19,'double');
xg2=zeros(1,nFrames-19,'double');
xg3=zeros(1,nFrames-19,'double');
NumOfPixels=vidHeight*vidWidth;
sumg1=0;
sumg2=0;
sumg3=0;
%%%%%%%%%%%%%%%%%%%  Explanation  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% We also measured the color group of each subject, and in order to do
% so we compared the averaged RGB color of his skin to the skin color
% chart as described in the final report. below are the value of the RGB
% value of all the color group in the chart
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
x_R=zeros(1,nFrames-19,'double');
x_B=zeros(1,nFrames-19,'double');
R=[244 236 250 253 253 254 250 243 244 251 252 254 255 255 241 238 224 242
   235 235 227 225 223 222 199 188 156 142 121 100 101 96 87 64 49 27];
G=[242 235 249 251 246 247 240 234 241 252 248 246 249 249 231 226 210 226
   214 217 196 193 193 184 164 151 107 88 77 49 48 49 50 32 37 28];
B=[245 233 247 230 230 229 239 229 234 244 237 225 225 225 195 173 147 151
   159 133 103 106 123 119 100 98 67 62 48 22 32 33 41 21 41 46];
 
 
%%% Working only on the green channel %%%
mov(1:nFrames) = ...
    struct('cdata', zeros(vidHeight, vidWidth, 3, 'double'),...
    'colormap', []);
 
 
%%%%%%%%%%%%%%%%%%%%%%%  Explanation  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% These loops calculates the spatial average in each frame in the
% selected ROI. The results are stored in the arrays og xg1/2/3.
% We start after the 19th frame because of the camera
% We are working on an area of 1200 pixels
% xg1 is the main ROI, and xg2/3 are shifted 15 pixels left and right
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x=240;
y=240;
for k = 1 : 1:nFrames-19
    mov(k).cdata = nt1(:,:,:,k+19);
    for i=y :1:y+20
        for j=x:1:x+60
            sumg1=double(mov(k).cdata(i,j,2))+sumg1;
            sumg2=double(mov(k).cdata(i,j-15,2))+sumg2;
            sumg3=double(mov(k).cdata(i,j+15,2))+sumg3;
            Skin_color_R= Skin_color_R+ double(mov(k).cdata(i,j,1));
            Skin_color_B= Skin_color_B+ double(mov(k).cdata(i,j,3));
        end
    end
    nop= 21*61;   %%Number of pixels
    xg1(k)=double(sumg1/nop);
    xg2(k)=double(sumg2/nop);
    xg3(k)=double(sumg3/nop);
    x_R(k)= double(Skin_color_R/nop);
    x_B(k)=double(Skin_color_B/nop);
    sumg1=0;
    sumg2=0;
    sumg3=0;
    Skin_color_R=0;
    Skin_color_B=0;
end
Skin_color_RR=mean(xg1);
Skin_color_GG=mean(x_R);
Skin_color_BB=mean(x_B);
 
%%%%%%%%%%%%%%%%%%%%%%%  Explanation  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Here we are calculating the minimum distance between the subject skin
% color to all of the groups, and selecting the one that mostly match
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:1:36
    Distance=((R(i)-Skin_color_RR)^2+(G (i)-Skin_color_GG)^2+(B(i)-Skin_color_BB)^2)^0.5;
    if (Distance<Final_Distance)
        Final_Distance=Distance;
        Color_Group=i;
    end
end
 
%%%%%%%%%%%%%%%%%%%%%%%  Explanation  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Now we will remove the DC part, and apply the FFT transform.
% Fs is 30 (frames per second) - depends on the computer and the camera
% The Yg4 signal is the final result after comparing between all three 
% signals.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xgm1=xg1-mean(xg1);
xgm2=xg2-mean(xg2);
xgm3=xg3-mean(xg3);
Full_Time_Array_1=xgm1;
Full_Time_Array_2=xgm2;
Full_Time_Array_3=xgm3;
Fs=30;
T=1/Fs;
L=nFrames-19;
t = (0:L-1);
NFFT = 2^nextpow2(L);
Yg1 = fft(xgm1,NFFT)/L;
Yg2 = fft(xgm2,NFFT)/L;
Yg3 = fft(xgm3,NFFT)/L;
f = Fs/2*linspace(0,1,NFFT/2+1);
Yg4=Yg1;
Y1=2*abs(Yg1(1:NFFT/2+1));
Y2=2*abs(Yg2(1:NFFT/2+1));
Y3=2*abs(Yg3(1:NFFT/2+1));
index=0;
f1=0;
figure
plot(f,2*abs(Yg1(1:NFFT/2+1)))
title('First graph of the main ROI')
xlabel('Frequency (Hz)')
ylabel('|Yg1(f)|')
xlim ([0,6])
 
%%%%%%%%%%%%%%%%%%%%%%%  Explanation  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Finding the pulse with a peak detector
% We will look for the frquency with the highest amplitude
% First we will look in the whole region of 1Hz to 3.5Hz
% And in the next measure we will only look close to the previos pulse that
% was found (8 samples to the left and to the right)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Finding the pulse in the main ROI
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for (a=floor((length(f)/15)*0.9)+1: floor((length(f)/15)*3.5))
    if f1<Y1(a)
        f1=Y1(a);
        index=a;
    end
end
f1_old=f1;
index_old=index;
Detected_PULSE_1(1)=60*(index-1)*15/(length(Y1)-1)
Pulse_Amplitude_1(1)=Y1(index);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Finding the next pulse in the main ROI
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
index2=0;
PICK=Y1(index);
f1=0;
Y1(index)=0;
for (b=floor((length(f)/15)*0.9)+1 : floor((length(f)/15)*3.5))
    if f1<Y1(b)
        f1=Y1(b);
        index2=b;
    end
end
NEXTPULSE_1(1)=60*(index2-1)*15/(length(Y1)-1)
Next_Pulse_Amplitude_1(1)=Y1(index2);
 
%%%%%%%%%%%%%%%%%%%%%%%  Explanation  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is the improvement of the algorithm that compare 3 overlapping
%  areas, and finds peaks that exists in all of them. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
for(a=floor((length(f)/15)*0.9)+1: floor((length(f)/15)*3.5))
    if ((abs(Yg1(a))>abs(Yg1(a-2))&&abs(Yg1(a))>abs(Yg1(a+2)))&&(abs(Yg2(a))>abs(Yg2(a-2))&&abs(Yg2(a))>abs(Yg2(a+2)))&&(abs(Yg3(a))>abs(Yg3(a-2))&&abs(Yg3(a))>abs(Yg3(a+2))))
        A=max(abs(Yg3(a)),abs(Yg1(a)));
        Yg4(a)=max(A,abs(Yg2(a)));
    else
        B=min(abs(Yg3(a)),abs(Yg1(a)));
        Yg4(a)=min(abs(Yg2(a)),B);
    end
end
Y4=2*abs(Yg4(1:NFFT/2+1));
for (a=floor((length(f)/15)*0.9)+1: floor((length(f)/15)*3.5))
    if f11<Y4(a)
        f11=Y4(a);
        index1=a;
    end
end
f11_old=f11;
index1_old=index1;
Detected_PULSE_2(1)=60*(index1-1)*15/(length(Y4)-1)
Pulse_Amplitude_2(1)=Y4(index1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Finding the next high pulse 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
index2=0;
PICK=Y4(index1);
f11=0;
Y4(index1)=0;
for (b=floor((length(f)/15))+1 : floor((length(f)/15)*3.5))
    if f11<Y4(b)
        f11=Y4(b);
        index2=b;
    end
end
NEXTPULSE_2(1)=60*(index2-1)*15/(length(Y4)-1)
Next_Pulse_Amplitude_2(1)=Y4(index2);
if (Detected_PULSE_1(1)< 60)
    sound(alarm);
S.ls = uicontrol('style','text',...
    'unit','pix',...
    'position',[535 40 85 60],...
    'min',0,'max',2,...
    'fontsize',40,...
    'string',sprintf('%d', round(Detected_PULSE_1(1))));
 
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% New movies
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
while (mf)
    %%% Settings Configurations %%%
    clear mov; clear vid; clear nt;
    imaqreset;
    info = imaqhwinfo('winvideo');
    vid = videoinput('winvideo', 1);
    set(vid,'ReturnedColorSpace','RGB');
    vid = videoinput('winvideo', 1,'RGB24_640x480');
    nFrames = 30*1;
    set(vid,'FramesPerTrigger',nFrames);
    
%%%%%%%%%%%%%%%%%%%     Explanation  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% in this part we create 3 different new signals xgnew1, xgnew2, xgnew3
% In each of them we will store the last 9 seconds in the start of
% the array (without the DC part), and a new one second will be stored
% in the rest of the array.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    xgnew1=zeros(1,length(xg1) ,'double');
    xgnew1(1:length(xg1)-nFrames+19)=xgm1(nFrames-18:length(xg1));
    xgnew2=zeros(1,length(xg2) ,'double');
    xgnew2(1:length(xg2)-nFrames+19)=xgm2(nFrames-18:length(xg2));
    xgnew3=zeros(1,length(xg3) ,'double');
    xgnew3(1:length(xg3)-nFrames+19)=xgm3(nFrames-18:length(xg3));
    start(vid);
    while (vid.FramesAcquired<nFrames)
    end
%%  Bringing Image Data into the MATLAB Workspace %%%
    nt1 = getdata(vid);%,nFrames);
    clear vid;
    vidHeight =480;
    vidWidth = 640;
    NumOfPixels=vidHeight*vidWidth;
    sumg1=0;
    mov(1:nFrames) = ...
        struct('cdata', zeros(vidHeight, vidWidth, 3, 'double'),...
        'colormap', []);
    
    x=240;
    y=240;
    for k = 1 : 1:nFrames-19
        mov(k).cdata = nt1(:,:,2,k+19);
        for i=y :1:y+20
            for j=x:1:x+60
                sumg1=double(mov(k).cdata(i,j))+sumg1;
                sumg2=double(mov(k).cdata(i,j-15))+sumg2;
                sumg3=double(mov(k).cdata(i,j+15))+sumg3;
            end
        end
        nop=21*61;   %%Number of pixels
        xgnew1(length(xg1)-nFrames+19+k)=double(sumg1/nop);
        sumg1=0;
        xgnew2(length(xg2)-nFrames+19+k)=double(sumg2/nop);
        sumg2=0;
        xgnew3(length(xg3)-nFrames+19+k)=double(sumg3/nop);
        sumg3=0;
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    clear Yg1; clear f; clear Y1;clear Yg2;clear Yg3;clear Yg4;
    xg1=xgnew1;
    xg2=xgnew2;
    xg3=xgnew3;
    xgm1_temp(1:length(xg1)-nFrames+19)=xgm1(nFrames-18:length(xg1));
    xgm1_temp(length(xg1)-nFrames+20:length(xg1))=xg1(length(xg1)-nFrames+20:length(xg1))-mean(xg1(length(xg1)-nFrames+20:length(xg1)));
    Full_Time_Array_1(length(Full_Time_Array_1)+1:length(Full_Time_Array_1)+nFrames-19)=xg1(length(xg1)-nFrames+20:length(xg1))-mean(xg1(length(xg1)-nFrames+20:length(xg1)));
    xgm2_temp(1:length(xg2)-nFrames+19)=xgm2(nFrames-18:length(xg2));
    xgm2_temp(length(xg2)-nFrames+20:length(xg2))=xg2(length(xg2)-nFrames+20:length(xg2))-mean(xg2(length(xg2)-nFrames+20:length(xg2)));
    Full_Time_Array_2(length(Full_Time_Array_2)+1:length(Full_Time_Array_2)+nFrames-19)=xg2(length(xg2)-nFrames+20:length(xg2))-mean(xg2(length(xg2)-nFrames+20:length(xg2)));
    xgm3_temp(1:length(xg3)-nFrames+19)=xgm3(nFrames-18:length(xg3));
    xgm3_temp(length(xg3)-nFrames+20:length(xg3))=xg3(length(xg3)-nFrames+20:length(xg3))-mean(xg3(length(xg3)-nFrames+20:length(xg3)));
    Full_Time_Array_3(length(Full_Time_Array_3)+1:length(Full_Time_Array_3)+nFrames-19)=xg1(length(xg3)-nFrames+20:length(xg3))-mean(xg1(length(xg3)-nFrames+20:length(xg3)));
    L=length(xg1);
    t = (0:L-1);
    NFFT = 2^nextpow2(L);
    Yg1 = fft(xgm1_temp,NFFT)/L;
    Yg2 = fft(xgm2_temp,NFFT)/L;
    Yg3 = fft(xgm3_temp,NFFT)/L;
    Yg4=Yg1;
    f = Fs/2*linspace(0,1,NFFT/2+1);
    Y1=2*abs(Yg1(1:NFFT/2+1));
    f1=0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Finding the pulse near the previous pulse found
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for (a=max(18,index_old-8): index_old+8)
        if f1<Y1(a)
            f1=Y1(a);
            index=a;
        end
    end
    xgm1=xgm1_temp;
    xgm2=xgm2_temp;
    xgm3=xgm3_temp;
    index_old=index;
    f1_old=f1;
    Detected_PULSE_1(p+1)=60*(index-1)*15/(length(Y1)-1);
    Pulse_Amplitude_1(p+1)=Y1(index);
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Finding the next pulse in the main ROI
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    index2=0;
    PICK=Y1(index);
    f1=0;
    Y1(index)=0;
    for (b=max(18,index_old-8): index_old+8)
        if f1<Y1(b)
            f1=Y1(b);
            index2=b;
        end
    end
    NEXTPULSE_1(p+1)=60*(index2-1)*15/(length(Y1)-1);
    Next_Pulse_Amplitude_1(p+1)=Y1(index2);
%%%%%%%%%%%%%%%%%%%%%%%  Explanation  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is the improvement of the algorithm that compare 3 overlapping
%  areas, and finds peaks that exists in all of them. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for (a=max(18,index1_old-8): index1_old+8)
        if ((abs(Yg1(a))>abs(Yg1(a-2))&&abs(Yg1(a))>abs(Yg1(a+2)))&&(abs(Yg2(a))>abs(Yg2(a-2))&&abs(Yg2(a))>abs(Yg2(a+2)))&&(abs(Yg3(a))>abs(Yg3(a-2))&&abs(Yg3(a))>abs(Yg3(a+2))))
            A=max(abs(Yg3(a)),abs(Yg1(a)));
            Yg4(a)=max(A,abs(Yg2(a)));
        else
            B=min(abs(Yg3(a)),abs(Yg1(a)));
            Yg4(a)=min(abs(Yg2(a)),B);
            %            index=a;
        end
    end
    Y4=2*abs(Yg4(1:NFFT/2+1));
    for (a=max(18,index1_old-8): index1_old+8)
        if f11<Y4(a)
            f11=Y4(a);
            index1=a;
        end
    end
    f11_old=f11;
    index1_old=index1;
    Detected_PULSE_2(p+1)=60*(index1-1)*15/(length(Y4)-1)
    Pulse_Amplitude_2(p+1)=Y4(index1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Finding the next high pulse 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    index2=0;
    PICK=Y4(index1);
    f11=0;
    Y4(index1)=0;
    for (b=max(18,index1_old-8): index1_old+8)
        if f11<Y4(b)
            f11=Y4(b);
            index2=b;
        end
    end
    NEXTPULSE_2(p+1)=60*(index2-1)*15/(length(Y4)-1)
    Next_Pulse_Amplitude_2(p+1)=Y4(index2);
    if (Detected_PULSE_1(p+1)< 60)
    sound(alarm);
    
    S.ls = uicontrol('style','text',...
        'unit','pix',...
        'position',[535 40 85 60],...
        'min',0,'max',2,...
        'fontsize',40,...
        'string',sprintf('%d', round(Detected_PULSE_1(p+1))));
    
end
imaqreset; clear all; clc;
info = imaqhwinfo('winvideo');
vid = videoinput('winvideo', 1);
nFrames = 30*1;
 
%%%  Setting specific color space : Gray scale/ RGB/ ... %%%
set(vid,'ReturnedColorSpace','RGB');
vid = videoinput('winvideo', 1,'RGB24_640x480');
 
%%%  Setting specific number of frames to acquire per trigger %%%
set(vid,'FramesPerTrigger',nFrames);
vidRes = get(vid, 'VideoResolution');
nBands = get(vid, 'NumberOfBands');
hImage = image( zeros(vidRes(2), vidRes(1), nBands) );
preview(vid, hImage);
gg = impoly(gca,[240, 240; 320, 240; 320,270; 240,270 ]);
setColor(gg,'black');
S.ls = uicontrol('style','text',...
    'unit','pix',...
    'position',[535 40 85 60],...
    'min',0,'max',2,...
    'fontsize',40,...
    'string',sprintf('%s',' ' ));
 
% --- Executes on pressing the STOP button
function pushbutton3_Callback(hObject, eventdata, handles)
global mf;
mf =0;
 
% --- Executes on pressing the QUIT button.
function pushbutton6_Callback(hObject, eventdata, handles)
quit;
 
% --- Executes during object creation, after setting all properties.
function axes7_CreateFcn(hObject, eventdata, handles)
imaqreset;  clc;
info = imaqhwinfo('winvideo');
vid = videoinput('winvideo', 1);
nFrames = 30*1;
 
%%%  Setting specific color space : Gray scale/ RGB/ ... %%%
set(vid,'ReturnedColorSpace','RGB');
vid = videoinput('winvideo', 1,'RGB24_640x480');
%set(vid,'ReturnedColorSpace','rgb'):
 
%%%  Setting specific number of frames to acquire per trigger %%%
set(vid,'FramesPerTrigger',nFrames);
vidRes = get(vid, 'VideoResolution');
nBands = get(vid, 'NumberOfBands');
hImage = image( zeros(vidRes(2), vidRes(1), nBands) );
 
preview(vid, hImage);
gg = impoly(gca,[240, 240; 320, 240; 320,270; 240,270 ]);
setColor(gg,'black');
 
% --- Executes on button press in pushbutton9.
function pushbutton9_Callback(hObject, eventdata, handles)

function [handles_out] = test(handles)
set(handles.display, 'String','Hello');
handles_out = handles;

function pushbutton2_Callback(hObject, eventdata, handles)
handles = test(handles);
guidata(hObject, handles);
