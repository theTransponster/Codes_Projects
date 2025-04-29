function varargout = Glaucoma(varargin)
% GLAUCOMA MATLAB code for Glaucoma.fig
%      GLAUCOMA, by itself, creates a new GLAUCOMA or raises the existing
%      singleton*.
%
%      H = GLAUCOMA returns the handle to a new GLAUCOMA or the handle to
%      the existing singleton*.
%
%      GLAUCOMA('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in GLAUCOMA.M with the given input arguments.
%
%      GLAUCOMA('Property','Value',...) creates a new GLAUCOMA or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before Glaucoma_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to Glaucoma_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help Glaucoma

% Last Modified by GUIDE v2.5 10-May-2018 20:55:03

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @Glaucoma_OpeningFcn, ...
                   'gui_OutputFcn',  @Glaucoma_OutputFcn, ...
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


% --- Executes just before Glaucoma is made visible.
function Glaucoma_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to Glaucoma (see VARARGIN)

% Choose default command line output for Glaucoma
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes Glaucoma wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = Glaucoma_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
[FileName Path]=uigetfile(('.jpg'));
handles.img=imread(FileName);
image(handles.img,'Parent',handles.axes1)
handles.imgray=rgb2gray(handles.img);
axes(handles.axes2)
imshow(handles.imgray);
pause(0.05);
umbral=graythresh(handles.imgray);
handles.bn=im2bw(handles.imgray,0.76);
axes(handles.axes2)
imshow(handles.bn)
R=handles.img(:,:,1);
G=handles.img(:,:,2);
B=handles.img(:,:,3);
b=regionprops(handles.bn,'BoundingBox','Area');
hold on 
axes(handles.axes2)
imshow(handles.bn)
length(b)
a_mayor=20;
for i=1:1:length(b)
    bb=b(i).BoundingBox;
    aa=b(i).Area;
    if(aa>a_mayor)
        a_mayor=aa;
    end
    im=rectangle('Position',bb,'EdgeColor','b','LineWidth',1);
end
a_mayor;
hold off
maxR=max(max(R));
minR=min(min(R));
maxG=max(max(G));
minG=min(min(G));
maxB=max(max(B));
minB=min(min(B));
%maxG>200&&maxB>130
if(length(b)>1&&a_mayor>5500)
    set(handles.text2,'String','Normal');
else
    set(handles.text2,'String','Glaucoma');
end
guidata(hObject,handles);
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
