%function cluster(fileName1,fileName2)
clc;%clear
close all;
if ~exist('fileName1','var')|| isempty(fileName1)
    [userfilein1, userdirin1]=uigetfile({
        '*.txt','Data file (*.txt)';...
        '*.*','All Files (*.*)'},'Select the Green file to process',...
        'G:\Andre\Data\2017\LTD\20171103\control1\analysis');
    fileName1=fullfile(userdirin1,userfilein1);
else
    if ~exist(fileName1,'file')
        fprintf('File not found: %s\n',fileName1);
        return;
    end
end
% if ~exist('fileName2','var')|| isempty(fileName2)
%     [userfilein2, userdirin2]=uigetfile({
%         '*.txt','Data file (*.txt)';...
%         '*.*','All Files (*.*)'},'Select the Red file to process',...
%         'C:\Users\athomaz\Dropbox\superresolution\analysis\mycodes\cluster1');
%     fileName2=fullfile(userdirin2,userfilein2);
% else
%     if ~exist(fileName2,'file')
%         fprintf('File not found: %s\n',fileName2);
%         return;
%     end
% end
% %% synapse_2.m for Green
% data1=textread(fileName1);
% data2=textread(fileName2);
% 
% colX1=(data1(:,5));
% colX2=(data2(:,5));
% 
% for i=1:length(colX1)
%% synapse_2.m for Green

data=textread(fileName1);
palmX=(data(:,5));
palmY=(data(:,6));
palmZ=(data(:,7));
data_Syn=[palmX, palmY, palmZ];


% Finding clusters
[Class,type]=dbscan(data_Syn,50,500); 
% Make new matrix  
Syn=[palmX,palmY,palmZ,Class',type'];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Seperate and plot clusters
x=Syn(:,1);
y=Syn(:,2);
z=0.79*Syn(:,3); 
particle=Syn(:,4);
PtType=Syn(:,5);
d=length(particle); %total row number
ptotal=max(particle);%total cluster number
numArrays=ptotal;
Synapses=cell(numArrays,1);
figure
for k=1:ptotal
        dp=find(particle==k); % matrix indcis of points in cluster #k 
        dplength=length(dp);
        %dpmax=max(dp);
        %dpmin=min(dp);
        m(k)=k; %test k value
        x1=x(dp);
        y1=y(dp);
        z1=z(dp);
        PtType1=PtType(dp);
        Syn1=[x1,y1,z1,PtType1];
        Synapses{k}=Syn1;
        %eval(['Synapse_' num2str(k) '=[x1,y1,z1,PtType1]']);% creating submatrix for particle# k sub_k 
        scatter3(x1,y1,z1,10,'.');
        hold all
        clear dp dplength dpmax dpmin x1 y1 z1 PtType1
        fprintf('Synapse number = %d\n',k);
end

hold off
% clear x y z PtType particle k d m x1 y1 z1 dp Syn 
axis equal
% load('D:\mydocument\MATLAB\for codes\Synapses2.mat')
% load('D:\mydocument\MATLAB\for codes\Synapses1.mat')
% userdirin='D:\mydocument\MATLAB\for codes\';
% ptotal=size(Synapses1,1);
fid=fopen([userdirin1 'homer-after-with_clusters.txt'],'w');
for k=1:ptotal
    for m=1:size(Synapses{k},1)
    fprintf(fid,'%f %f %f %d\r\n',Synapses{k}(m,:));
    end
end
fclose(fid);
save(strcat(userdirin1,'Synapses-homer-after.mat'),'Synapses');

% %% synapse_2.m for red
% data=textread(fileName2);
% 
% palmX=data(:,5);
% palmY=data(:,6);
% palmZ=data(:,7);
% data_Syn=[palmX, palmY, palmZ];
% 
% % Finding clusters
% [Class,type]=dbscan(data_Syn,50,500); 
% % Make new matrix  
% Syn=[palmX,palmY,palmZ,Class',type'];
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Seperate and plot clusters
% x=Syn(:,1);
% y=Syn(:,2);
% z=0.79*Syn(:,3); 
% particle=Syn(:,4);
% PtType=Syn(:,5);
% d=length(particle); %total row number
% ptotal=max(particle);%total cluster number
% numArrays=ptotal;
% Synapses2=cell(numArrays,1);
% figure
% for k=1:ptotal
%         dp=find(particle==k); % matrix indcis of points in cluster #k 
%         dplength=length(dp);
%         %dpmax=max(dp);
%         %dpmin=min(dp);
%         m(k)=k; %test k value
%         x1=x(dp);
%         y1=y(dp);
%         z1=z(dp);
%         PtType1=PtType(dp);
%         Syn1=[x1,y1,z1,PtType1];
%         Synapses2{k}=Syn1;
%         %eval(['Synapse_' num2str(k) '=[x1,y1,z1,PtType1]']);% creating submatrix for particle# k sub_k 
%         scatter3(x1,y1,z1,10,'.');
%         hold all
%         clear dp dplength dpmax dpmin x1 y1 z1 PtType1
% end
% 
% hold off
% clear x y z PtType particle k d m x1 y1 z1 dp Syn 
% axis equal
% % load('D:\mydocument\MATLAB\for codes\Synapses2.mat')
% % load('D:\mydocument\MATLAB\for codes\Synapses1.mat')
% % userdirin='D:\mydocument\MATLAB\for codes\';
% % ptotal=size(Synapses2,1);
% fid=fopen([userdirin2 'Red_clusters_locations.txt'],'w');
% for k=1:ptotal
%     for m=1:size(Synapses2{k},1)
%     fprintf(fid,'%f %f %f %d\n',Synapses2{k}(m,:));
%     end
% end
% fclose(fid);
% save(strcat(userdirin2,'Synapses2.mat'),'Synapses2');
    