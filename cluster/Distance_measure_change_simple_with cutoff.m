%%Code to Calculate Distance between a Cluster (Homer) and another Cluster
%%(AMPAR, NMDAR...) before and after some change Original code by Chaoyi
%%11/10/2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Modified to cutoff the data according to some distance to the soma
%%%%% (Cutoff) by Andre
%%%%%%%%%%%%%%%
%%%%%TO DO
% Better algorithm to do the cutoff (right know just a flat cut in just one
% direction, needs to calculate the distance to a central point)
% 

clear;%clear
close all;% Close all figures
%%
%load ampar and homer cluster center data before
path_before='D:\Andre\Data\2017\LTD\20171027\control1\analysis';
if ~exist('fileName1','var')|| isempty(fileName1)
    [userfilein, userdirin]=uigetfile({
        '*.xlsx','Data file (*.xlsx)';...
        '*.*','All Files (*.*)'},'Select the ampar cluster center before file to process',...
        path_before);
    fileName1=fullfile(userdirin,userfilein);
else
    if ~exist(fileName1,'file')
        fprintf('File not found: %s\n',fileName1);
        return;
    end
end

AMPAR_Center_before = xlsread(fileName1);

if ~exist('fileName2','var')|| isempty(fileName2)
    [userfilein, userdirin]=uigetfile({
        '*.xlsx','Data file (*.xlsx)';...
        '*.*','All Files (*.*)'},'Select the homer cluster center before file to process',...
        path_before);
    fileName2=fullfile(userdirin,userfilein);
else
    if ~exist(fileName2,'file')
        fprintf('File not found: %s\n',fileName2);
        return;
    else [userdirin,~,~]=fileparts(fileName2);
        userdirin=strcat(userdirin,'\');
    end
end

%%%Cutoff addition 
%%% Change to < or > depending on the orientation of the neuron
cut = 40000
Homer_Center_before = xlsread(fileName2);
cutOff = find(Homer_Center_before(:,2)>cut);
Homer_Center_before = Homer_Center_before(cutOff,:);

d_homer_before=length(Homer_Center_before);
d_ampar_before=length(AMPAR_Center_before);


%%
%load ampar and homer cluster center data after
path_after='D:\Andre\Data\2017\LTD\20171027\control1\analysis';
if ~exist('fileName3','var')|| isempty(fileName3)
    [userfilein, userdirin]=uigetfile({
        '*.xlsx','Data file (*.xlsx)';...
        '*.*','All Files (*.*)'},'Select the ampar cluster center after file to process',...
        path_after);
    fileName3=fullfile(userdirin,userfilein);
else
    if ~exist(fileName3,'file')
        fprintf('File not found: %s\n',fileName3);
        return;
    end
end

AMPAR_Center_after = xlsread(fileName3);

if ~exist('fileName4','var')|| isempty(fileName4)
    [userfilein, userdirin]=uigetfile({
        '*.xlsx','Data file (*.xlsx)';...
        '*.*','All Files (*.*)'},'Select the homer cluster center after file to process',...
        path_after);
    fileName4=fullfile(userdirin,userfilein);
else
    if ~exist(fileName4,'file')
        fprintf('File not found: %s\n',fileName4);
        return;
    else [userdirin,~,~]=fileparts(fileName4);
        userdirin=strcat(userdirin,'\');
    end
end

%%%Cutoff addition 
%%% Change to < or > depending on the orientation of the neuron
Homer_Center_after = xlsread(fileName4);
cutOff2 = find(Homer_Center_after(:,2)>cut);
Homer_Center_after = Homer_Center_after(cutOff2,:);


d_homer_after=length(Homer_Center_after);
d_ampar_after=length(AMPAR_Center_after);

%%
dist_change=zeros(d_homer_after,2);

for i=1:d_homer_after
    X1=Homer_Center_after(i,1);
    Y1=Homer_Center_after(i,2);
    Z1=Homer_Center_after(i,3);
    for j=1:d_ampar_after  %find AMPAR Homer distance after
        X3=AMPAR_Center_after(j,1);
        Y3=AMPAR_Center_after(j,2);
        Z3=AMPAR_Center_after(j,3);
        dist_HA_after(j)=[((X3-X1)^2+(Y3-Y1)^2+(Z3-Z1)^2)^0.5];
        dist_change(i,2)=min(dist_HA_after);
    end
    
    for k=1:d_homer_before  %find the same homer as after from before
        X2=Homer_Center_before(k,1);
        Y2=Homer_Center_before(k,2);
        Z2=Homer_Center_before(k,3);
        dist_HH(k)=[((X2-X1)^2+(Y2-Y1)^2+(Z2-Z1)^2)^0.5];
        real_dist_HH=min(dist_HH);
        ID_HH=find(dist_HH==real_dist_HH); 
        X4=Homer_Center_before(ID_HH,1);  %homer before that is closest to homer after, considered as the same cluster
        Y4=Homer_Center_before(ID_HH,2);
        Z4=Homer_Center_before(ID_HH,3);
        
        for p=1:d_ampar_before    %find the ampar distance to this homer before
            X5=AMPAR_Center_before(p,1);
            Y5=AMPAR_Center_before(p,2);
            Z5=AMPAR_Center_before(p,3);
            dist_HA_before(p)=[((X5-X4)^2+(Y5-Y4)^2+(Z5-Z4)^2)^0.5];
            dist_change(i,1)= min(dist_HA_before);
        end
       
    end
end
%%
% output
xlswrite(strcat(userdirin,'dist-change-homer-nmdar-far4000-LTD.xlsx'), dist_change);