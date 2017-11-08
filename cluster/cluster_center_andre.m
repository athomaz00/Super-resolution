% This script calculates the center of clusters after the analysis by
% my_cluster.m Andre Thomaz 11/07/2017.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%TO DO
%make a function so we dont keep repeating

%%%%%%%%%%Load the homer-synapses positions
if ~exist('fileName_synapses','var')|| isempty(fileName_synapses)
    [userfilein_synapses, userdirin_synapses]=uigetfile({
        '*.mat','Data file (*.mat)';...
        '*.*','All Files (*.*)'},'Select the synapse cluster .mat file ',...
        'C:\Users\athomaz\Google Drive\superresolution\analysis\mycodes\superres-matlab\cluster');
    fileName_synapses=fullfile(userdirin_synapses,userfilein_synapses);
else
    if ~exist(fileName_synapses,'file')
        fprintf('File not found: %s\n',fileName_synapses);
        return;
    end
end

homer = load(fileName_synapses);

warning('off')
[synNumb,synNull]=size(homer.Synapses); % get number of synapses from the output of cluster.m 

%Create the matrix depending on the number of synapses. 
Homer_centers=zeros(synNumb,3);
synNumb_rows(:,1)=[1:synNumb];




%Load ampar data
if ~exist('fileName_ampar','var')|| isempty(fileName_ampar)
    [userfilein_ampar, userdirin_ampar]=uigetfile({
        '*.txt','Data file (*.txt)';...
        '*.*','All Files (*.*)'},'Select the ampar cluster file ',...
        'C:\Users\athomaz\Google Drive\superresolution\analysis\mycodes\superres-matlab\cluster\data test');
    fileName_ampar=fullfile(userdirin_ampar,userfilein_ampar);
else
    if ~exist(fileName_ampar,'file')
        fprintf('File not found: %s\n',fileName_ampar);
        return;
    end
end
ampar = load(fileName_ampar);
amparNumb = size(ampar.Synapses)
Ampar_centers=zeros(amparNumb(1,1),3);

%Load nmdar data
if ~exist('fileName_nmdar','var')|| isempty(fileName_nmdar)
    [userfilein_nmdar, userdirin_nmdar]=uigetfile({
        '*.txt','Data file (*.txt)';...
        '*.*','All Files (*.*)'},'Select the nmdar cluster file ',...
        'C:\Users\athomaz\Google Drive\superresolution\analysis\mycodes\superres-matlab\cluster\data test');
    fileName_nmdar=fullfile(userdirin_nmdar,userfilein_nmdar);
else
    if ~exist(fileName_nmdar,'file')
        fprintf('File not found: %s\n',fileName_nmdar);
        return;
    end
end
nmdar = load(fileName_nmdar);
nmdarNumb = size(nmdar.Synapses)
Nmdar_centers=zeros(nmdarNumb(1,1),3);



for i=1:length(homer.Synapses)
syn_rg=homer.Synapses{i}; % load data


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Find the center of Homer
radii=1;
[Ctr_homer,Sig_homer] = subclust(syn_rg,radii); % find center of a subcluster and the range of influence of the center


%%%%Check if the center is right
%plot cluster
%figure
% scatter3(syn_rg(:,1), syn_rg(:,2), syn_rg(:,3),30,'filled','MarkerFaceColor',[0/255,128/255,255/255],'MarkerEdgeColor','k'); % Plot scattered data in original space
% hold on
% axis equal
% scatter3(Ctr_homer(1,1), Ctr_homer(1,2),Ctr_homer(1,3), 100, 'filled','k'); % Plot center of cluster
%hold off 
 
%Save center position
Homer_centers(i,:) = [Ctr_homer(1,1), Ctr_homer(1,2),Ctr_homer(1,3)];

%Print what center has been calculated
fprintf('Synapse number = %d of %d\n',i, synNumb);
 


end

%%%%%%%%% Calculation of AMPAR Center
for i=1:length(ampar.Synapses)
syn_rg=ampar.Synapses{i}; % load data


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Find the center of Ampar
radii=1;
[Ctr_ampar,Sig_ampar] = subclust(syn_rg,radii); % find center of a subcluster and the range of influence of the center


%%%%Check if the center is right
%figure
%scatter3(syn_rg(:,1), syn_rg(:,2), syn_rg(:,3),30,'filled','MarkerFaceColor',[153/255,255/255,204/255],'MarkerEdgeColor','k'); % Plot scattered data in original space
%hold on
%axis equal
%scatter3(Ctr_ampar(1,1), Ctr_ampar(1,2),Ctr_ampar(1,3), 220, 'filled','k'); % Plot center of cluster
%hold off 
 
%Save center position
Ampar_centers(i,:) = [Ctr_ampar(1,1), Ctr_ampar(1,2),Ctr_ampar(1,3)];

%Print what center has been calculated
fprintf('Ampar number = %d of %d\n',i, amparNumb(1,1));
 


end

%%%%%%%%% Calculation of NMDAR Center
for i=1:length(nmdar.Synapses)
syn_rg=nmdar.Synapses{i}; % load data


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Find the center of Ampar
radii=1;
[Ctr_nmdar,Sig_nmdar] = subclust(syn_rg,radii); % find center of a subcluster and the range of influence of the center


%%%%Check if the center is right
%figure
%scatter3(syn_rg(:,1), syn_rg(:,2), syn_rg(:,3),30,'filled','MarkerFaceColor',[153/255,255/255,204/255],'MarkerEdgeColor','k'); % Plot scattered data in original space
%hold on
%axis equal
%scatter3(Ctr_ampar(1,1), Ctr_ampar(1,2),Ctr_ampar(1,3), 220, 'filled','k'); % Plot center of cluster
%hold off 
 
%Save center position
Nmdar_centers(i,:) = [Ctr_nmdar(1,1), Ctr_nmdar(1,2),Ctr_nmdar(1,3)];

%Print what center has been calculated
fprintf('Nmdar number = %d of %d\n',i, nmdarNumb(1,1));
 


end

%plot all centers
scatter3(Homer_centers(:,1), Homer_centers(:,2), Homer_centers(:,3),10,'filled','MarkerFaceColor',[0/255,128/255,255/255],'MarkerEdgeColor','k');
axis equal
hold on
scatter3(Ampar_centers(:,1), Ampar_centers(:,2), Ampar_centers(:,3),10,'filled','MarkerFaceColor',[153/255,255/255,204/255],'MarkerEdgeColor','k');
scatter3(Nmdar_centers(:,1), Nmdar_centers(:,2), Nmdar_centers(:,3),10,'filled','MarkerFaceColor',[255/255,153/255,153/255],'MarkerEdgeColor','k');


xlswrite(strcat(userdirin_synapses,'Homer_centers-after.xlsx'),Homer_centers);
xlswrite(strcat(userdirin_synapses,'Ampar_centers-after.xlsx'),Ampar_centers);
xlswrite(strcat(userdirin_synapses,'Nmdar_centers-after.xlsx'),Nmdar_centers);




%%%
%Check what can be used from here
% rec_numb_mat=cell2mat(Synapses_properties(:,6));
% 
% ampar_numb_ind=find(rec_numb_mat==1);
% ampar_numb=length(ampar_numb_ind');
% fprintf('ampar = %d\n',ampar_numb);
% fprintf('%2.1f%%\n',100*ampar_numb/synNumb);
% 
% 
% nmdar_numb_ind=find(rec_numb_mat==2);
% nmdar_numb=length(nmdar_numb_ind');
% fprintf('nmdar = %d\n',nmdar_numb);
% fprintf('%2.1f%%\n',100*nmdar_numb/synNumb);
% 
% 
% amp_nmd_numb_ind=find(rec_numb_mat==3);
% amp_nmd_numb=length(amp_nmd_numb_ind');
% fprintf('ampar + nmdar = %d\n',amp_nmd_numb);
% fprintf('%2.1f%%\n',100*amp_nmd_numb/synNumb);
% 
% 
% none_numb_ind=find(rec_numb_mat==0);
% none_numb=length(none_numb_ind');
% fprintf('none = %d\n',none_numb);
% fprintf('%2.1f%%\n',100*none_numb/synNumb);
% 
% if (ampar_numb+nmdar_numb+amp_nmd_numb+none_numb)~=synNumb
%     fprintf('wrong number of points\n');
% else
%    fprintf('Number of synapses = %d\n',ampar_numb+nmdar_numb+amp_nmd_numb+none_numb);
% end
% 
% axis equal
% 
% SynapseSize=[Numb, Vol_Syn, Vol_ampar, Vol_nmdar];
% SynapseSize(amp_nmd_numb_ind,5)=Vol_ampar(amp_nmd_numb_ind)-Vol_nmdar(amp_nmd_numb_ind);
% fprintf('Volume AMPAR larger than NMDAR = %d\n',length(find(SynapseSize(:,5)>0)));
% fprintf('%2.1f%% percent\n',100*length(find(SynapseSize(:,5)>0))/amp_nmd_numb);
% fprintf('AMPAR<NMDAR = %d\n',length(find(SynapseSize(:,5)<0)));
% fprintf('%2.1f%% percent\n',100*length(find(SynapseSize(:,5)<0))/amp_nmd_numb);
% SynapseSize(1,6)=length(find(SynapseSize(:,5)>0));
% SynapseSize(2,6)=100*length(find(SynapseSize(:,5)>0))/amp_nmd_numb;
% SynapseSize(2,7)=100*length(find(SynapseSize(:,5)<0))/amp_nmd_numb;
% SynapseSize(1,7)=length(find(SynapseSize(:,5)<0));
% SynapseSize(1,8)=ampar_numb;
% SynapseSize(2,8)=100*ampar_numb/synNumb;
% SynapseSize(1,9)=nmdar_numb;
% SynapseSize(2,9)=100*nmdar_numb/synNumb;
% SynapseSize(1,10)=amp_nmd_numb;
% SynapseSize(2,10)=100*amp_nmd_numb/synNumb;
% SynapseSize(1,11)=none_numb;
% SynapseSize(2,11)=100*none_numb/synNumb;
% final_table1=num2cell(SynapseSize);
% final_table2=cell2table(final_table1,'VariableNames',{'Number','Vol_Homer','Vol_AMPAR','Vol_NMDAR','AMPAR_NMDAR_Vol','Vol_AMPAR_larger','Vol_NMDAR_larger','AMPAR','NMDAR','AMPAR_NMDAR','none'});
% 
% 
% SynapseSize=[Numb, Vol_Syn, Vol_ampar, Vol_nmdar];
% 
% 
% xlswrite(strcat(userdirin_synapses,'Synapase_properties_',num2str(Syn_buff),'.xlsx'),SynapseSize);
% writetable(final_table2,strcat(userdirin_synapses,'Synapase_properties_header_',num2str(Syn_buff),'.xlsx'));
% 
% 
% homer_vmd=cell2mat(Synapses_properties(:,1));
% homer_vmd_length=length(homer_vmd);
% homer_vmd_tb=zeros(homer_vmd_length,15);
% homer_vmd_tb(:,5:7)=homer_vmd(:,1:3);
% homer_vmd_cell=num2cell(homer_vmd_tb);
% homer_vmd_tb_final=cell2table(homer_vmd_cell,'VariableNames',{'Number','Intensity','Xpx','Ypx','Xnm','Ynm','Znm','LeftWidthpx','RightWidthpx','UpHeightpx','DownHeightpx','XSymmetryperc','YSymmetryperc','WidthminusHeightpx','FrameNumber'});
% writetable(homer_vmd_tb_final,strcat(userdirin_synapses,'aftercluster-corrected-homer-',num2str(Syn_buff),'.txt'),'delimiter','\t');
% 
% 
% ampar_vmd=cell2mat(Synapses_properties(:,2));
% if isempty(ampar_vmd)
% else
%     ampar_vmd_length=length(ampar_vmd);
%     ampar_vmd_tb=zeros(ampar_vmd_length,15);
%     ampar_vmd_tb(:,5:7)=ampar_vmd(:,1:3);
%     ampar_vmd_cell=num2cell(ampar_vmd_tb);
%     ampar_vmd_tb_final=cell2table(ampar_vmd_cell,'VariableNames',{'Number','Intensity','Xpx','Ypx','Xnm','Ynm','Znm','LeftWidthpx','RightWidthpx','UpHeightpx','DownHeightpx','XSymmetryperc','YSymmetryperc','WidthminusHeightpx','FrameNumber'});
%     writetable(ampar_vmd_tb_final,strcat(userdirin_synapses,'aftercluster-corrected-ampar-',num2str(Syn_buff),'.txt'),'delimiter','\t');
% end
% 
% nmdar_vmd=cell2mat(Synapses_properties(:,4));
% if isempty(nmdar_vmd)
% else
%     nmdar_vmd_length=length(nmdar_vmd);
%     nmdar_vmd_tb=zeros(nmdar_vmd_length,15);
%     nmdar_vmd_tb(:,5:7)=nmdar_vmd(:,1:3);
%     nmdar_vmd_cell=num2cell(nmdar_vmd_tb);
%     nmdar_vmd_tb_final=cell2table(nmdar_vmd_cell,'VariableNames',{'Number','Intensity','Xpx','Ypx','Xnm','Ynm','Znm','LeftWidthpx','RightWidthpx','UpHeightpx','DownHeightpx','XSymmetryperc','YSymmetryperc','WidthminusHeightpx','FrameNumber'});
%     writetable(nmdar_vmd_tb_final,strcat(userdirin_synapses,'aftercluster-corrected-nmdar-',num2str(Syn_buff),'.txt'),'delimiter','\t');
% end
% 
