clc
clear all
close all
warning off

% typeFitur
% typeFitur, jika = 0, maka default, merger dari Filter ke-1,2 dan 3
% jika = 1, maka hanya menggunakan Filter ke-1
% jika = 2, maka hanya menggunakan Filter ke-2
% jika = 3, maka hanya menggunakan Filter ke-3
% jika = 4, maka hanya menggunakan rata-rata Filter ke-1,2, dan 3 type A
% sehingga banyak fitur sama dengan numel(hP{1}{1}(:)');
% jika = 5, maka hanya menggunakan rata-rata Filter ke-1,2, dan 3 type B
% sehingga banyak fitur sama dengan bykFilter
typeFitur = 6;

% banyak percobaan
% nCoba = 25
% nLoop = 10
%nCoba = 5
nCoba = 5
IterMaxPSO=20
%IterMaxPSO=1

figure
x = 1:IterMaxPSO;
%plot(x,hasilELM','color',[1 0 0]);
title('Plot Uji Konvergensi PSODLCNNELM')
ylabel('Akurasi (%)')
hold on

for i=1:nCoba
        
        % ELM 
%         tic;
%         hasilELM(i,j)=FnArsitekeLM;
%         toc;
%         time_hasilELM(i,j)=toc;
        
        % DLCNN-ELM
%         tic;
%         hasilDLCNNELM(i,j)=FnArsitekDLCNNeLM(typeFitur);
%         toc;
%         time_hasilDLCNNELM(i,j)=toc;
%         
        % PSO-DLCNN-ELM
        tic;
        hasilPSODLCNNELM{i}=FnMyIPSO_DLCNNeLM_UjiKonv(typeFitur,IterMaxPSO);
        if i==1
            plot(x,hasilPSODLCNNELM{i}','r--*','DisplayName','uji ke-1');
        elseif i==2
            plot(x,hasilPSODLCNNELM{i}','g--^','DisplayName','uji ke-2');
        elseif i==3
            plot(x,hasilPSODLCNNELM{i}','b--+','DisplayName','uji ke-3');
        elseif i==4
            plot(x,hasilPSODLCNNELM{i}','k-->','DisplayName','uji ke-4');
        else
            plot(x,hasilPSODLCNNELM{i}','m--o','DisplayName','uji ke-5');  
        end
        
        hold on
        toc;
        time_hasilPSODLCNNELM(i)=toc;
        
        disp(strcat("Uji ke-",num2str(i)));
   
    %final_time_hasilELM(i)=mean(time_hasilELM(i,:));
    %final_time_hasilDLCNNELM(i)=mean(time_hasilDLCNNELM(i,:));
    final_time_hasilPSODLCNNELM(i)=mean(time_hasilPSODLCNNELM(i));
    %Min_hasilELM(i)=min(hasilELM(i,:));
    %Min_hasilDLCNNELM(i)=min(hasilDLCNNELM(i,:));
    Min_hasilPSODLCNNELM(i)=min(hasilPSODLCNNELM{i}(:));
    %Mean_hasilELM(i)=mean(hasilELM(i,:));
    %Mean_hasilDLCNNELM(i)=mean(hasilDLCNNELM(i,:));
    Mean_hasilPSODLCNNELM(i)=mean(hasilPSODLCNNELM{i}(:));
end

legend('show')

%[hasilELM' hasilDLCNNELM' hasilPSODLCNNELM']


% 
% plot(x,hasilDLCNNELM','color',[0 1 0])
% hold on
% 
% plot(x,hasilPSODLCNNELM','color',[0 0 1]);
% hold off



save('hasilujikonv.mat','hasilPSODLCNNELM','Min_hasilPSODLCNNELM',...
    'Mean_hasilPSODLCNNELM')

disp("Done......!");