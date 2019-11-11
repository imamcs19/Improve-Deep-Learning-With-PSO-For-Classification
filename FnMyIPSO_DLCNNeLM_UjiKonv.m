function [MaxAkurasitiapIterasi]=FnMyIPSO_DLCNNeLM_UjiKonv(typeFitur,IterMaxPSO)
% close all
% clear all
% clc

%% Parameter-parameter yang digunakan 
% 
% untuk normalisasi nilai dari fitur
max1 = 300;
% max1 = 263;
min1 = 0;
max2 = 1;
min2 = 0;

%% Proses Training
% pre-Proses data training
%[bykData,byk_fitur,target,norm]=FnPreProses('datatrainForcast.xlsx',...
%    max1, min1, max2, min2);

[bykData,byk_fitur,target,norm]=FnPreProses('datatrainCitraClassify.xlsx',...
    max1, min1, max2, min2);

byk_kelas=numel(unique(target));

%% Tentang PTVPSO (Chen, Hui, Ling, at all, 2011)
% Algoritma particle swarm optimization (PSO) 
% merupakan algoritma optimasi yang pertama kali dikenalkan oleh Kennedy 
% dan Eberhart pada tahun 1995.
% 
% Time variant yang digunakan adalah 
% time varying acceleration coefficients (TVAC) dan 
% time varying inertia weight (TVIW) Dimana,
% TVIW (wmin = 0.4 dan wmax = 0.9) 
% Nilai range c1 dan c2 (TVAC) yang digunakan adalah 
% (c1i=2.5, c1f=0.5) dan c2i=0.5, c2f=2.5) karena terbukti optimal. 
% c1 dan c2 (cognitive dan social components).
% w (bobot inersia), d (banyaknya dimensi data, 
% atau banyaknya fitur data), tmax (iterasi max.)
% ----------------------------------------------------------------------

%% koding PSO
% ========================================
%% Inisialisasi
%pop_size=3;
pop_size=5;
% byk_dimensi=4;
%IterMaxPSO=10;
% IterMaxPSO=20;
wmin=0.4; wmax=0.9;
c1i=2.5; c1f=0.5; c2i=0.5; c2f=2.5;
tmax=IterMaxPSO;

%% Batas Bawah dan Batas Atas Parameter yang dioptimasi, untuk X
% sigma_lower=0.0001; sigma_upper=2; delta_sigma=sigma_upper-sigma_lower;
% lambda_lower=1; lambda_upper=67; delta_lambda=lambda_upper-lambda_lower;
% C_param_lower=1; C_param_upper=200; delta_C_param=C_param_upper-C_param_lower;
% % epsilon_lower= 1e-8; epsilon_upper= 1e-5; delta_epsilon=epsilon_upper-epsilon_lower;
% cLR_lower= 1e-5; cLR_upper= 1e-1; delta_cLR=cLR_upper-cLR_lower;

k_padding_lower=1; k_padding_upper=5; %delta_k_padding=k_padding_upper-k_padding_lower;
Wjk_lower=-0.5; Wjk_upper=0.5; %delta_Wjk=Wjk_upper-Wjk_lower;
%sigma_lower=1; sigma_upper=1000; delta_sigma=sigma_upper-sigma_lower;
%lambda_lower=0.01; lambda_upper=5; delta_lambda=lambda_upper-lambda_lower;
%C_param_lower=0.01; C_param_upper=10000; delta_C_param=C_param_upper-C_param_lower;
% epsilon_lower= 1e-8; epsilon_upper= 1e-5; delta_epsilon=epsilon_upper-epsilon_lower;
%cLR_lower= 1e-4; cLR_upper= 10; delta_cLR=cLR_upper-cLR_lower;

% SLCcLR_lower=[sigma_lower lambda_lower C_param_lower cLR_lower];
% SLCcLR_upper=[sigma_upper lambda_upper C_param_upper cLR_upper];

% susunan cluster dimensi pada partikel PSO untuk DL-CNN base ELM
% byk_covnet_lower = 1; covnet_in_aktivasi_lower=1
% byk_covnet_upper = 10;

%typeFitur=6
bykFilter=3;


if typeFitur==0
    %disp("typeFitur==0")
    byk_dim_data_input=bykFilter*numel(hP{1}{1}(:)');    
elseif typeFitur==4
    %disp("typeFitur==4")
    byk_dim_data_input=numel(hP{1}{1}(:)');
elseif typeFitur==5
    %disp("typeFitur==5")
    byk_dim_data_input=bykFilter;
elseif typeFitur==6
    %disp("typeFitur==6")
    byk_dim_data_input=bykFilter+numel(norm{1}(1,:));
end

pengaliHidden=1;
FC1_byk_neuron_hidden_layer=pengaliHidden*5;
FC2_byk_neuron_hidden_layer=pengaliHidden*7;
FC3_byk_neuron_hidden_layer=pengaliHidden*4;

FC1_ukuran_baris_Wjk=FC1_byk_neuron_hidden_layer;
ukuran_kolom_Wjk=byk_dim_data_input;
FC1_Wjk_lower = Wjk_lower.*ones(1,FC1_ukuran_baris_Wjk*ukuran_kolom_Wjk);
FC1_Wjk_upper = Wjk_upper.*ones(1,FC1_ukuran_baris_Wjk*ukuran_kolom_Wjk);

FC2_ukuran_baris_Wjk=FC2_byk_neuron_hidden_layer;
ukuran_kolom_Wjk=byk_dim_data_input;
FC2_Wjk_lower = Wjk_lower.*ones(1,FC2_ukuran_baris_Wjk*ukuran_kolom_Wjk);
FC2_Wjk_upper = Wjk_upper.*ones(1,FC2_ukuran_baris_Wjk*ukuran_kolom_Wjk);

FC3_ukuran_baris_Wjk=FC3_byk_neuron_hidden_layer;
ukuran_kolom_Wjk=byk_dim_data_input; 
FC3_Wjk_lower = Wjk_lower.*ones(1,FC3_ukuran_baris_Wjk*ukuran_kolom_Wjk);
FC3_Wjk_upper = Wjk_upper.*ones(1,FC3_ukuran_baris_Wjk*ukuran_kolom_Wjk);

% [FC1_ukuran_baris_Wjk FC2_ukuran_baris_Wjk FC3_ukuran_baris_Wjk]
% ukuran_kolom_Wjk
% 
% pause(50000)

% k_padding=[3 5 7 9 11] % 2*int(rand) + 1

SLCcLR_lower = [k_padding_lower FC1_Wjk_lower FC2_Wjk_lower FC3_Wjk_lower];
SLCcLR_upper = [k_padding_upper FC1_Wjk_upper FC2_Wjk_upper FC3_Wjk_upper];

byk_dimensi=size(SLCcLR_lower,2);

% replika matrik, dari satu baris menjadi barisnya sebanyak pop_size
repmat_SLCcLR_lower=ones(pop_size,1)*SLCcLR_lower;
repmat_SLCcLR_upper=ones(pop_size,1)*SLCcLR_upper;

SLCcLR_delta=SLCcLR_upper-SLCcLR_lower;
repmat_SLCcLR_delta=ones(pop_size,1)*SLCcLR_delta;

%% Batas Bawah dan Batas Atas Parameter yang dioptimasi, untuk V
prosentase_V=0.6;
% Vsigma_lower=-prosentase_V*sigma_upper; Vsigma_upper=prosentase_V*sigma_upper;
% Vlambda_lower=-prosentase_V*lambda_upper; Vlambda_upper=prosentase_V*lambda_upper;
% VC_param_lower=-prosentase_V*C_param_upper; VC_param_upper=prosentase_V*C_param_upper;
% VcLR_lower=-prosentase_V*cLR_upper; VcLR_upper=prosentase_V*cLR_upper;

V_SLCcLR_lower=-prosentase_V*SLCcLR_upper;
V_SLCcLR_upper=prosentase_V*SLCcLR_upper;

% replika matrik, dari satu baris menjadi barisnya sebanyak pop_size
repmat_V_SLCcLR_lower=ones(pop_size,1)*V_SLCcLR_lower;
repmat_V_SLCcLR_upper=ones(pop_size,1)*V_SLCcLR_upper;


%% Generate populasi awal

% init size
X=zeros(pop_size,byk_dimensi);
V=zeros(pop_size,byk_dimensi);
Pbest=zeros(pop_size,byk_dimensi);
Gbest=zeros(1,byk_dimensi);
 
%for t=0:IterMaxPSO
for t=0:IterMaxPSO
    t
    
    % hitung nilai w, c1, c2, r1, r2
    w=wmin+((wmax-wmin)*((tmax-t)/tmax));
    c1=((c1f-c1i)*(t/tmax))+c1i;
    c2=((c2f-c2i)*(t/tmax))+c2i;
    r1=rand(1,1); % random [0,1] dengan distribusi uniform
    r2=rand(1,1);
    
    if(t==0)    
        %% masuk ke proses inisialisasi yaitu t=0   
        
        % inisialisasi posisi awal
        X=repmat_SLCcLR_lower + (random('unif',0,1,pop_size,byk_dimensi).*repmat_SLCcLR_delta);
        
        % inisialisasi kecepatan awal
        V;
        
        %pause
        
        % inisialisasi Pbest dan Gbest
        Pbest=X;
        [FitnessAllPbest,FitnessGbest,Gbest]=...
            FnGetFitnessNbestIndividuIPSODL(Pbest);
        
    else
        % update kecepatan 
        %V=w*V+(c1*r1*(Pbest-X))+(c2*r2*(Gbest-X))
        V=(w.*V)+(c1*r1.*(Pbest-X))+(c2*r2.*((ones(pop_size,1)*Gbest)-X));
        V=FnBringtoRangeLowUpIPSODL(V,repmat_V_SLCcLR_lower,repmat_V_SLCcLR_upper);
        
        % update posisi
        X=X+V;
        X=FnBringtoRangeLowUpIPSODL(X,repmat_SLCcLR_lower,repmat_SLCcLR_upper);
        
        % hitung nilai fitness X
        [FitnessAllX,IndexSortingDesc]=FnGetFitnessIPSODL(X);
        
        %FitnessAllX 
        %IndexSortingDesc
        
        %IndexSortingDesc;
        
        % update Pbest dan Gbest
        [FitnessAll_Update_Pbest,Fitness_Update_Gbest,Update_Pbest,Update_Gbest]=...
            FnUpdatePbestGbestIPSODL(FitnessAllPbest,FitnessAllX,FitnessGbest,X,Pbest,Gbest);
        FitnessAllPbest=FitnessAll_Update_Pbest;
        FitnessGbest=Fitness_Update_Gbest;
        Pbest=Update_Pbest;
        Gbest=Update_Gbest;
        
        % simpan rata-rata fitness tiap iterasi
        %MeanFitness(t)=mean(FitnessAllX);
        MeanFitness(t)=mean(FitnessAllPbest);
        MaxFitness(t)=max(FitnessAllPbest);


%         if(mod(t,5)==0) % lakukan random injection
%             % byk pasrtikel yang direplace untuk di-injection
%             byk_partikel_rand_injection=0.2*pop_size;
%             
%             X(IndexSortingDesc(pop_size-byk_partikel_rand_injection+1:pop_size),:)=...
%             repmat_SLCcLR_lower(1:byk_partikel_rand_injection,:) + (random('unif',0,1,byk_partikel_rand_injection,byk_dimensi).*repmat_SLCcLR_delta(1:byk_partikel_rand_injection,:));    
%             
%             % hitung nilai fitness X
%             [FitnessAllX,IndexSortingDesc]=FnGetFitnessIPSODL(X);
%             
%             % update Pbest dan Gbest
%             [FitnessAll_Update_Pbest,Fitness_Update_Gbest,Update_Pbest,Update_Gbest]=...
%                 FnUpdatePbestGbestIPSODL(FitnessAllPbest,FitnessAllX,FitnessGbest,X,Pbest,Gbest);
%             FitnessAllPbest=FitnessAll_Update_Pbest;
%             FitnessGbest=Fitness_Update_Gbest;
%             Pbest=Update_Pbest;
%             Gbest=Update_Gbest;
%             
%         end
        
        %pause(5000)
    end    
end

% menampilkan individu terbaik dan nilai Akurasinya
k=2*floor(Gbest(1))+1;
awal_arr_Wjk11=2;
akhir_arr_Wjk11=FC1_ukuran_baris_Wjk*byk_dim_data_input + 1;
Wjk11=reshape(Gbest(2:akhir_arr_Wjk11),[FC1_ukuran_baris_Wjk byk_dim_data_input]);

awal_arr_Wjk12=akhir_arr_Wjk11 + 1;
akhir_arr_Wjk12=FC2_ukuran_baris_Wjk*byk_dim_data_input + awal_arr_Wjk12 - 1;
Wjk12=reshape(Gbest(awal_arr_Wjk12:akhir_arr_Wjk12),[FC2_ukuran_baris_Wjk byk_dim_data_input]);

awal_arr_Wjk13=akhir_arr_Wjk12 + 1;
akhir_arr_Wjk13=FC3_ukuran_baris_Wjk*byk_dim_data_input + awal_arr_Wjk13 - 1;
Wjk13=reshape(Gbest(awal_arr_Wjk13:akhir_arr_Wjk13),[FC3_ukuran_baris_Wjk byk_dim_data_input]);

% disp('k Wjk11 Wjk12 Wjk13 :');
[k Gbest(2:end)];
Fitness_Gbest=FitnessGbest;
%MAD_Gbest=1/Fitness_Gbest;
Akurasi_Gbest=Fitness_Gbest;
%MeanMADtiapIterasi=1./MeanFitness
MeanAkurasitiapIterasi=MeanFitness
MaxAkurasitiapIterasi=MaxFitness
%ResultMADgBest=MAD_Gbest;
ResultAkurasigBest=Akurasi_Gbest;

%% Save as .mat
save('Gbest',...
    'k',...
    'Wjk11',...
    'Wjk12',...
    'Wjk13',...
    'MeanFitness');  

%disp('Done....!');



% X
% V
% Pbest
% Gbest

