function [FitnessAll_Update_Pbest,Fitness_Update_Gbest,Update_Pbest,Update_Gbest]=...
    FnUpdatePbestIPSODL(FitnessAllPbestOld,FitnessAllXbaru,FitnessGbestOld,Xbaru,PbestOld,GbestOld)

Update_Pbest=PbestOld;
FitnessAll_Update_Pbest=FitnessAllPbestOld;

FitnessAllPbestOld_minus_FitnessAllXbaru=...
    FitnessAllPbestOld-FitnessAllXbaru;

% jika nilai selisih negatif, maka nilai Fitness pada Xbaru lebih besar
% mencari indek yang nilai selisihnya negatif
idx_selisih_negatif=find(FitnessAllPbestOld_minus_FitnessAllXbaru<0);
if(isempty(idx_selisih_negatif)) 
else
    Update_Pbest(idx_selisih_negatif,:)=Xbaru(idx_selisih_negatif,:);
    FitnessAll_Update_Pbest(idx_selisih_negatif)=FitnessAllXbaru(idx_selisih_negatif);
end

% mencari nilai_MaxFitness dan indexnya dari FitnessAll_Update_Pbest
[nilai_MaxFitness,index_MaxFitness]=max(FitnessAll_Update_Pbest);
Calon_Gbest=Update_Pbest(index_MaxFitness,:);

FitnessCalon_Gbest=nilai_MaxFitness;

% membandingkan Calon_Gbest dengan GbestOld dari nilai fitness-nya
if(FitnessCalon_Gbest>FitnessGbestOld)
   Fitness_Update_Gbest=FitnessCalon_Gbest;
   Update_Gbest=Calon_Gbest;
else
   Fitness_Update_Gbest=FitnessGbestOld; 
   Update_Gbest=GbestOld;
end

