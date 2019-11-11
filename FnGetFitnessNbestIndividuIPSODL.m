function [FitnessAll,FitnessGbest,Gbest]=FnGetFitnessNbestIndividuIPSODL(X_or_P_or_G)

pop_size=size(X_or_P_or_G,1);
Best_Partikel=zeros(1,size(X_or_P_or_G,2));
FitnessAll=zeros(pop_size,1);
for i=1:pop_size
%    FitnessAll(i)=FnMySVM(X_or_P_or_G(i,:));
    %FitnessAll(i)=1/FnArsitekPSODLCNNeLM(X_or_P_or_G(i,:));
	 FitnessAll(i)=FnArsitekPSODLCNNeLM(X_or_P_or_G(i,:));
    
end

% mengambil X_or_P_or_G yang fitness-nya terbaik
[FitnessGbest,index_Gbest]=max(FitnessAll);
Gbest=X_or_P_or_G(index_Gbest,:);
