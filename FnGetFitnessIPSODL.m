function [FitnessAwal,IndexSortingDesc]=FnGetFitnessIPSODL(X_or_P_or_G)

pop_size=size(X_or_P_or_G,1);
Fitness=zeros(pop_size,1);
for i=1:pop_size
    %Fitness(i)=1/FnArsitekPSODLCNNeLM(X_or_P_or_G(i,:));
	Fitness(i)=FnArsitekPSODLCNNeLM(X_or_P_or_G(i,:));
end

% mencari indexsorting secara descending
for i=1:pop_size
    IndexAwal(i) = i;
end

FitnessAwal= Fitness;



for i=1:pop_size
   TempNilaiFx = Fitness(i);
   TempIndexAwal = IndexAwal(i);
   for j=i+1:pop_size
       %Fitness(j)
        if (Fitness(j) > TempNilaiFx)
            TempNilaiFx = Fitness(j);
            Fitness(j) = Fitness(i);
            Fitness(i) = TempNilaiFx;
            TempIndexAwal = IndexAwal(j);
            IndexAwal(j) = IndexAwal(i);
            IndexAwal(i) = TempIndexAwal;
        end
   end
end

%[FitnessAwal Fitness];
%IndexAwal;

% pause

IndexSortingDesc=IndexAwal;