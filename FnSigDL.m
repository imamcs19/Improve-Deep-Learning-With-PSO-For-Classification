function [Result]=FnSigDL(hC,bykFilter,bykData)

% Result = 1./(1+exp(I))
for i=1:bykFilter
    for j=1:bykData
        Result{i}{j} = 1./(1+exp(-hC{i}{j}));
    end
end
% hC{1}{1}
% Result{1}{1}
% 
% pause(50000000)
