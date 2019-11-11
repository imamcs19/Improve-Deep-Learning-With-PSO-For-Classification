function [Result]=FnConvInDL(hA,bykData,k,bykFilter)

%% melakukan convolution dengan average filter, max filter,
% std filter secara iteratif
for i=1:bykFilter
    for j=1:bykData
        if(i==1)
        Result{i}{j}=Function_AvgFilter(hA{i}{j},k);
        end
        if(i==2)
        Result{i}{j}=Function_MaxFilter_(hA{i}{j},k);
        end
        if(i==3)
        Result{i}{j}=Function_STDFilter(hA{i}{j},k);
        end
    end
end
%%

% Result{1} = d1_conv11AvgFilter
% Result{2} = d1_conv12MaxFilter
% Result{3} = d1_conv13STDFilter

