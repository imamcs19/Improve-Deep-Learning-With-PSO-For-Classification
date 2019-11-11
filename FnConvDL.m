function [Result]=FnConvDL(norm,bykData,k)

%% melakukan convolution dengan average filter, max filter,
% std filter secara iteratif
for i=1:bykData
    d1_conv11AvgFilter{i}=Function_AvgFilter(norm{i},k);
    
    
    
    d1_conv12MaxFilter{i}=Function_MaxFilter_(norm{i},k);
    
    d1_conv13STDFilter{i}=Function_STDFilter(norm{i},k);
    
%     norm{1}
%     d1_conv13STDFilter{1}
%     pause(500000000)
    
    
%     Filter1{i}=Function_AvgFilter(norm{i},k);
%     Filter2{i}=Function_MaxFilter_(norm{i},k);
%     Filter3{i}=Function_STDFilter(norm{i},k);
end
%%

Result{1} = d1_conv11AvgFilter;
Result{2} = d1_conv12MaxFilter;
Result{3} = d1_conv13STDFilter;

% for i=1:bykFilter
%     Result{i}=strcat('Filter',num2str(i));
% % Result{1} = d1_conv11AvgFilter
% % Result{2} = d1_conv12MaxFilter
% % Result{3} = d1_conv13STDFilter
% end
