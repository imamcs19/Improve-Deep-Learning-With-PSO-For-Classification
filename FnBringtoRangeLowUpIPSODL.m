function [XV]=FnBringtoRangeLowUpIPSODL(XV,XV_lower, XV_upper)

% dioperasikan untuk menandai yang nilainya kurang dari batas lower
XV_minus_XV_lower=XV-XV_lower;

% dioperasikan untuk menandai yang nilainya lebih dari batas upper
XV_minus_XV_upper=XV-XV_upper;

% mendapatkan index yang nilainya kurang dari batas bawah
idx_KurangDariLower=find(XV_minus_XV_lower<0);
if(isempty(idx_KurangDariLower)) 
else
    XV(idx_KurangDariLower)=XV_lower(idx_KurangDariLower);
end

% mendapatkan index yang nilainya lebih dari batas atas
idx_LebihDariUpper=find(XV_minus_XV_upper>0);
if(isempty(idx_LebihDariUpper)) 
else
    XV(idx_LebihDariUpper)=XV_upper(idx_LebihDariUpper);
end