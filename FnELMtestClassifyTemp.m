function [Result,kelas_prediksi, Ytest_predict]=...
    FnELMtestClassify(hP,target,...
    W,Bias,Beta_topi,bykData,bykFilter)

byk_dim_data_input=bykFilter*numel(hP{1}{1}(:)');
Xtest=zeros(bykData,byk_dim_data_input);

% convert kelas target ke vektor
byk_kelas=size(Beta_topi,2);
Ytest=-ones(bykData,byk_kelas);
for i=1:bykData
    Ytest(i,target(i))=1;
end

% get data training dari hasil pooling
% yang sudah diubah dalam bentuk vektor
for i=1:bykData
    hP_init=[];
    for j=1:bykFilter
        hP_init=[hP_init hP{j}{i}(:)'];
    end
    %hP_init
    Xtest(i,:)=hP_init;
end

% digits(4)
% hP{1}{1}
% hP{2}{1}
% hP{3}{1}
% Xtest(1,:)
% Xtest
% pause(50000000)
W = [0.0304   -0.0287   -0.0985   -0.0100   -0.0613    0.1180   -0.0674   -0.2246    0.0408    0.1016   -0.0919    0.0467
    0.3433   -0.0395   -0.1786   -0.3792    0.2078    0.0021   -0.0730    0.2434   -0.3307    0.1074    0.0976   -0.3729
   -0.2847    0.4425   -0.1212   -0.3601    0.0296   -0.0146   -0.4707   -0.3832    0.4301   -0.0311   -0.0743   -0.4185
    0.2167    0.1283    0.2190    0.4472    0.1524   -0.3899    0.0104    0.2693   -0.2918    0.0963   -0.3782   -0.2105
    0.3627    0.3064   -0.3069    0.1309    0.2782    0.4165    0.4637    0.2657   -0.4802    0.2235   -0.1500    0.1768]
Bias = [0.6388    0.0110    0.9773    0.3855    0.3997]
Beta_topi = [5.4406e+05    -5.2334e+06    4.6893e+06
2.0666e+05    -1.6559e+06    1.4492e+06
-4.3485e+05    4.1289e+06    -3.6941e+06
68473              -6.5347e+05    5.85e+05
-2.8728e+05    2.6e+06          -2.3127e+06]


    [vMaxa,idxMaxa]=max(Ytest');
    kelas_aktual=idxMaxa';
    
    byk_data_test=size(Xtest,1);
    %tempH_test=InputWeight*Xtest';
    tempH_test=Xtest*W'
    
    %ind=ones(1,byk_data_test);
    %BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
    % atau cara lain
    BiasofHiddenNeurons=Bias;
    BiasMatrix=(ones(byk_data_test,1))*BiasofHiddenNeurons; %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
    tempH_test=tempH_test + BiasMatrix;
    
    
    %H_test = 1 ./ (1 + exp(-tempH_test))
    H_test =Fn_Aktivasi(tempH_test);
    
    
    %   TY: output of the testing data (Y_test_predict)
    Ytest_predict=(H_test * Beta_topi);
    [vMax,idxMax]=max(Ytest_predict');
    kelas_prediksi=idxMax';
    
    % [kelas_aktual kelas_prediksi]
    nBenar=numel(find(kelas_aktual-kelas_prediksi==0));
    akurasi=(nBenar/byk_data_test)*100;
    Result=akurasi;
    
    %pause(50000000)
    
    
    