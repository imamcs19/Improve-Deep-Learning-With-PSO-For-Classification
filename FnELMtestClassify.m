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

    [vMaxa,idxMaxa]=max(Ytest');
    kelas_aktual=idxMaxa';
    
    byk_data_test=size(Xtest,1);
    %tempH_test=InputWeight*Xtest';
    tempH_test=Xtest*W';
    
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
    
    
    