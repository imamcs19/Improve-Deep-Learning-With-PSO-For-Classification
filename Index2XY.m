function [Vsx,Vsy] = Index2XY(Vs,baris) 

Vsx=mod(Vs,baris)';
VsxIndex=find(Vsx==0);
Vsx(VsxIndex)=Vsx(VsxIndex)+baris;

Vsy=ceil(Vs/baris)';













