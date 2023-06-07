%% Experiment script for the simulated data.
ranks=5; %r
parts=2; %number of participant matrices 
shuffle_columns=60; %total columns of participant matrices
perm_num=100;
missing_percent=60; %Percentage of missing entries
eps_init=0.1;
eps_decay=0.5;
max_out_iter=50000;
max_in_iter=10000;
verbose=false;
lambda_list=[0.55];
snr=0.1;
width=100;
seed=256449302;
[B1,A1,An,A1_row_ind,An_row_ind,oracle_image,permutation_matrix,test_ind,test_label,norm_constant]=generate_matrix(ranks,...
    parts,missing_percent,shuffle_columns,perm_num,width,snr,seed);
results=zeros(2,10);
seeds=zeros(1,10);
rng(seed)
for ii=1:10
    seed=randi(10000000);
    rng(seed)
    seeds(ii)=seed;
    Bn=generate_Bn(parts,100,shuffle_columns/parts,norm_constant,seed);
    [best_P,best_B,final_cert,history,result]=CD_complete(B1,Bn,A1,An,A1_row_ind,An_row_ind,lambda_list, ... 
    eps_init,eps_decay,verbose,max_out_iter,max_in_iter,test_ind,test_label,100,1.5,0.000001,permutation_matrix,0,1);
    results(:,ii)=[history(1,end);history(6,end)];
    fprintf('%d trial, obj:%.5f, Perr:%.5f. seed:%d\n',ii,results(1,ii),results(2,ii),seed)
end
save('exp_result.mat','results')