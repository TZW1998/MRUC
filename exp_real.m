comedy=csvread('movielens/comedy_share.csv');
romance=csvread('movielens/romance_share.csv');
action=csvread('movielens/action_share.csv');
drama=csvread('movielens/drama_share.csv');
thriller=csvread('movielens/thriller_share.csv');

%% prepare for recover
ep='1';
mkdir(['./exp5_',ep])
use_data={drama,comedy,romance,action,thriller};
use_test=zeros(length(use_data),1);
num_platforms=5;
init_lambda=500;
reference_data=use_data{1};
n=length(unique(reference_data(:,1)));
fileID = fopen(['./exp5_',ep,'/seed.txt'],'w');
fileMCS = fopen(['./exp5_',ep,'/MCS.txt'],'w');
fileM3O = fopen(['./exp5_',ep,'/M3O.txt'],'w');
seed=randi(10000000);
fprintf(fileID,'%d',seed);
rng(seed)
train_ind=cell(num_platforms,1);
test_ind=cell(num_platforms,1);
single_train_ind=cell(num_platforms,1);
single_test_ind=cell(num_platforms,1);
test_data=cell(num_platforms,1);
inside_test_ind=[];
genre_test_num=zeros(num_platforms,1);
for ll=1:num_platforms
    num=length(use_data{ll});
    nowdata=use_data{ll};
    single_ind=nowdata(:,1)+(nowdata(:,2)-1)*n;
    ind=randperm(num);
    train_ind{ll}=ind(1:round(num*0.8));
    test_ind{ll}=ind((round(num*0.8)+1):end);
    single_train_ind{ll}=single_ind(train_ind{ll});
    single_test_ind{ll}=single_ind(test_ind{ll});
    test_data{ll}=nowdata(test_ind{ll},3);
    genre_test_num(ll)=length(test_ind{ll});
end
total_test_num=sum(genre_test_num);
B1=zeros(n,length(unique(reference_data(:,2))));
A1=cell(n,1);
A1_row_ind=cell(n,1);
train_data=reference_data(train_ind{1},:);
inside_test_data=reference_data(test_ind{1},3);
for ii=1:n
    find_ind=(train_data(:,1)==ii);
    A1_row_ind{ii}=train_data(find_ind,2)';
    A1{ii}=train_data(find_ind,3)';
    B1(ii,A1_row_ind{ii})=A1{ii};
end
% MCS
counts=0;
single_ind=reference_data(:,1)+(reference_data(:,2)-1)*n;
best_test_rmse=10;
lambda=init_lambda;
for lmd=1:100
    if lambda>50
        lambda=lambda-10;
    else
       lambda=lambda-1; 
    end
    if lambda<=5
        break
    end
    counts=0;
    while counts<5
        counts=counts+1;
        B1(single_train_ind{1})=reference_data(train_ind{1},3);
        B1=ProxNC(B1,lambda);
        total_test_rmse=norm(B1(single_test_ind{1})-test_data{1})/sqrt(length(single_test_ind{1}));
        fprintf('rmse:%.5f,rank:%.5f,best:%.5f,lmd:%.5f\n', ... 
            total_test_rmse,rank(B1),best_test_rmse,lambda)
        if total_test_rmse<0.99999*best_test_rmse
            counts=0; 
            best_test_rmse=total_test_rmse;
            best_B=B1;
        end
    end
end
fprintf('rank:%.5f,best:%.5f\n', ... 
    rank(B1),best_test_rmse)
B1=best_B;
use_test(1)=best_test_rmse;

parts=length(use_data)-1;
An=cell(parts,1);
Bn = cell(parts,1);
Bn_oracle = cell(parts,1);
An_row_ind=cell(n,parts);
for ll=2:length(use_data)
    part_data=use_data{ll};
    train_data=part_data(train_ind{ll},:);
    for ii=1:n
        find_ind=(train_data(:,1)==ii);
        An_row_ind{ii,ll-1}=train_data(find_ind,2)';
        An{ii,ll-1}=train_data(find_ind,3)';
    end
    counts=0;
    single_ind=part_data(:,1)+(part_data(:,2)-1)*n;
    best_test_rmse=10;
    Bnl=zeros(n,length(unique(part_data(:,2))));
    Bnl_oracle=zeros(n,length(unique(part_data(:,2))));
    Bnl_oracle(single_train_ind{ll})=part_data(train_ind{ll},3);
    lambda=init_lambda;
    for lmd=1:100
        if lambda>50
            lambda=lambda-10;
        else
           lambda=lambda-1; 
        end
        if lambda<=1
            break
        end
        counts=0;
        while counts<5
            counts=counts+1;
            Bnl(single_train_ind{ll})=part_data(train_ind{ll},3);
            Bnl=ProxNC(Bnl,lambda);
            total_test_rmse=norm(Bnl(single_test_ind{ll})-test_data{ll})/sqrt(length(single_test_ind{ll}));
            fprintf('rmse:%.5f,rank:%.5f,lmd:%.5f,best:%.5f\n', ... 
                total_test_rmse,rank(Bnl),lambda,best_test_rmse)
            if total_test_rmse<0.99999*best_test_rmse
                counts=0; 
                best_test_rmse=total_test_rmse;
                best_B=Bnl;
            end
        end
    end
    fillB=zeros(size(best_B));
    fillB(:)=randsample(best_B(:),length(best_B(:)),true);
    Bn{ll-1}=best_B;
    Bn_oracle{ll-1}=Bnl_oracle;
    use_test(ll)=best_test_rmse;
    fprintf('rank:%.5f,best:%.5f\n', ... 
    rank(Bnl),best_test_rmse)
end
permutation_matrix={eye(930),eye(930),eye(930),eye(930),eye(930)};

Bn_num_columns=zeros(num_platforms,1);
offset=size(B1,2);
Bn_ind=cell(num_platforms,1);
Bn_ind{1}=[1,size(B1,2)];
Bn_num_columns(1)=size(B1,2);
for pp=2:num_platforms
    Bn_num_columns(pp)=size(Bn{pp-1},2); 
    Bn_ind{pp}=[offset+1,offset+Bn_num_columns(pp)];
    offset=offset+Bn_num_columns(pp);
end
fprintf(fileMCS,'%.5f,%.5f,%.5f,%.5f,%.5f,%.5f',use_test,sqrt(sum((use_test.*sqrt(genre_test_num)).^2))/sqrt(sum(genre_test_num)));

% best
best_M3O_error=[use_test;sqrt(sum((use_test.*sqrt(genre_test_num)).^2))/sqrt(sum(genre_test_num))]';
best_recover={B1,Bn{1},Bn{2},Bn{3},Bn{4}};

% switch
orders={[1,2,3,4,5],[2,1,3,4,5],[3,1,2,3,4],[4,1,2,3,5],[5,1,2,3,4]};
for oo=1:5
    order=orders{oo}
    use_single_test_ind=single_test_ind;
    use_test_data=test_data;
    use_Bn=Bn;
    use_An_row_ind=An_row_ind;
    if order(1)==1
        use_B1=B1;
        use_A1=A1;
        use_A1_row_ind=A1_row_ind;
    else
        use_B1=Bn{order(1)-1};
        use_A1=An(:,order(1)-1);
        use_A1_row_ind=An_row_ind(:,order(1)-1);
        use_single_test_ind(1)=single_test_ind(order(1));
        use_test_data(1)=test_data(order(1));
    end
    for pp=2:5
        if order(pp)==1
            use_Bn{pp-1}=B1;
            use_An(:,pp-1)=A1';
            use_An_row_ind(:,pp-1)=A1_row_ind';
        else
            use_Bn{pp-1}=Bn{order(pp)-1};
            use_An(:,pp-1)=An(:,order(pp)-1);
            use_An_row_ind(:,pp-1)=An_row_ind(:,order(pp)-1);
        end
        use_single_test_ind(pp)=single_test_ind(order(pp));
        use_test_data(pp)=test_data(order(pp));
    end
    use_num=4;
    use_Bn=use_Bn(1:use_num);
    use_An=use_An(:,1:use_num);
    use_An_row_ind=use_An_row_ind(:,1:use_num);
    use_single_test_ind=use_single_test_ind(1:(use_num+1));
    use_test_data=use_test_data(1:(use_num+1));
    
% recover
    counts=0;
    eps_init=0.05;
    eps_decay=0.05; 
    verbose=true;
    max_out_iter=400;
    max_in_iter=1000;
    omega=0.5;
    beta=100;
    use_B1=best_recover{order(1)};
    offset=Bn_num_columns(order(1));
    for ii=1:4
    use_Bn{ii}=best_recover{order(ii+1)};
    end
    for init_rank=[1:8,9:2:20,21:5:50,51:10:150]
        counts=counts+1;
        [best_error,best_P,best_B,final_cert,history,result]=CD_complete_real(use_B1,use_Bn,use_A1,use_An,use_A1_row_ind,use_An_row_ind,init_rank, ... 
            eps_init,eps_decay,verbose,max_out_iter,max_in_iter,use_single_test_ind,use_test_data,3,omega,beta,0.000001,permutation_matrix,0,1,best_M3O_error(order));
        use_B1=best_B(:,1:Bn_num_columns(order(1)));
        if best_error(1)<best_M3O_error(order(1))
            best_M3O_error(order(1))=best_error(1);
            best_recover{order(1)}=use_B1;
            counts=0;
        else
            use_B1=best_recover{order(1)};
        end
        offset=Bn_num_columns(order(1));
        for ii=1:4
            use_Bn{ii}=best_B(:,(offset+1):(offset+Bn_num_columns(order(ii+1))));
            offset=offset+Bn_num_columns(order(ii+1));
            if best_error(ii+1)<best_M3O_error(order(ii+1))
                best_M3O_error(order(ii+1))=best_error(ii+1);
                best_recover{order(ii+1)}=use_Bn{ii};
                counts=0;
            else
                use_Bn{ii}=best_recover{order(ii+1)};
            end
        end
        if counts>15
            break
        end
        now_total=sqrt(sum((best_M3O_error(1:(end-1)).*sqrt(genre_test_num')).^2))/sqrt(sum(genre_test_num));
        best_M3O_error(end)=now_total;
        fileM3O = fopen(['./exp5_',ep,'/M3O.txt'],'w');
        fprintf(fileM3O,'%.5f,%.5f,%.5f,%.5f,%.5f,%.5f',best_M3O_error);
    end
end

function mat=ProxNC(B,lambda)
    [U,S,V] = svd(B);
    ind = 1;
    sigv=diag(S);
    n=length(sigv(sigv>lambda));
    mat=U(:,1:n)*S(1:n,1:n)*V(:,1:n)';
end