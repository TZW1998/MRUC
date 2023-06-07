% load corrupted image
load('perm_miss_image.mat')
load('image.mat')
nnn=180;
figure('Renderer', 'painters', 'Position', [10 10 800 200])
subplot(151)
imshow(image)
title('Original')
subplot(152)
imshow(perm_miss_image)
title('Corrupted')


%% prepare the initialization
perm_num=6;% the image is cropped in to 6X6 grid
perm_index=[1,2,3,7,8,13,14,15,19,20,21,26,27]; % only grids with index in perm_index are shuffled
sub_matrices={};
sub_known_ind={};
factor_ind=nnn/perm_num;
init_image=zeros(nnn);
lambda=1.2;
for ii=1:perm_num
    for jj=1:perm_num
        sub_matrices{ii+(jj-1)*perm_num}=perm_miss_image(((ii-1)*factor_ind+1):(ii*factor_ind),((jj-1)*factor_ind+1):(jj*factor_ind));
        sub_known_ind{ii+(jj-1)*perm_num}=(sub_matrices{ii+(jj-1)*perm_num}~=0);
        nowB=sub_matrices{ii+(jj-1)*perm_num};
        B=nowB;
        B_old=zeros(nnn/perm_num);
        while norm(B_old-B,'fro')>0.0001
            B_old=B;
            B(sub_known_ind{ii+(jj-1)*perm_num})=nowB(sub_known_ind{ii+(jj-1)*perm_num});
            B=ProxNC(B,lambda);
        end
        B(sub_known_ind{ii+(jj-1)*perm_num})=nowB(sub_known_ind{ii+(jj-1)*perm_num});
        init_image(((ii-1)*factor_ind+1):(ii*factor_ind),((jj-1)*factor_ind+1):(jj*factor_ind))=B;
    end
end
[~,init_image]=best_approx(init_image,1);
subplot(153)
imshow(init_image)
title('Initial')

% algorithm set up
max_in_iter=10000;
eps_init=100;
max_out_iter=200000;
verbose=false;
use_rank=4;
lr1=0.2;
lr2=1;

%% run Baseline
[base_recover,~,~]=baseline_recover(max_out_iter,verbose,use_rank,init_image,sub_matrices,lr1,lr2,perm_num,factor_ind,perm_index);
subplot(154)
imshow(base_recover);
title('Baseline')
norm(base_recover-image,'fro')/norm(image,'fro')

%% run Mcubic
[best_recover,~,~]=McubicO_recover(max_in_iter,eps_init,max_out_iter,verbose,use_rank,init_image,sub_matrices,lr1,lr2,perm_num,factor_ind,perm_index);
subplot(155)
imshow(best_recover);
title('McubicO')
norm(best_recover-image,'fro')/norm(image,'fro')

%% utils
function mat=ProxNC(B,lambda)
    [U,S,V] = svd(B);
    ind = 1;
    sigv=diag(S);
    n=length(sigv(sigv>lambda));
    mat=U(:,1:n)*S(1:n,1:n)*V(:,1:n)';
end

function [lambda,mat]=best_approx(B,ranks)
    [U,S,V] = svd(B);
    mat=U(:,1:ranks)*S(1:ranks,1:ranks)*V(:,1:ranks)';
    lambda=S(ranks,ranks);
end

function C=compute_cost(sub_recovers,sub_observes,perm_index)
    perm_n=length(perm_index);
    C=zeros(perm_n);
    for ii=1:perm_n
        now_recover=sub_recovers{perm_index(ii)};
        for jj=1:perm_n
            now_obs=sub_observes{perm_index(jj)};
            obs_ind=(now_obs>0);
            C(ii,jj)=norm(now_recover(obs_ind)-now_obs(obs_ind),'fro')^2;
        end
    end 
end

function softP=Sinkhorn(C,eps,max_iter)
    [n,m]=size(C);
    f=zeros(n,1);
    g=zeros(m,1);
    one_vector_f=ones(m,1);
    one_vector_g=ones(n,1);
    for kk=1:max_iter
        f=Softmin(C-f*one_vector_f'-one_vector_g*g',eps)+f;
        g=Softmin(C-f*one_vector_f'-one_vector_g*g',eps,2)+g;
        if mod(kk,10)==0
            softP=exp((f*one_vector_f'+one_vector_g*g'-C)/eps);
            if norm(sum(softP,2)-1)<0.01
                break
            end
        end
    end
end

function value=Softmin(M,eps,axis)
    if nargin<=2
        axis=1;
    end
    if axis==1
        rmin=min(M,[],2);
        value=rmin-eps*log(sum(exp((rmin-M)/eps),2));
    else
        rmin=min(M);
        value=rmin-eps*log(sum(exp((rmin-M)/eps)));
        value=value';
    end
end

function [best_recover,softP,now_direction]=McubicO_recover(max_in_iter,eps_init,max_out_iter,verbose,use_rank,init_image,sub_observes,lr1,lr2,perm_num,factor_ind,perm_index)
    recover_image=init_image;
    eps=eps_init;
    eps_counts=0;
    fix_eps=false;
    best_obj=9999;
    best_cert=1;
    best_recover=recover_image;
    length_perm=length(perm_index);
    sub_recovers={};
    lambda=1.5;
    now_direction=zeros(size(init_image));
    for iter=1:max_out_iter
        eps_counts=eps_counts+1;
        recover_image_old=recover_image;

        for ii=1:length_perm
           ind1=mod(perm_index(ii)-1,perm_num)+1;
           ind2=ceil(perm_index(ii)/perm_num);
           sub_recovers{perm_index(ii)}=recover_image(((ind1-1)*factor_ind+1):(ind1*factor_ind),((ind2-1)*factor_ind+1):(ind2*factor_ind));
        end

        now_norm=sum(svd(recover_image));
        cost=compute_cost(sub_recovers,sub_observes,perm_index);
        softP=Sinkhorn(cost,eps,max_in_iter);
        now_obj=sum(cost.*softP,'all')+lambda*now_norm;

        for ii=1:(perm_num*perm_num)
           ind1=mod(ii-1,perm_num)+1;
           ind2=ceil(ii/perm_num);
           now_recover=recover_image(((ind1-1)*factor_ind+1):(ind1*factor_ind),((ind2-1)*factor_ind+1):(ind2*factor_ind));
           if ismember(ii,perm_index)
               direction=zeros(factor_ind);
               for jj=1:length_perm
                   direction=direction+softP(find(perm_index==ii),jj)*sub_observes{perm_index(jj)};
               end
               obs_ind=(direction~=0);
               now_recover(obs_ind)=(1-lr1)*now_recover(obs_ind)+lr1*direction(obs_ind);
               now_direction(((ind1-1)*factor_ind+1):(ind1*factor_ind),((ind2-1)*factor_ind+1):(ind2*factor_ind))=direction;
           else
               now_observes=sub_observes{ii};
               now_obs_ind=(now_observes~=0);
               %now_recover(now_obs_ind)=now_observes(now_obs_ind);
               now_recover(now_obs_ind)=(1-lr2)*now_recover(now_obs_ind)+lr2*now_observes(now_obs_ind);
               now_direction(((ind1-1)*factor_ind+1):(ind1*factor_ind),((ind2-1)*factor_ind+1):(ind2*factor_ind))=now_observes;
           end 
           recover_image(((ind1-1)*factor_ind+1):(ind1*factor_ind),((ind2-1)*factor_ind+1):(ind2*factor_ind))=now_recover;
        end
        [lambda,recover_image]=best_approx(recover_image,use_rank);

        if now_obj<(0.9999*best_obj)
            best_obj=now_obj;
            eps_counts=0;
            best_recover=recover_image;
        end
        
        cert=mean(abs(max(softP,[],2)-1));
        if cert<0.01
            [~,recover_perm]=max(softP,[],2);
            while norm(recover_image_old-recover_image,'fro')>0.001
                recover_image_old=recover_image;
                [~,recover_image]=best_approx(recover_image,15);
                for ii=1:(perm_num*perm_num)
                   ind1=mod(ii-1,perm_num)+1;
                   ind2=ceil(ii/perm_num);
                   now_recover=recover_image(((ind1-1)*factor_ind+1):(ind1*factor_ind),((ind2-1)*factor_ind+1):(ind2*factor_ind));
                   if ismember(ii,perm_index)
                       now_observes=sub_observes{perm_index(recover_perm(find(perm_index==ii)))};
                   else
                       now_observes=sub_observes{ii};

                   end 
                   now_obs_ind=(now_observes~=0);
                   now_recover(now_obs_ind)=now_observes(now_obs_ind);
                   now_direction(((ind1-1)*factor_ind+1):(ind1*factor_ind),((ind2-1)*factor_ind+1):(ind2*factor_ind))=now_observes;
                   recover_image(((ind1-1)*factor_ind+1):(ind1*factor_ind),((ind2-1)*factor_ind+1):(ind2*factor_ind))=now_recover;
                end
            end
            best_recover=recover_image;
            break
        end

        if cert<(0.9999*best_cert)
            best_cert=cert;
            eps_counts=0;
            best_recover=recover_image;
        end

        if (eps_counts>100) && (~fix_eps) && (eps>0.0001)
            eps=eps/2;
            eps_counts=0;
            recover_image=best_recover;
        end

        
        if verbose
            fprintf('%dth iteration,norm:%.5f,obj:%.5f,cert:%.5f,eps:%.5f \n', ... 
                iter,now_norm,now_obj,eps);
        end
    end
end

function [best_recover,softP,now_direction]=baseline_recover(max_out_iter,verbose,use_rank,init_image,sub_observes,lr1,lr2,perm_num,factor_ind,perm_index)
    recover_image=init_image;
    counts=0;
    fix_eps=false;
    best_obj=9999;
    best_recover=recover_image;
    length_perm=length(perm_index);
    sub_recovers={};
    lambda=1.5;
    now_direction=zeros(size(init_image));
    softP=zeros(length(perm_index));
    for iter=1:max_out_iter
        softP_old=softP;
        counts=counts+1;
        recover_image_old=recover_image;

        for ii=1:length_perm
           ind1=mod(perm_index(ii)-1,perm_num)+1;
           ind2=ceil(perm_index(ii)/perm_num);
           sub_recovers{perm_index(ii)}=recover_image(((ind1-1)*factor_ind+1):(ind1*factor_ind),((ind2-1)*factor_ind+1):(ind2*factor_ind));
        end

        now_norm=sum(svd(recover_image));
        cost=compute_cost(sub_recovers,sub_observes,perm_index);
        softP=munkres(-cost);
        now_obj=sum(cost.*softP,'all')+lambda*now_norm;


        for ii=1:(perm_num*perm_num)
           ind1=mod(ii-1,perm_num)+1;
           ind2=ceil(ii/perm_num);
           now_recover=recover_image(((ind1-1)*factor_ind+1):(ind1*factor_ind),((ind2-1)*factor_ind+1):(ind2*factor_ind));
           if ismember(ii,perm_index)
               direction=sub_observes{perm_index(softP(find(perm_index==ii)))};
               obs_ind=(direction~=0);
               now_recover(obs_ind)=(1-lr1)*now_recover(obs_ind)+lr1*direction(obs_ind);
               now_direction(((ind1-1)*factor_ind+1):(ind1*factor_ind),((ind2-1)*factor_ind+1):(ind2*factor_ind))=direction;
           else
               now_observes=sub_observes{ii};
               now_obs_ind=(now_observes~=0);
               %now_recover(now_obs_ind)=now_observes(now_obs_ind);
               now_recover(now_obs_ind)=(1-lr2)*now_recover(now_obs_ind)+lr2*now_observes(now_obs_ind);
               now_direction(((ind1-1)*factor_ind+1):(ind1*factor_ind),((ind2-1)*factor_ind+1):(ind2*factor_ind))=now_observes;
           end 
           recover_image(((ind1-1)*factor_ind+1):(ind1*factor_ind),((ind2-1)*factor_ind+1):(ind2*factor_ind))=now_recover;
        end
        [lambda,recover_image]=best_approx(recover_image,use_rank);


        if (counts>30) || (iter>1000)
            recover_perm=softP;
            while norm(recover_image_old-recover_image,'fro')>0.001
                recover_image_old=recover_image;
                [~,recover_image]=best_approx(recover_image,15);
                for ii=1:(perm_num*perm_num)
                   ind1=mod(ii-1,perm_num)+1;
                   ind2=ceil(ii/perm_num);
                   now_recover=recover_image(((ind1-1)*factor_ind+1):(ind1*factor_ind),((ind2-1)*factor_ind+1):(ind2*factor_ind));
                   if ismember(ii,perm_index)
                       now_observes=sub_observes{perm_index(recover_perm(find(perm_index==ii)))};
                   else
                       now_observes=sub_observes{ii};

                   end 
                   now_obs_ind=(now_observes~=0);
                   now_recover(now_obs_ind)=now_observes(now_obs_ind);
                   now_direction(((ind1-1)*factor_ind+1):(ind1*factor_ind),((ind2-1)*factor_ind+1):(ind2*factor_ind))=now_observes;
                   recover_image(((ind1-1)*factor_ind+1):(ind1*factor_ind),((ind2-1)*factor_ind+1):(ind2*factor_ind))=now_recover;
                end
            end
            best_recover=recover_image;
            break
        end

        if norm(softP_old-softP,'fro')>0
            counts=0;
        end
        
        if verbose
            fprintf('%dth iteration,norm:%.5f,obj:%.5f,cert:%.5f\n', ... 
                iter,now_norm,now_obj);
        end
    end
end