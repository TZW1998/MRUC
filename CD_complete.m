function [best_P,best_B,final_cert,history,result]=CD_complete(B1,Bn,A1,An,A1_row_ind,An_row_ind,lambda_list, ... 
    eps_init,eps_decay,verbose,max_out_iter,max_in_iter,test_ind,test_label,tol,alpha,eps_min,perm_label,default_lr,gradient_step)
    final_cert=1;
    fix=false;
    n=size(B1,1);
    one_vector=ones(n,1); 
    parts=length(Bn);
    Bn_gradient = cell(parts,1);
    softP = cell(parts,1);
    softP_old = cell(parts,1);
    PP = cell(parts,1);
    lr = zeros(parts,1);
    B1_num_columns=size(B1,2);
    Bn_num_columns=zeros(parts,1);
    offset=B1_num_columns;
    Bn_ind=cell(parts,1);
    for pp=1:parts
        Bn_num_columns(pp)=size(Bn{pp},2);
        Bn_gradient{pp} = zeros(n,Bn_num_columns(pp));
        lr(pp)=0;
        softP{pp}=zeros(n,n);  
        Bn_ind{pp}=[offset+1,offset+Bn_num_columns(pp)];
        offset=offset+Bn_num_columns(pp);
    end
    
    history=zeros(6,0);
    cert=zeros(parts,1);
    diff=zeros(parts,1);
    result=[99999999999,999999999999,9999999999]';
    counts=0;
    eps_counts=0;
    best_B1=B1;
    best_Bn=Bn;
    out_offset=0;
    out=0;
    
    for lambda=lambda_list
        eps=eps_init;
        eps_init=eps_init*eps_decay;
        best_cert=1;
        B1=best_B1;
        Bn=best_Bn;
        out_offset=out_offset+out;     
        for out=1:max_out_iter

            counts=counts+1;
            eps_counts=eps_counts+1;
            for ii=1:n
               B1(ii,A1_row_ind{ii})=A1{ii};
            end
            for gg=1:gradient_step
                B=B1;
                for pp=1:parts 
                    Bn{pp}=Bn{pp}-lr(pp)*Bn_gradient{pp};
                    B=[B,Bn{pp}];
                end
                B=ProxNC(B,lambda);
                B1=B(:,1:B1_num_columns);
                for pp=1:parts
                    indx=Bn_ind{pp};
                    nowB=B(:,indx(1):indx(2));
                    Bn{pp}=nowB;
                end
            end
            rvs=0;
            for pp=1:parts
                indx=Bn_ind{pp};
                nowB=B(:,indx(1):indx(2));
%                 Bn{pp}=nowB;
                C=zeros(n);
                for ii=1:n
                    right_diff_sum=zeros(1,n);
                    for jj=1:n
                        right_diff_sum(jj)=sum((An{jj,pp}-nowB(ii,An_row_ind{jj,pp})).^2);
                    end
                    C(ii,:)=right_diff_sum;
                end
                % inner loop
                if ~fix
                    f=zeros(n,1);
                    g=zeros(n,1);
                    softP_old{pp}=softP{pp};
                    for kk=1:max_in_iter
                        f=Softmin(C-f*one_vector'-one_vector*g',eps)+f;
                        g=Softmin(C-f*one_vector'-one_vector*g',eps,2)+g;
                        if mod(kk,10)==0
                            softPP=exp((f*one_vector'+one_vector*g'-C)/eps);
                            if norm(sum(softPP,2)-1)<0.01
                                break
                            end
                        end
                    end
                    softP{pp}=softPP;
                    P=zeros(n);
                    [~, argmax] = max(softP{pp},[],2);
                    for ii=1:n
                        P(ii,argmax(ii))=1;
                    end
                    PP{pp}=P;
                end
                Bgradient = zeros(n,Bn_num_columns(pp));
                for ii=1:n
                    for jj=1:n            
                        Bgradient(ii,An_row_ind{jj,pp})=Bgradient(ii,An_row_ind{jj,pp})+softPP(ii,jj)*(nowB(ii,An_row_ind{jj,pp})-An{jj,pp});
                    end
                end
                Bn_gradient{pp}=Bgradient;
                rvs=rvs+sum(sum(C.*softPP));
                cert(pp)=mean(abs(max(softPP,[],2)-1));
                diff(pp)=max(abs(softPP-softP_old{pp}),[],'all');
                if default_lr~=0
                    lr(pp)=default_lr;
                else
                    lr(pp) = min(max((0.5*(1-cert(pp))^alpha+0.01)*(1-diff(pp)),0.01),0.5);
                end
            end
            nnorm=sum(svd(B));
            rvs_total=rvs+lambda*nnorm;
            test_err=zeros(parts+1,1);
            test_length=zeros(parts+1,1);
            test_err(1)=norm(B1(test_ind{1})-test_label{1})^2; 
            test_length(1)=length(test_ind{1});
            perm_err=0;
            for testii=2:length(test_ind)
                nowB=Bn{testii-1};
                test_err(testii)=norm(nowB(test_ind{testii})-test_label{testii}).^2;
                test_length(testii)=length(test_ind{testii});
                perm_err=perm_err+norm(PP{testii-1}-perm_label{testii-1},'fro')^2/2;
            end
            total_test_err=sqrt(sum(test_err./test_length));
            history(:,end+1)=[rvs_total;total_test_err;mean(lr);lambda;eps;perm_err];
            
            if  mean(cert)<(0.9999*best_cert) 
                best_cert=mean(cert);
                counts=0;
                eps_counts=0;
            end
            if (rvs_total<=(0.9999*result(2))) % || (total_test_err<=(0.9999*result(3))) 
                result(1)=rvs;
                result(2)=rvs_total;
                result(3)=total_test_err;
                best_B=B;
                best_B1=B1;
                best_Bn=Bn;
                best_P=PP;
                best_softP=softP;
                final_cert=mean(cert);
                counts=0;
                eps_counts=0;
            end
            
            if verbose
                fprintf('%dth iteration: Outer loop: (%.5f,%.5f,%.5f,%.5f,%.1f), cert:%.5f, diff:%.5f, lr:%.5f, eps:%.5f, lbd:%.5f,rank:%d\n',out+out_offset,rvs,total_test_err, ... 
                    nnorm,rvs_total,perm_err,mean(cert),mean(diff),mean(lr),eps,lambda,rank(B));
                disp(sqrt(test_err./test_length)');
            end
            if (counts>(5*tol)) %|| ((cert < 0.001) || (eps<0.000001))
                break
            end
            if eps<eps_min
                break
            end
            if eps_counts>tol
                eps_counts=0;
                eps=eps/2;
            end
        end
    end
end

function mat=ProxNC(B,lambda)
    [U,S,V] = svd(B);
    sigv=diag(S);
    n=length(sigv(sigv>lambda));
    mat=U(:,1:n)*S(1:n,1:n)*V(:,1:n)';
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