function [varargout] = RFuzzyEn2LapRank_modified(H,p,m,lambda)
[N,k,M] = size(H); beta = lambda; 
F = mean(H,3); Sigma = repmat(1/sqrt(k)*eye(k),[1,1,M]); Alpha = repmat(1/M,[1,1,M]);
Itermax = 50; iter = 1; fk = 1; fk1=1; evs = zeros(k+1,Itermax); U = F;
%%
while ((fk>1e-11 || fk1<1e-10) && iter<Itermax)
    %%
     A =  F*F';
     V = A - (beta/4*lambda)*pdist2(U,U).^2;
    for i = 1:N
        ppp = V(i,:);
        ppp(i) = [];
        sss = MY_NSS(ppp);
        sss = VecInsert(sss,i,0);
        S(i,:) = sss;
%         S(i,:) = MY_NSS(V(i,:));
    end
    %%
    SVDF = zeros(N,N);
    for v = 1:M
        q(:,:,v) = norm(F*F' - H(:,:,v)*Sigma(:,:,v)*H(:,:,v)','fro').^(p-2);
        w(:,:,v) = Alpha(:,:,v).^m*q(:,:,v);
        SVDF = SVDF + w(:,:,v)* H(:,:,v)*Sigma(:,:,v)*H(:,:,v)';
    end
    [F,~,~] = svds(SVDF+lambda*S,k);
    for v = 1:M
        G(:,:,v) = w(:,:,v)*H(:,:,v)'*F*F'*H(:,:,v);
        g = diag(G(:,:,v));
        Sigma(:,:,v) = diag(g./sqrt(sum(g.^2)));
        D(:,:,v) = norm(F*F' - H(:,:,v)*Sigma(:,:,v)*H(:,:,v)','fro');
    end
    d = reshape(D,1,M);
    alpha = bsxfun(@rdivide,d.^(-p/(m-1)),sum(d.^(-p/(m-1)),2));
    Alpha = reshape(alpha,[1,1,M]);
%%    
    Sn = (S+S')/2;
    LSn = diag(sum(Sn,2)) - Sn;
    [Efull,Evfull] = eig(LSn);
    U = Efull(:,1:k);
    Evfull = diag(Evfull);
    evs(:,iter) = Evfull(1:k+1);
    if  sum(evs(1:k+1,1)) < 1e-10
        error('The original graph has more than %d connected component', k);
    end
    fk = sum(evs(1:k,iter));
    fk1 = sum(evs(1:k+1,iter));
    % updaing lambda
    if fk > 1e-11
        beta = 2*beta;
    elseif fk1 < 1e-10
        beta = beta/2;
    else
        break;
    end
    iter = iter + 1;
end
evs = evs(:,1:iter);
[clusternum, label]=graphconncomp(sparse(Sn)); label = label';
if clusternum ~= k
    sprintf('Can not find the correct cluster number: %d', k)
end
varargout{1} = label;
varargout{2} = F;
varargout{3} = S;
varargout{4} = evs;
varargout{5} = Alpha;
varargout{6} = Sigma;
function [x] = MY_NSS(v)
%% solve 
%  min  1/2 || x - v||^2
%  s.t. x>=0, 1'x=1
%  Transform it into f(lambda_m) = 1/n*sum_{i=1}^n max(lambada_m - u_j,0) -
%  lambda_m = 0; x_j = max(u - lambda_m, 0);
%  if umin > 0, lambda_m = 0; x = u;
%  else : Newton Method.
%%
n = length(v);
u = v-mean(v) + 1/n;
umin = min(u);
if umin >= 0
    x = u;
else
    f = 1;
    iter = 1;
    lambda_m = 0;
    while abs(f) > 10^-10
        p = lambda_m - u;
        k  = p>0;
        g = sum(k)/n - 1;
        f = sum(p(k))/n - lambda_m;
        lambda_m = lambda_m - f/g;
        iter = iter + 1;
        if iter>100
            break;
        end
    end
    x = max(-p,0);
end
function data=VecInsert(mat,ind,num)
n=length(mat);
data(ind)=num;
data(1:ind-1)=mat(1:ind-1);
data(ind+1:n+1)=mat(ind:n);