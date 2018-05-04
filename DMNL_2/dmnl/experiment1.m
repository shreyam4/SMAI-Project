
function out = main
load dt/sydata;
 s = RandStream.create('mt19937ar','seed',1);
        RandStream.setGlobalStream(s);


%maximum iteration number
param.max_iter = 10;
%maximum inner iteration number for W & H
param.inner_iter = 50;
%regularizer parameters for H and W respectively
param.lambda1 = 0.00001;
param.lambdaw = 0.00001;
%the number of potential new labels
param.k = 2;
%ADMM parameter
param.rho = 0.125;
%Other gradient descent parameters
param.step_size = .03125/8;
param.clipnorm = 10000;
param.decayw = 0;
param.decayh = 0;
param.decayhsquare = 1;
param.disc = 0.2;

Bags = X';
Bags_back = Bags;


[X, ~]=AddBoundBag(X,0,20,100);
X_bound_data=X{length(X)};
X_data=X(1:length(X)-1); 

k=200; 
g=randn(k,size(X{1},1)); 
save('g.mat','g');

[X]=PreprocessingX(X,'kernel',[],0.1); 

for i=1:length(Bags)
    z = y{i};
    y{i} = z;
    X{i} = X{i}';
    Bags_back{i} = Bags_back{i}';
end
X{end} = X{end}';

for rr = 1:1;

  

insts = cell2mat(X);
zz = sum(insts.^2,1);

mx = max(insts,[],1);
mn = min(insts,[],1);

Y(Y<0) = 0;


for i= 1:length(X)
   
    Bag = X{i};
    bgsz = size(Bag,1);
   
    Bag = (Bag-repmat(mn,bgsz,1))./(repmat(mx-mn,bgsz,1)+1e-6);

    X{i} = Bag;

end


for i= 1:length(X)-1
    yi = Y(i,:);
    Bag = X{i};
    bgsz = size(Bag,1);
    
    y_i = y{i};
    y{i} = y_i;
    Bags{i} = Bag;
end



train_data = Bags;
train_target = Y(:,1:4);

[W, AW, Anum, H]=dmnl_train(train_target, train_data, param);
Wm = 0;
for i = 1:length(AW)
     Wm = Wm + AW{i};
end
Wm = Wm/length(AW);


insts = X{end};
F = get_g_func(insts*(Wm));
figure;
PlotResultBoundary(F,X_bound_data,19, 1);

end

Fp = get_g_func(cell2mat(Bags)*(Wm));
[~,idx]  = max(Fp,[],2);
gt = cell2mat(y');
[~,idx2]  = max(gt,[],2);
%%instance annotation accuracy.
acc = sum(idx==idx2)/length(idx)

end


function [ W, AW, Anum, H, Beta] = dmnl_train(Y, Bags, param)
  %%training dmnl
   k = param.k;
   mx_rho = param.rho;
   lambdaw = param.lambdaw;
   step_size0 = param.step_size;
   lambda1 = param.lambda1;
   max_iter = param.max_iter;
   inner_iter = param.inner_iter;
   disc = param.disc;
   decayw = param.decayw;
   clipnorm = param.clipnorm;
   decayh = param.decayh;
   squaredecayh = param.decayhsquare;
   
   AW = 0;
   Anum = 0;
   trounds = 0;
   [ H, Beta, W, Alpha, S] = initialization(Y, Bags,param);
   insts = cell2mat(Bags);
   rho = 0;
   for i = 1:max_iter
       i
       %%update W
       for j=1:inner_iter
           [ W, AW, Anum, trounds ] = update_w(W, AW, Anum, Y, Bags, Beta, ...
                H, Alpha, S, rho, lambdaw, step_size0, trounds, k,disc, clipnorm, decayw);
       end
    
       Wm = 0;
       for j = 1:length(AW)
         Wm = Wm + AW{j};
       end
       Wm = Wm/length(AW);
       W = Wm;
       F = get_g_func(insts*W);


       %%update H
       [ H ] = update_h(H, F, S, Alpha, Beta, Y, Bags, insts,  lambda1, rho, k, i, disc, decayh, squaredecayh, inner_iter);

       %%update Beta
       [ Beta ] = update_beta(Bags, H, F, S);

       %%update Alpha
       [ Alpha ] = update_alpha( H, F, S, Alpha);
       
       %%update S
       [S] = update_s(H, F, S);
       
       
       if rho < mx_rho
            rho = rho + mx_rho/max_iter*2;
       else
            rho = mx_rho;
       end
       
%        [~, idx] = max(F,[],2);
%        [~, idx2] = max(y,[],2);
%        acc = sum(idx==idx2)/length(idx)
       
%        Fn = max(F(:,end-param.k+1:end),[],2);
%        nl = idx2>=20;
%        [~,~,~,AUC] = perfcurve(nl,Fn,1)
       
   end
end

function G = get_g_func(Z)
%%Compute softmax function g for instance prediction
%%Given z=xw
   mx = max(Z,[],2);
   Z = Z - repmat(mx,1,size(Z,2));
   Z(Z<-15) = -15;
   P = exp(Z);
   s = sum(P,2);
   idx = s==0;
   invs = 1./s;
   invs(idx) = 0;
   G = diag(invs)*P;
end

function [ H, Beta, W, Alpha, S] = initialization(Y, Bags,param )
%initialization of variables

   k = param.k;

   
   insts = cell2mat(Bags);
   
   [idx, ~] = kmeans(insts, size(Y,2)+k);
   H  = ones(size(insts,1), size(Y,2)+k)*1e-3;
   for j=1:size(insts,1)
      H(j,idx(j)) = 1;
   end
   nmz = sum(H);
   for j=1:size(insts,1)
      H(j,idx(j)) = 1/sqrt(nmz(idx(j)));      
   end

   S = repmat(1./sqrt(nmz),size(H,1),1);
   
   [ Beta ] = update_beta(Bags, H, H./S, S );
   W = (rand(size(insts,2),size(Y,2)+k)*2-1)/1000;
   Alpha = zeros(size(H));
end

function [ alpha ] = update_alpha( H, F, S, alpha)
  %%ADMM update rule 3, update alpha
  alpha = alpha + H-F.*S;
end

function [ Beta ] = update_beta(Bags, H, F, S)
   %%Update beta, the instance weights
   idx = 0;
   %H = (H + F.*S)/2;
   Beta = cell(length(Bags),1);
   for i=1:length(Bags)
       bgsz = size(Bags{i},1);
       Hi = H(idx+1:idx+bgsz,:);
       betai = zeros(bgsz,1);
       [~,lidx] = max(Hi,[],2);
       lp = unique(lidx);
       for j=1:length(lp)
           a = 1/sum(lidx==lp(j));
           betai(lidx==lp(j)) = a;           
       end
       Beta{i} = betai;
       idx = idx + bgsz;
   end
end

function  [H] = update_h(H, F, S, A, Beta, Y, Bags,   insts,  lambda1 , rho, k,i, disc, decayh, squaredecayh, inner_iter )
     %%ADMM update rule 2, update H, the normalized clustering index matrix                   
    for i=1:inner_iter
       [pgrad,ngrad] = GetGradientH(H, F, S, A, Beta, Y, Bags, insts,  lambda1, rho, k, squaredecayh);
       H = H*(1-disc) + disc* (H.* (ngrad./pgrad).^(1/(1+sqrt(i)*decayh)));
        z=sum(H'*H);
        z = repmat(z,size(H,1),1);
        H = H./sqrt(z);
    end
end

function [pgrad,ngrad] = GetGradientH(H, F, S, A, Beta, Y, Bags, insts,  lambda1, rho, k,t)
   pgrad = zeros(size(H));
   ngrad = zeros(size(pgrad));
   idx = 0;
   for i=1:length(Bags)
       bgsz = size(Bags{i},1);
       Hi = H(idx+1:idx+bgsz,:);
       Ci = S(idx+1:idx+bgsz,:);
       Yi = Y(i,:);
       betai = Beta{i}';
       [pgradi,ngradi] = GetGradientHi(Hi,Yi,Ci,betai,k);
       pgrad(idx+1:idx+bgsz,:) = pgradi;
       ngrad(idx+1:idx+bgsz,:) = ngradi;
       idx = idx + bgsz;
   end
      
   pgrad0 = pgrad + rho*H + rho*(abs(A)+A)/2;
   ngrad0 = ngrad + rho*F.*S + rho*(abs(A)-A)/2 +  lambda1*insts*(insts'*H);
      
    pgrad = pgrad0 + H*ngrad0'*H*t;
    ngrad = ngrad0 + H*pgrad0'*H*t;
end

function [pgradi,ngradi] = GetGradientHi(Hi,Yi,Ci,betai,k)
   pgradi = zeros(size(Hi));
   ngradi = pgradi;
   Hitmp = Hi(:,1:end-k);
   Citmp = Ci(:,1:end-k);
   pgradi(:,1:end-k) = (betai'*(betai*(Hitmp./Citmp)))./Citmp;
   ngradi(:,1:end-k) = (betai'*Yi)./Citmp;
end

function S = update_s(H, F, S)
    %update S, normalization coefficient of H
       %H = (H + F.*S)/2;
       [~,idx] = max(H,[],2);     
       Hhat  = ones(size(H,1), size(H,2))*1e-3;
       for j=1:size(H,1)
          Hhat(j,idx(j)) = 1;
       end
       nmz = sum(Hhat+1e-6);
       S = repmat(1./sqrt(nmz),size(Hhat,1),1);
end

function [ W, AW, Anum, trounds ] = update_w(W, AW, Anum, Y, Bags, Beta, H, Alpha, S, rho, lambdaw, step_size0, trounds, k, disc, clipnorm, decayw)
   %%ADMM rule 1, update W via sgd, weight matrix in the model
   idx = 0;
   Wt = W;
   AWt = AW;
   Anumt = Anum;
   troundst = trounds;
   idx2 = 0;
   
   idxtmp = [1];
   idxtmp2 = [1];
   for i=1:length(Bags)
        bgsz = size(Bags{i},1);
        idx2 = idx2 + bgsz;
        idxtmp2 = [idxtmp2;idx2+1];
   end
   Yis = cell(length(Bags));
   Betais = Yis;
   His = Yis;
   Alphais = Yis;
   Cis = Yis;
   for i = 1:length(Bags)
       bgsz = size(Bags{i},1);
       Yi = Y(i,:);
       betai = Beta{i}';
       Hi = H(idx+1:idx+bgsz,:);
       Alphai = Alpha(idx+1:idx+bgsz,:);
       Ci = S(idx+1:idx+bgsz,:);
       Yis{i} = Yi;
       Betais{i} = betai;
       His{i} = Hi;
       Alphais{i} = Alphai;
       Cis{i} = Ci;
       idx = idx + bgsz;
       idxtmp = [idxtmp;idx+1];
   end
   ii = randperm(length(Bags));
   for i = 1:length(Bags)
       Bagi = Bags{ii(i)};
       Yi = Yis{ii(i)};
       betai = Betais{ii(i)};
       Hi = His{ii(i)};
       Alphai = Alphais{ii(i)};
       Ci = Cis{ii(i)};
       
       grad = GetGradWPerBag(W, Yi, Bagi, betai, Hi, Alphai, Ci, rho, lambdaw, k);
       grad_norm = norm(grad,'fro');
       if grad_norm > clipnorm
            grad = grad*(clipnorm/grad_norm);
       end
       
       step_size=step_size0/(1+decayw*trounds*step_size0);
       W = W*disc + (1-disc) * (W - step_size*grad);

       AW = [AW; {W}];
       Anum = Anum + 1;

       trounds = trounds + 1;
       
   end
   trounds = 0;
   AW(1:floor(length(AW)/2)) = [];
   Anum = floor(length(AW)/2);
end

function grad = GetGradWPerBag(W, Yi, Bagi, betai, Hi, Alphai, Ci, rho, lambdaw, k)
   Fi = get_g_func(Bagi*W);
   grad = lambdaw*W + rho*Bagi'*((Fi.*Ci-Hi-Alphai).*Ci.*Fi.*(1-Fi));
   grad(:,1:end-k) = grad(:,1:end-k) + Bagi'* ((betai'*(betai*Fi(:,1:end-k)-Yi)).*Fi(:,1:end-k).*(1-Fi(:,1:end-k)));
end

function [X, X_bound]=AddBoundBag(X,minx,maxx,resolution)
u = linspace(minx,maxx, resolution);
v = linspace(minx, maxx, resolution);
X_bound=[];
for i = 1:length(u)
    for j = 1:length(v)
        X_bound=[X_bound [u(i) v(j)]'];
    end
end
X{length(X)+1}=X_bound;
end

function [X_out]=kernelize(X, kerneltype, width)
%%Compute random Fourier feature, which is equivalent to a kernel mapping
load('g.mat'); 
g = g*width;

for i = 1:length(X)
    X_out{i}=[cos(X{i}'*g'), sin(X{i}'*g')]';
end
X_out = X_out';
end

function PlotData(X,p,maxx, minx)
color=['b' 'g' 'r' 'c' 'm' 'y' 'k' 'w' 'b' 'g'];
y=zeros(size(X,2),1);

for i=1:size(X,2)
    [value,label]=max(p(i,:));
    y(i)=label;
end

z=unique(y);

for i=1:length(z)
    index=find(y==z(i));
    plot(X(1,index),X(2,index),strcat('.',color(z(i))),'MarkerSize',20);
    axis([minx maxx minx maxx]);
    hold on; 
end
end

function PlotResultBoundary(F,X_bound_data,maxx, minx)
PlotData(X_bound_data,F,maxx, minx);
end

function [X_out]=PreprocessingX(X,kernel,kernel_method,scale)
w = warning ('on','all');
rmpath('folderthatisnotonpath');
warning(w);
id = w.identifier;
warning('off',id);
rmpath('folderthatisnotonpath');

if(strcmp(kernel,'kernel')==1)
    [X_out]=kernelize(X,kernel_method,scale);
end
end






