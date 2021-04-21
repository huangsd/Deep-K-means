%function [F,G] = CNMF(X, Finit, Ginit, ratio, maxiter)
function [Finres, Hres, myobj] = DeepKMeans(X, layers, labels, rho, maxiter)
%
%solve the following problem£º
% Deep $K$-Means for data clustering
% min||X-Y||_{2,1}
% s.t. Y = U1U2...UrVr^{T}, Vi = Vi^{+}, Vi^{+}>=0, i\in [1, ..., r].
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% written by Shudong Huang on 15/1/2019
% Reference: Shudong Huang, Zenglin Xu, Zhao Kang, Ivor Tsang. 
% Deep $K$-Means: A Simple and Efficient Method for Data Clustering. 
% In: Proceedings of the International Conference on Neural Computing for Advanced Applications (NCAA), 2020.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% ATTN1: This package is free for academic usage. The code was developed by Mr. SD. Huang (huangsd@std.uestc.edu.cn). You can run
% it at your own risk. For other purposes, please contact Prof. Zenglin Xu (zenglin@gmail.com)
%
% ATTN2: This package was developed by Mr. SD. Huang (huangsd@std.uestc.edu.cn). For any problem concerning the code, please feel
% free to contact Mr. Huang.
%
% Input:
% X: dim x num
% k: num of clusters
% H: num x k
% Z: dim x k
% rho: penalty parameter
% ratio: the ratio of outliers, set it according to the paper
% e.g., layers = [100 50] ;
%
% Output:
% Hres: the clustering indicator matrix
% myobj: objective value
%

disp('Deep K-Means');

% dimension of data
[numFea, numSamp] = size(X);

% normalization
% X = bsxfun(@rdivide,X,sqrt(sum(X.^2,1)));

%
numlayer = length(layers);
Z = cell(1, numlayer);
H = cell(1, numlayer);

% initialize each layer
for i_layer = 1:numlayer
    if i_layer == 1
       V = X; 
       % For the first layer we go linear from X to Z*H
     else
        V = H{1,i_layer-1};
    end
     if i_layer == 1
         disp('The 1st layer ...');
     else if i_layer == 2
             disp('The 2nd layer ...');
         else
             fprintf('the %d-th layer\n', i_layer)
         end
     end
     if i_layer > 1
         V = V';
     end 
        % initialize input with kmeans
        % [Z{i_layer}, H{i_layer}] = KMeansdata(V, layers(i_layer));
        
        % initialize input with NMF
          [Z{i_layer}, H{i_layer}] = NMFdata(V, layers(i_layer));
end

% initialize \lambda and H^+
for i = 1:numlayer
    lambda{i} = zeros(size(H{i}));
    Hplus{i} = H{i}; 
end

% update ...

H_err = cell(1, numlayer);

  %  Delta = X - F*G';

for iter = 1:maxiter 

    if mod(iter, 50) == 0
       % fprintf('numOfOutliers = %d, ratio = %f\n', length(Idx),ratio);
       % fprintf('%dth iteration, obj = %f \n', it, obj);
       fprintf('processing iteration %d...\n', iter);
    end
    
    H_err{numel(layers)} = H{numel(layers)};
    % update Z
    for i_layer = numel(layers)-1:-1:1
        H_err{i_layer} = (Z{i_layer+1} * H_err{i_layer+1}')';
    end
    
    % initialize \mu 
    if iter ==1
        mu = zeros(size(Z{1}*H_err{1}'));
    end
    
    % compute D
    D = computeD(X, Z{1}, H_err{1});
    
    for i = 1:numel(layers)
        
%          for v = 1: numView
%              M = (G0'*D4{v}*G0);
%              N = inXCell{v}*D4{v}*G0;
%              F{v} = N/M;
%          end
%       
%        update Z, i.e., U
        if i == 1
           % Z{i} = Z{i}.*((X*D*H_err{i})./(Z{i}*H_err{i}'*D*H_err{i}+eps));
           % Z{i} = (X*D*H_err{i})/(H_err{i}'*D*H_err{i}+eps);
           
           % initialize Y
           Y = Z{1}*H_err{1}';
           
           Z{i} = (mu*H_err{1}/rho+Y*H_err{1})*pinv(H_err{1}'*H_err{1});
          % Z{i} = (mu*H_err{1}/rho+Y*H_err{1})/(H_err{1}'*H_err{1});
            
        else
           
          % update Z, i.e., U  
         %  Z{i} = ((ZZ'*ZZ)\(ZZ'*Y*H_err{i}+(ZZ'*mu*H_err{i})/rho))/(H_err{i}'*H_err{i});
           Z{i} =  pinv(ZZ'*ZZ)*(ZZ'*Y*H_err{i}+(ZZ'*mu*H_err{i})/rho)*pinv(H_err{i}'*H_err{i});
          % A/B = A*B^{-1}
          % B\A = B^{-1}*A
          % A\b=inv(A)*b=pinv(A)*b
           
        end
        
        if i == 1
            ZZ = Z{i};
        else
            ZZ = ZZ*Z{i};
        end

      % update H, i.e., V
        H{i} = (Y'*ZZ+Hplus{i}+(mu'*ZZ)/rho-(lambda{i})/rho)*pinv(eye(size(Z{i},2))+ZZ'*ZZ);
       % H{i} = (Y'*ZZ+Hplus{i}+(mu'*ZZ)/rho-(lambda{i})/rho)/(eye(size(Z{i},2))+ZZ'*ZZ);
       
      % update Y
        Y = (2*X*D+rho*Z{1}*H_err{1}'-mu)/(2*D+rho*eye(size(D,1)));
      % Y = (2*X*D+rho*Z{1}*H_err{1}'-mu)*pinv(2*D+rho*eye(size(D,1)));
      
      % update V^{+}
      Hplus{i} = max(H{i}+lambda{i}/rho,0);
      
      % update mu and lambda
      mu = mu+rho*(Y - Z{1}*H_err{1}');
      lambda{i} = lambda{i}+rho*(H{i}-Hplus{i});   
    end
     
    % objective value
    T = X - Y;
    obj = sum(diag(T*D*T'));
    myobj(iter) = obj;
end
    
% 
Hres = H{numel(H)};
% Finres = litekmeans(H',nClass,'Replicates',100);
Finres = litekmeans(Hres,length(unique(labels)),'Replicates',20);

% result = ClusteringMeasure(labels, Finres)  % result = [ACC MIhat Purity];

end