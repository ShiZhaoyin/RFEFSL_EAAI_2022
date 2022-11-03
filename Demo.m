load Jaffe_32x32.mat
for iter = 1:size(H,3)
    [~,lable1] = max(U(:,:,iter),[],2);
    rr(iter,:) = ClusteringMeasure(gnd,lable1);
end
Max = max(rr); Min = min(rr);  Mean = mean(rr); Std = std(rr);
p = 1; m = [1.01,1.05,1.08,1.1,1.2,1.3,1.4,1.5,2]'; lambda = [0.01,0.05,0.08,0.1,0.2,0.3,0.4,0.5,1]';
ACC = zeros(length(m),length(lambda)); NMI = zeros(size(ACC)); Purity = zeros(size(ACC)); T = zeros(size(ACC));
for i = 1:length(m)
    for j = 1:length(lambda)
        Tstart = tic;
        %              H = gpuArray(H);
        [label,F,S,~,Alpha,Sigma] = RFuzzyEn2LapRank_modified(H,p,m(i),lambda(j));
        T(i,j) = toc(Tstart);
        result =  ClusteringMeasure(gnd,label);
        ACC(i,j) = result(1); NMI(i,j) = result(2); Purity(i,j) = result(3);
%       fprintf('Current Data is:%s\n',DataName);
        fprintf('Current Results: ACCis%4.4f NMIis%4.4f Purityis%4.4f\n',ACC(i,j),NMI(i,j),Purity(i,j))
    end
end
Tsum = sum(T(:));
save('Result.mat','Max','Mean','Min','Std','rr',...
    'U','H','F','S','Alpha','Sigma','m','lambda','p',...
    'T','Tsum','ACC','NMI','Purity','gnd');