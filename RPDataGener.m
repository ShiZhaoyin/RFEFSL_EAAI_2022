DataPath = 'D:\Myresearch\Ensemble\NewData\Image\';
Files = dir(fullfile([DataPath,'*.mat']));
LengthFiles = length(Files); 
for i = 7
    DataName = Files(i).name;          
    load([DataPath,'\',DataName]);    
    iter = 0;
    %% 处理代码
    H = zeros(size(X,2),k,30);
    for d = size(X,1)/8:size(X,1)/8:size(X,1)/8*6
        for j = 1:5
            RP = randsrc(size(X,1),d,[-sqrt(3),sqrt(3),0;1/6,1/6,2/3]);
            Xnew = 1/sqrt(d)*RP'*X;
%             [~,uu] = fcm(Xnew',k,1.1);
%             uu = uu';
                        [uu,lable] = MyFCM(Xnew,k,1.1);
%                         try
            aa = uu/diag(sum(uu,1))*uu';
            [hh,~,~] = svds(aa,k);
            H(:,:,iter+1) = hh;
            %             catch
            %                 continue
            %             end
            U(:,:,iter+1) = uu;
            [~,label] = max(uu,[],2);
            result = ClusteringMeasure(gnd,label);
            iter = iter + 1;
        end
    end
    save(['D:\Myresearch\Ensemble\BaseClusterings\',DataName],'gnd','H','U')
    U = []; H = [];
end