clear;
load('./Seq_Num/sn90.mat')
load('./Seq_Num/sn180.mat')
load('./Seq_Num/sn135.mat')
idx_90 = []; idx_180 = [];
last_90 = []; last_135 = [];
for i = 1:size(seq_num_135,1)
    for j = 1:size(seq_num_90,1)
        if (prod( seq_num_135(i,:) == seq_num_90(j,:)))
            idx_90 = [idx_90,j];
        end
    end
end

% for i = 1:size(seq_num_180,1)
%     if (prod( seq_num_135(idx_90(1),:) == seq_num_180(i,:)))
%         idx_180_mtch = i;
%         idx_180 = [idx_180,i]
%     end
% end

for i = idx_90(1):size(seq_num_90,1)
    for j = 1:size(seq_num_180,1)
        if (prod( seq_num_90(i,:) == seq_num_180(j,:)))
            idx_180 = [idx_180,j];
        end
    end
end

%%% last index
for i = idx_90(1):size(seq_num_90,1)
    if seq_num_180(end,:)== seq_num_90(i,:)
        last_90 = [last_90,i];
    end
end
for i = 1:size(seq_num_135,1)
    if seq_num_180(end,:)== seq_num_135(i,:)
        last_135 = [last_135,i];
    end
end

%%
csi = csi_buff(1:24,:);
csi_c1 = csi(1:4:end,:);
csi_c2 = csi(2:4:end,:);
csi_c3 = csi(3:4:end,:);
csi_c4 = csi(4:4:end,:);