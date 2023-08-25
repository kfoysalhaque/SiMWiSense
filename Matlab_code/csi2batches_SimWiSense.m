% Copyright (C) 2023 Khandaker Foysal Haque
% contact: haque.k@northeastern.edu
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <https://www.gnu.org/licenses/>.

clc;
clear all;
close all;


env = ["Classroom"; "Office"];
monitor =["m1", "m2", "m3"];
Test = 'proximity';   % Options are "coarse" or "proximity"


BW = '80MHz'
tot_mon='3';
window_size=50;


if Test == "proximity"
    
    activity=['A' ; 'B' ; 'C' ; 'D' ; 'E' ; 'F'; 'G'; 'H'; 'I'; 'J'; 'K'; 'L'; 'M' ; 'N' ; 'O' ; 'P'; 'Q' ;'R'; 'S'; 'T'];
    slots=["Train_m1", "Test_m1", "Train_m2", "Test_m2", "Train_m3", "Test_m3" ];
    
    start=[5, 180, 245, 427, 490, 672 ];
    stop=[180, 240, 425, 485, 670, 730];

elseif Test == "coarse"
    
    activity=['A' ; 'B' ; 'C' ; 'D'];
    slots=["Train", "Test"];

    start=[5, 242];
    stop=[240, 300];

end


for e = 1:length(env)
    for k=1:length(monitor)
        for m = 1:length(activity)
             folder_name = sprintf('../Data/%s/%s/%s/%smo/%s/%s/', Test, env(e,:), BW, tot_mon, monitor(k), activity(m));
    
    
        files = dir(fullfile(folder_name, '*.mat'));
    
            for file_idx = 1:numel(files)
                FILE = strcat(folder_name, files(file_idx).name); % capture file
                load(FILE);
                discard=5;
                disp(activity(m))
                a = abs(length(csi)/814);
                packet_start=[a*start];
                packet_stop=[a*stop];
    
                for v = 1:length(packet_stop)
                    csi_slot=csi(packet_start(v):packet_stop(v),:);
                    num_p= size(csi_slot,1)
                    window=window_size;
                    num_image = floor(num_p/window)
                    folder_save = sprintf('../Data/%s/%s/%s/%smo/%s/Slots/%s/%s_batch/',Test, env(e,:), BW, tot_mon, monitor(k), slots(v), activity(m));
    
                    % process data window by window
                    for i = discard:num_image-discard
                        csi_mon = [];
                        csi_mon = [csi_mon; csi_slot((i-1)*window+1:i*window,:)];
                        mat_name = strcat(folder_save,'batch','_',string(i-discard),'.mat');
                        save(mat_name, 'csi_mon');
                    end
    
                end
    
            end
    
    
        end
    end
end
