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

clc; clear; close all

%%% change the parameters as required
Test= "proximity";  %%%% Options are "proximity", "coarse", "fine_grained"
env = ["Classroom"; "Office"];
BW = 80;
CHIP = '4366c0';     % wifi chip (possible values 4339, 4358, 43455c0, 4366c0)
total_monitor=3;
save_in =sprintf('../Data/%s', Test);


if Test == "proximity"
    act = ['A' ; 'B' ; 'C' ; 'D' ; 'E' ; 'F'; 'G'; 'H'; 'I'; 'J'; 'K'; 'L'; 'M' ; 'N' ; 'O' ; 'P'; 'Q' ;'R'; 'S'; 'T'];

elseif Test == "coarse"
    act=['A' ; 'B' ; 'C' ; 'D'];

else
    act = ['A' ; 'B' ; 'C' ; 'D' ; 'E' ; 'F'; 'G'; 'H'; 'I'; 'J'; 'K'; 'L'; 'M' ; 'N' ; 'O' ; 'P'; 'Q' ;'R'; 'S'; 'T'];
end

NPKTS_MAX = 100000000000;



if BW == 20    %%% Discard the pilot, null, guard subcarriers
    non_zero = [5:32,34:61]; %%% non-zero indices
else
    non_zero = [7:128,132:251];
end

check = [];

for e = 1:length(env)

    for num_mo = 1:total_monitor
        for a = 1:length(act)
            seq_plane = cell(num_mo,1);
            core_plane = cell(num_mo,1);
            bad_key = cell(num_mo,2);
            data_mon = cell(num_mo,1);
            data_raw = cell(num_mo,1);
    
            %%% generate the desired .pcap file name
            sub_file = sprintf('%s.pcap',act(a));
            FILE = sprintf('%s/%s/%dMHz/%dmo/m%d/CSI_pcap/%s',  save_in, env(e,:),BW,total_monitor, num_mo,sub_file);
            fprintf(1, 'Now reading %s\n', FILE);
    
            %%% read file
            HOFFSET = 16;           % header offsets
            NFFT = BW*3.2;          % fft size
            p = readpcap();
            p.open(FILE);
            n = min(length(p.all()),NPKTS_MAX);
            p.from_start();
            csi_buff = complex(zeros(n,NFFT),0);
            k = 1;
            seq_num = [];
            core_num = [];
            while (k <= n)
                f = p.next();
                if isempty(f)
                    print('no more frames');
                    fprintf('no more frames');
                    break;
                end
    
                if f.header.orig_len-(HOFFSET-1)*4 ~= NFFT*4
                    disp('skipped frame with incorrect size');
                    continue;
                end
                payload = f.payload;
                P14 = dec2hex(payload(14),8);
                seq_num = [seq_num; P14(5:end)];
                core_num = [core_num; P14(1:2)];
    
                H = payload(HOFFSET:HOFFSET+NFFT-1); %% header removed
    
                if (strcmp(CHIP,'4339') || strcmp(CHIP,'43455c0'))
                    Hout = typecast(H, 'int16');
                elseif (strcmp(CHIP,'4358'))
                    Hout = unpack_float(int32(0), int32(NFFT), H);
                elseif (strcmp(CHIP,'4366c0'))
                    Hout = unpack_float(int32(1), int32(NFFT), H);
                else
                    disp('invalid CHIP');
                    break;
                end
                Hout = reshape(Hout,2,[]).';
    
                cmplx = double(Hout(1:NFFT,1))+1j*double(Hout(1:NFFT,2));
                csi_buff(k,:) = cmplx.';               
                k = k + 1;
            end
            seq_plane{num_mo} = seq_num;
            core_plane{num_mo} = core_num;
            data_raw{num_mo} = csi_buff;
    
            activity = act(a)
    
            fprintf('you made it to saving!');
    
            csi = [];
            csi_mon=data_raw{num_mo};
    
            csi = csi_mon(:,non_zero);
            num_p= size(csi_mon,1);
    
            mat_name = sprintf('./%s/%s/%dMHz/%dmo/m%d/%s/%s.mat',...
                save_in, env(e,:), BW, total_monitor, num_mo, act(a), activity);
            save(mat_name,'csi',  '-v7.3')
    
        end
    
    end
end
