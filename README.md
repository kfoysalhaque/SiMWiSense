# SiMWiSense

This is the implementation of the paper [SiMWiSense: Simultaneous Multi-Subject Activity Classification Through Wi-Fi Signals](https://ieeexplore.ieee.org/document/10195411). The repository shares both the datasets and the source code of **SiMWiSense.**

If you find the project useful and you use this code, please cite our paper:

```

@INPROCEEDINGS{10195411,
  author={Haque, Khandaker Foysal and Zhang, Milin and Restuccia, Francesco},
  booktitle={2023 IEEE 24th International Symposium on a World of Wireless, Mobile and Multimedia Networks (WoWMoM)}, 
  title={SiMWiSense: Simultaneous Multi-Subject Activity Classification Through Wi-Fi Signals}, 
  year={2023},
  volume={},
  number={},
  pages={46-55},
  doi={10.1109/WoWMoM57956.2023.00019}}

```
## Download SiMWiSense Dataset

(I) clone the repository with ``` git clone git@github.com:kfoysalhaque/SiMWiSense.git ```  <br/>
(II) ```cd SiMWiSense``` <br/>
(III) Then download the [SiMWiSense dataset](https://drive.google.com/file/d/1VYuxtIjM5tMCzduyCQrc5L_msXmtJq5U/view?usp=sharing) within the repository. <br/>
(IV) Unzip the downloaded file with ``` sudo unzip Data.zip ``` <br/>

## SiMWiSense Tests
We captured the IEEE 802.11ac CSI pcap files with 80MHz bandwidth for **three different Tests**: <br/>

**(I) proximity_test**: We address the simultaneous multi-subject classification issue by using the Channel State Information (CSI) computed from the device positioned closest to the subject. We experimentally prove this intuition by confirming that the best accuracy is experienced when the CSI computed by the transceiver positioned closest to the subject is used for classification. We call this **Proximity Test.** <br/>

**(II) coarse_detection**: Now, we move on to the simultaneous sensing. To tackle the challenge of scalability, we propose decentralized detection. Specifically, a learning model is assigned to each device to sense the subject which is closest to it. We call this coarse detection. (**subject identification**) <br/>

**(III) fine_grained**: After the coarse detection, another fine-grained DL model is used to determine the activities. (**activity classification**) <br/>

<br/>


## Data Preprocessing

### CSI extraction with Matlab and generation of CSI samples 

To extract the CSI data from the captured Pcap files, we leverage Matlab. However, you can also use Python parsing(I have started using Python, it's faster). The Matlab scripts are in the ```SiMWiSense/Matlab_code/``` directory. <br/>

**(I)** At first extract the CSI from the pcap files by executing the ```CSI_extractor_SimWiSense.m``` script. Please change the ```Test="xxxx"``` field to different Test names. If you want to extract the data for all three above-mentioned tests,  execute the script separately for ```Test="proximity"```, ```Test="coarse"``` and ```Test="fine_grained"``` . 
You can either execute the script from Matlab GUI or from the terminal using: <br/>

```
matlab -batch "CSI_extractor_SimWiSense.m; exit;"
```
<br/>
**Please be patient in this step as it takes quite a long time for complete execution as the captured files are very big and Matlab parsing is slow.** <br/>
<br/>

**(II)** The next step would be to divide the extracted CSI file into multiple samples each having 50 packets.  <br/>

**For ```coarse``` and ```proximity``` tests, execute ```csi2batches_SimWiSense.m```** <br/>
<br/>
==> For ```proximity``` and ```coarse``` test, change the ```Test='xxxx'``` field of the script to ```Test='proximity'``` and  ```Test='coarse'``` respectively.  <br/>

<br/>
<br/>

**For ```fine_grained``` Test, execute ```csi2batches_SimWiSense_fine_grained.m```** <br/>
<br/>
Please change ```monitor ="xx"``` field to ```monitor ="m1"``` or ```monitor ="m2"``` or ```monitor ="m3"``` for different monitors which will simultaneously classify human activities. 
<br/>

### Create CSV files by listing the generated samples for datagenerator 

<br/>

All the Python scripts including the scripts to generate CSV are with the directory ```SiMWiSense/Python_code/```

<br/>

==> execute the script ``` csv_main.py ``` which takes the Test name (proximity / coarse / fine_grained) as an argument. For example:
<br/>
```
python csv_main.py proximity
```
<br/>



## SiMWiSense with Baseline CNN

### To do the proximity test with baseline CNN execute

```
python baseline_proximity.py <'Environment'> <'Closest_STA'> <'Train_Test_STA'> <'model_name'> <'no of subcarriers'>
```
<br/>

**Example:** If you want to perform the test when the subject is performing activity closest to station "m1" and train & test with data from station "m1" with 242 subcarriers

```
python baseline_proximity.py Classroom m1 m1 Classroom_m1_m1.h5 242
```
<br/>

If you want to perform the test when the subject is performing activity closest to station "m1" and train & test with data from station "m3"

```
python baseline_proximity.py Classroom m1 m3 Classroom_m1_m3.h5 242
```
<br/>


### To do the coarse test with baseline CNN execute

```
python baseline_coarse.py <'Train_Environment'> <'Train_STA'> <'Test_Environment'> <'Test_STA'> <'model_name'> <'no of subcarriers'>
```
<br/>

**Example:** If you want to perform the coarse test when trained with ```Classroom``` ==> station ```m1```, and test with ```Classroom``` ==> station ```m1``` with 242 subcarriers
```
python baseline_coarse.py Classroom m1 Classroom m1 Cls_m1_Cls_m1.h5 242
```
<br/>

If you want to perform the coarse test when trained with ```Classroom``` ==> station ```m1```, and test with ```Office``` ==> station ```m1``` with 80 subcarriers

```
python baseline_coarse.py Classroom m1 Office m1 Cls_m1_Ofc_m1.h5 80
```
<br/>

### To do the fine_grained test with baseline CNN execute

```
python baseline_fine_grained.py <'Train_Environment'> <'Train_STA'> <'Test_Environment'> <'Test_STA'> <'model_name'> <'no of subcarriers'>
```
<br/>

**Example:** If you want to perform the coarse test when trained with ```Classroom``` ==> station ```m1```, and test with ```Classroom``` ==> station ```m1``` with 242 subcarriers
```
python baseline_fine_grained.py Classroom m1 Classroom m1 Cls_m1_Cls_m1.h5 242
```
<br/>

If you want to perform the coarse test when trained with ```Office``` ==> station ```m2```, and test with ```Classroom``` ==> station ```m3``` with 80 subcarriers

```
python baseline_fine_grained.py Office m2 Classroom m3 Ofc_m2_Cls_m3.h5 80
```
<br/>


## SiMWiSense generalization with Feature Reusable  Feature Reusable Embedding Learning (FREL)

### To do the coarse test and fine_grained test with FREL

```
python main.py <'Test_name'> <'Train_Environment'> <'Train_STA'> <'Test_Environment'> <'Test_STA'> <'model_name'> <'no of subcarriers'> -ft -tr
```
<br/>

**Test_name can be coarse or fine_grained**
<br/>

**If you want to only fine tune (from the earlier trained model) use only ``` -ft ``` instead of ``` -ft -tr ```**

<br/>

**Example:** If you want to perform the coarse test when trained with ```Classroom``` ==> station ```m1```, and test with ```Classroom``` ==> station ```m1``` with 242 subcarriers
```
python main.py coarse Classroom m1 Classroom m1 Cls_m1_Cls_m1.h5 242 -ft -tr
```
<br/>

If you want to perform the fine_grained test when trained with ```Office``` ==> station ```m1```, and test with ```Classroom``` ==> station ```m3``` with 242 subcarriers

```
python main.py fine_grained Office m1 Classroom m3 Ofc_m1_Cls_m3.h5 242 -ft -tr
```
<br/>

For either coarse or fine_grained test, If you have the saved model ==> ```classroom_m2.h5``` trained in ```Classroom``` ==> station ```m2```, **AND** fine tune with **FREL** for a different environment, for example, ```Office``` with station ```m3```, execute:

For ```fine_grained``` test:
<br/>
```
python main.py fine_grained Classroom m2 Office m3 classroom_m2.h5 242 -ft 
```
<br/>

For ```coarse``` test:
<br/>
```
python main.py coarse Classroom m2 Office m3 classroom_m2.h5 242 -ft 
```
<br/>


### For any question please contact [Foysal Haque](https://kfoysalhaque.github.io/) at _**haque.k@northeastern.edu**_
