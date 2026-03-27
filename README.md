# Installation
1. Create new environment   
``` conda create -n acdc python=3.10 ```

2. Activate environment   
```conda activate acdc```

3. Change directory to extracted project folder   
```cd [project_folder]/acdc-inverse```

4. Install requirements   
``` pip install -r requirements.txt```

5. Create a folder for pretrained models   
``` mkdir pretrained-models```

6. Download and put the FFHQ model in this directory [Pretrained Model Link](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh) or use the following command   
```gdown --id 1BGwhRWUoguF-D8wlZ65tf227gp3cDUDh -O pretrained-models/ffhq_10m.pt```   


7. Download file for bkse blur from [bkse model](https://drive.google.com/file/d/1vRoDpIsrTRYZKsOMPNbPcMtFDpCT6Foy/view?usp=drive_link) to  *measurements/bkse/experiments/pretrained/*  or just run    
``` gdown --id 1vRoDpIsrTRYZKsOMPNbPcMtFDpCT6Foy -O measurements/bkse/experiments/pretrained/GOPRO_wVAE.pth ```

8. Run the test    
```sh test.sh```   
