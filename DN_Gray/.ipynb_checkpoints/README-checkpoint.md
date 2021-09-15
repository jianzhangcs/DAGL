# Train
python train.py --save_path=exp --dir_data=DIV2K --lr=2e-4 --noiseL=25 --batch_size=32

Download datasets from the provided links and place them in this directory. Your directory structure should look something like this：

`DIV2K` <br/>
  `└──`DIV2K_HQ <br/>

# Test
python test.py --ensemble --logdir=checkpoints/25/model/model_best.pt --test_noiseL=25. --test_data=testsets/Set12
