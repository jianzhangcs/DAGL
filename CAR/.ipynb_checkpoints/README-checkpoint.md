# Train
python train.py --dir_data=DIV2K/ --q=10 --save_path=exp --lr=2e-4 --batch_size=32

Download datasets from the provided links and place them in this directory. Your directory structure should look something like this：
      
`DIV2K` <br/>
  `├──`DIV2K_HQ <br/>
  `└──`DIV2K_LQ <br/>
      `└──`Quality factor (10,20,30,40)

# Test
python test.py --logdir=checkpoints/mq/model/model_best.pt --quality=10 --test_data=testsets/Classic5 --ensemble --save_res
