# Image Demosaicing  
![PSNR_DN_Gray](/Figs/PSNR_SSIM_dmic.PNG)
![Visual_DN_Gray](/Figs/vis_dmic.PNG)
# Train
python train.py --save_path=exp --dir_data=DIV2K --lr=2e-4 --batch_size=32

Download datasets from the provided links and place them in this directory. Your directory structure should look something like this：

`DIV2K` <br/>
  `├──`DIV2K_HQ <br/>
  `└──`DIV2K_LQ <br/>

# Test
python test.py --test_data=testsets/McMaster18 --results_path=results/set24 --save_image --logdir=checkpoints/dmic/model/model_best.pt --ensemble
