# fractional-denoising-diffusion-probabilistic-model

## Clone git repository branch
```bash
git clone -b master --single-branch https://github.com/UNIST-LIM-Lab/fractional-denoising-diffusion-probabilistic-model.git
```

## Environment Setting
create conda environment with required packages(requirements.txt).

```bash
conda create -n dpm python=3.9
conda activate dpm
pip install -r requirements.txt
```

## Training
URDF files are saved in urdf directory, which specify robot(number of components) of environment.
If you want to change the robot in environment, just add your own config yml file in configs directory, and modify `--config` setting.

```bash
python main.py --config <config yml file> --exp <exp dir name>

python main.py --config cifar10.yml --exp "train_cifar10"
nohup python main.py --config cifar10.yml --exp "train_cifar10" > train_cifar10.out &
```

If you want to resume training, specify `--exp` same as previous `--exp` and add below commands.
```bash
nohup python main.py --config cifar10.yml --exp "train_cifar10" --resume_training --ni > train_cifar10_resume.out &
```
   
If you want to specify GPUs to use, modify below code at the top of main.py. You can easily perform distributed data-parallel training by only specifying GPU indicies like below.
```bash
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
```

## tensorboard
You can see the training loss plot with tensorboard, but it should be accessed during training.
After running below command, you can access tensorboard `http://localhost:<port number>`.

```bash
tensorboard --logdir=<exp/tensorboard path> --port=<port number>

tensorboard --logdir=/home/~/exp/tensorboard --port=6001
```
   
If you cannot access `http://localhost:<port number>` in your local computer(MAC), change port of server to available local port like below,
```bash
ssh -NfL localhost:<local port number>:localhost:<server port number> hostname@server_ip

ssh -NfL localhost:8891:localhost:6001 keeeehun@batman
```

## sampling
```bash
python main.py --sample --config <config yml file> --sample_type <sample_type> --skip_type <skip_type> --time_step <time_step> --fid --seed <seed_num> --exp <exp dir name>

python main.py --sample --config cifar10.yml --sample_type ddim --skip_type logSNR --timesteps 100 --fid --seed 66 --exp "sample_cifar10"
```
--sample_type : ddim, ddim_second, ddim_third, ddim_fast, dpm_solver_12, dpm_solver_23, ddim_levy   
--skip_type : logSNR, uniform, quad, logSNR_quad   
--time_step : Number of Function Evaluation(NFE)
--fid : calculate fid score and IS score
--seed : seed_number for fixed random generation (to compare same sample with same random seed)
