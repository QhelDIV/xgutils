import os
import shutil
import argparse

from pytorch_lightning import Trainer as plTrainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint#, LearningRateLogger

from nnrecon.models import find_model_using_name
#from nnrecon import options
#from nnrecon.options import Options
from nnrecon.util import util
class Trainer():
    def default_opt(self):
        return dict(
            auto_lr_find=False,
        )
    def __init__(self, opt):
        if type(opt) is str:
            opt = Options(opt)
        self.opt = argparse.Namespace(**self.default_opt())
        if opt is not None:
            if type(opt) == dict:
                opt = argparse.Namespace(**opt)
            self.opt.__dict__.update(opt.__dict__)

        hparams = self.opt
        self.model = find_model_using_name(opt.model)(hparams)

        if opt.logger == 'tensorboard':
            logger = loggers.TensorBoardLogger(opt.logs_dir)
            version = logger.experiment.log_dir.split('_')[-1]
        elif opt.logger == 'wandb':
            logger = loggers.WandbLogger(name='testwandb',project='nnrecon')
            version=0
        checkpoint_callback = ModelCheckpoint(
            filepath=os.path.join(opt.checkpoints_dir,'v%s'%version+'_{epoch:03d}-{val_loss:.2e}'),
            save_top_k = -1,
            #save_last=True,
            verbose=True,
            monitor='val_loss',
            mode='min',
            period=1,
        )
        #lr_logger = LearningRateLogger()
        MainTrainerOptions=dict(
            max_epochs=opt.max_epochs,
            gpus=opt.gpus,
            auto_select_gpus=True, # very useful when gpu is in exclusive mode
            check_val_every_n_epoch=self.opt.check_val_every_n_epoch,
            auto_lr_find=self.opt.auto_lr_find,
            terminate_on_nan=True,
            progress_bar_refresh_rate=0,
            train_percent_check=1.,
        )
        OtherTrainerOptions=dict(
            logger=logger,
            distributed_backend=opt.distributed_backend,
            checkpoint_callback=checkpoint_callback,
            #callbacks=[lr_logger], # leave for pl version .9+
            resume_from_checkpoint=self.parse_ckpt(opt.resume_from),
        )
        #print(MainTrainerOptions, OtherTrainerOptions)
        self.trainer = \
            plTrainer(**MainTrainerOptions, 
                    **OtherTrainerOptions,
                )
    def train(self):
        opt = self.opt
        #shutil.copy(opt.optpath, os.path.join(opt.expr_dir,'config.yaml'))
        options.dump(opt.__dict__, os.path.join(opt.expr_dir,'config.yaml') )
        src_backup = os.path.join(opt.expr_dir,'src')
        util.makeArchive(options.src_dir, src_backup)
        
        self.trainer.fit(self.model)
        # backup after training
        if False:
            print('Finished training')
            save_path = os.path.join(opt.experiments_dir, opt.session_name)
            print('Save the experiment folder as %s'%(save_path))
            shutil.copytree(opt.expr_dir, save_path)
            shutil.copytree(options.src_dir,  os.path.join(save_path,'src'))
            print('Done.')
    def test(self):
        pass
    def parse_ckpt(self, ckpt):
        if self.opt.resume_from == '':
            return None
        if self.opt.resume_from == 'latest':
            ckpts = glob.glob( os.path.join(self.opt.checkpoints_dir, '*') )
            if len(ckpts)==0:
                return None
            latest_ckpt = max(ckpts, key=os.path.getctime)
            ckpt_path = latest_ckpt
        else:
            if ckpt[0]!='/': # if it is relative path
                ckpt_path = os.path.join(self.opt.checkpoints_dir, ckpt)
            else:
                ckpt_path = ckpt
        return ckpt_path
        
from nnrecon.util import qdaq
import sys
class ExpJob(qdaq.Job):
    def __init__(self, opt):
        if type(opt) is str:
            opt = optutil(opt)
        self.opt = opt
    def run(self, cuda_device_id):
        # Get cuda device
        PID = os.getpid()
        print(f"Name: {self.opt.expr_name} CUDA: {cuda_device_id}, PID: {PID}")
        self.opt.mkdirs()
        sys.stdout = open(os.path.join(self.opt.logs_dir, "stdout.out"), "w")
        #self.default_stdout = sys.__stdout__
        #sys.__stdout__ = sys.stdout
        if type(cuda_device_id) is int:
            cuda_device_id = [cuda_device_id]
        self.opt['pltrainer_opt']['gpus'] = cuda_device_id
        #os.environ["CUDA_VISIBLE_DEVICES"]=str(cuda_device_id)
        #print(self.opt.__dict__)
        print(self.pltrainer_opt.gpus)
        return True
        trainer = Trainer(self.opt)
        trainer.train()
        return True
import glob
import torch.multiprocessing as mp
if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser(add_help=False)
    #parser.add_argument('--option_path', required=True, type=str, help='path to project options')
    parser.add_argument('--opts', type=str, nargs='+', help='path to project options')
    parser.add_argument('--gpus', type=int, nargs='*', help='gpus to use')
    parsed=parser.parse_args()
    gpus = parsed.gpus
    if gpus is None or len(gpus)==0:
        gpus = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # opt = Options(parsed.option_path)
    # trainer = Trainer(opt)
    # trainer.train()
    
    #opts = glob.glob(parsed.opts)
    #print(opts)
    print(parsed.opts)
    # os.system("tmux new-sess -s {} -n {} -d '{}'".
    #                   format("test", 'test', 'bash'))    
    exps = [ExpJob(opt) for opt in parsed.opts]
    qdaq.start(exps, gpus)
