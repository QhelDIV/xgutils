import os
import sys
import glob
import torch
import wandb

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from xgutils import sysutil, nputil, ptutil, visutil

# Pytorch Lightning
from pytorch_lightning import Callback, LightningModule, Trainer
def dataset_generator(pl_module, dset, data_indices=[0,1,2], **get_kwargs):
    is_training = pl_module.training
    with torch.no_grad():
        pl_module.eval()
        for ind in data_indices:
            dataitem = dset.__getitem__(ind,**get_kwargs)
            batch = {}
            for key in dataitem:
                datakey = dataitem[key]
                if type(datakey) is not np.ndarray and type(datakey) is not torch.Tensor:
                    continue
                datakey = dataitem[key][None,...]
                if type(datakey) is np.ndarray:
                    datakey = torch.from_numpy(datakey)
                batch[key] = datakey.to(pl_module.device)
            yield batch
        if is_training:
            pl_module.train()
        else:
            pl_module.eval()

from abc import ABC, abstractmethod
from pytorch_lightning import Callback, LightningModule, Trainer

class FlyObj():
    def __init__(self, save_dir=None, load_dir=None, on_the_fly=True, data_processor=None):
        if data_processor is None:
            data_processor = self.dflt_data_processor
        self.__dict__.update(locals())
    def process_iter(self, input_iter):
        for name, input_data in input_iter:
            processed = self.load(name)
            if processed is None:
                processed = self.data_processor(input_data)
            yield name, processed

    def __call__(self, input_iter):
        process_iter = self.process_iter(input_iter)
        
        if self.on_the_fly==False:
            all_processed = list(process_iter)
            list(starmap(self.save, all_processed))
            for name, processed in all_processed:
                yield name, processed
        else:
            for name, processed in process_iter:
                self.save(name, processed)
                yield name, processed
    @staticmethod
    def dflt_data_processor(input_data):
        return input_data
    def save(self, name, data):
        if self.save_dir is not None:
            sysutil.mkdirs(self.save_dir)
            save_path = os.path.join(self.save_dir, f"{name}.npy")
            np.save(save_path, ptutil.ths2nps(data))
    def load(self, name):
        if self.load_dir is None:
            return None
        load_path = os.path.join(self.load_dir, f"{name}.npy")
        if os.path.exists(load_path) == False:
            return None
        loaded    = np.load(load_path,allow_pickle=True).item()
        return loaded
class ImageFlyObj(FlyObj):
    def save(self, name, imgs):
        if self.save_dir is not None:
            sysutil.mkdirs(self.save_dir)
            for key in imgs:
                save_path = os.path.join(self.save_dir, f"{name}_{key}.png")
                visutil.saveImg(save_path, imgs[key])
    def load(self, name):
        if self.load_dir is None:
            return None
        load_paths = os.path.join(self.load_dir, f"{name}_*.png")
        files = glob.glob(load_paths)
        files.sort(key=os.path.getmtime)
        if len(files) == 0:
            return None
        loaded = {}
        for imgf in files:
            key = "_".join(imgf[:-4].split("_")[1:])
            loaded[key] = visutil.readImg(imgf)
        return loaded
def dataset_generator(pl_module, dset, data_indices=[0,1,2], yield_ind=True, **get_kwargs):
    is_training = pl_module.training
    with torch.no_grad():
        pl_module.eval()
        for ind in data_indices:
            dataitem = dset.__getitem__(ind,**get_kwargs)
            batch = {}
            for key in dataitem:
                datakey = dataitem[key]
                if type(datakey) is not np.ndarray and type(datakey) is not torch.Tensor:
                    continue
                datakey = dataitem[key][None,...]
                if type(datakey) is np.ndarray:
                    datakey = torch.from_numpy(datakey)
                batch[key] = datakey.to(pl_module.device)
            if yield_ind==True:
                yield str(ind), batch
            else:
                yield batch
        if is_training:
            pl_module.train()
        else:
            pl_module.eval()
class VisCallback(Callback):
    def __init__(self,  visual_indices=[0,1,2,3,4,5], all_indices=False, \
                        every_n_epoch=3, no_sanity_check=False, \
                        data_dir = None, use_dloader=False):
        super().__init__()
        self.__dict__.update(locals())
        self.classname = self.__class__.__name__
        if self.data_dir is None:
            self.data_dir = f"/studio/nnrecon/temp/{self.classname}/"
    def process(self, pl_module, dloader, data_dir=None, visual_summary=False, \
                load_compute=False, load_visual=False, fly_compute=True):
        self.pl_module = pl_module
        if data_dir is None:
            data_dir = self.data_dir
        compute_dir = os.path.join(data_dir, "computed")
        cload_dir   = compute_dir if load_compute==True else None
        visual_dir  = os.path.join(data_dir, "visual")
        vload_dir   = visual_dir  if load_visual==True  else None
        dset = dloader.dataset
        if self.all_indices==True:
            self.visual_indices = list(range(len(dset)))
        if self.use_dloader==False:
            datagen      = dataset_generator(pl_module, dset, self.visual_indices)
        else:
            datagen      = dloader
        computegen   = FlyObj(data_processor=self.compute_batch, save_dir=compute_dir, load_dir=cload_dir, on_the_fly=fly_compute)
        visgen       = ImageFlyObj(data_processor=self.visualize_batch, save_dir=visual_dir, load_dir=vload_dir)
        imgsgen,imgs = visgen(computegen(datagen)), []
        for ind in sysutil.progbar(self.visual_indices):
            imgs.append(next(imgsgen))
        if visual_summary==True:
            summary_imgs = self.get_summary_imgs(imgs, zoomfac=.5)
        else:
            summary_imgs = None
        self.imgs, self.summary_imgs = imgs, summary_imgs
        
        #for l,img in visgen(computegen(datagen)):
        #    visutil.showImg(img["recon"])
        #visutil.showImg(self.summary_imgs[self.summary_imgs.keys()[0]]["image"])
        #return self.summary_imgs
    def compute_batch(batch):
        logits = batch["Ytg"].clone()
        logits[:]= torch.rand(logits.shape[0])
        return {"logits":logits, "batch":batch}
    def visualize_batch(computed):
        computed = ptutil.ths2nps(computed)
        batch = computed["batch"]
        Ytg = computed["logits"]
        Ytg = batch["Ytg"]
        vert, face = geoutil.array2mesh(Ytg, thresh=.5)
        img = fresnelvis.renderMeshCloud({"vert":vert,"face":face})
        return {"recon":img}
    def on_sanity_check_end(self, trainer, pl_module):
        print(f"\n{self.__class__.__name__} callback")
        if self.no_sanity_check:
            print("no_sanity_check is set to True, skipping...")
            return
        self.process(pl_module, pl_module.val_dataloader(), visual_summary=True)
        self.log_summary_images(trainer, pl_module, self.summary_imgs)
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule, *args, **kwargs) -> None:
        if trainer.current_epoch % self.every_n_epoch == self.every_n_epoch-1:
            print(f"\n{self.__class__.__name__} callback")
            self.process(pl_module, pl_module.val_dataloader(), visual_summary=True)
            self.log_summary_images(trainer, pl_module, self.summary_imgs)
    def get_summary_imgs(self, imgs, zoomfac=.5):
        all_images = []
        rows = len(imgs)
        for name, image_array in imgs:
            for img_name in image_array:
                img = image_array[img_name]
                all_images.append(img)
        summary = visutil.imageGrid(all_images, shape=(rows, -1), zoomfac=zoomfac)
        return {self.classname: {"caption":self.classname, "image":summary}}
    def log_summary_images(self, trainer, pl_module, summary_imgs, x_axis="epoch"):
        # wandb logger
        for key in summary_imgs:
            t = summary_imgs[key]
            title   = key
            caption = t["caption"]
            image   = t["image"]
            #log_image(trainer, title, caption, image, trainer.global_step)
            x_val = trainer.current_epoch if x_axis=="epoch" else trainer.global_step
            trainer.logger.experiment.log( \
                {title:[wandb.Image(image,caption=caption)], \
                    x_axis: x_val})


def null_logger(*args, **kwargs):
    return None
def get_debug_model(trainer, resume=False):
    if resume == True:
        pl_model, test_dloader = trainer.test_mode()
    else:
        pl_model, test_dloader = trainer.test_mode(resume_from=None)
    trainer.data_module.setup("train")
    train_dloader = trainer.data_module.train_dataloader(shuffle=False)

    train_dloader.num_workers = 0 # it will be very slow to invoke subprocesses (num_workers>0)
    test_dloader.num_workers  = 0
    return pl_model, train_dloader, test_dloader
def debug_model(trainer, resume=False, load_compute=False, load_visual=False):
    pl_model, train_dloader, test_dloader = get_debug_model(trainer, resume=resume)
    print("Test run train/val step")
    th_train_batch = ptutil.ths2device(next(iter(train_dloader)), "cuda")
    th_test_batch  = ptutil.ths2device(next(iter(test_dloader)), "cuda")
    origin_logger = pl_model.log
    try:
        pl_model.log = null_logger
        loss = pl_model.training_step(th_train_batch, batch_idx=0)
        print(f"Batch {0} train loss:", loss)
        loss = pl_model.validation_step(th_test_batch, batch_idx=0)
        print(f"Batch {0} val loss:",   loss)
    finally:
        pl_model.log = origin_logger

    print(trainer.callbacks)
    for callback in trainer.callbacks:
        if callback.__class__.__name__ == "ModelCheckpoint":
            continue
        if callback.__class__.__name__ == "ProgressBar":
            continue
        returns = callback.process(pl_model, test_dloader, visual_summary=False, load_compute=load_compute, load_visual=load_visual)
    print("Success")
    return pl_model, train_dloader, test_dloader
    
