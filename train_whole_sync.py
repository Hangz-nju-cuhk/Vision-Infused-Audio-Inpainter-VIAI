import sys
import os
from os.path import dirname, join, expanduser
import matplotlib
from visdom_utils.visualizer import Visualizer
from tqdm import tqdm  # , trange
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import Models.Whole_Sync_inpainting_modify as Audio_model
import torch
from utils import util
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import numpy as np
import Data_loaders.audio_loader as data_loader_utils
import time
import Options_inpainting

hparams = Options_inpainting.Inpainting_Config()

visualizer = Visualizer(hparams)
global_step = 0
global_epoch = 0
global_test_step = 0
use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.benchmark = False
device = torch.device("cuda" if use_cuda else "cpu")


def train_loop(model, data_loaders, writer, checkpoint_dir=None):

    global global_step, global_epoch, global_test_step
    count = 0
    while global_epoch < hparams.nepochs:
        for phase, data_loader in data_loaders.items():
            train = (phase == "train")
            running_loss = 0.
            mel_running_loss = 0.
            audio_ebds = []
            image_ebds = []
            EmbeddingL2_loss = 0.
            test_evaluated = False

            for step, data in enumerate(data_loader):
                iter_start_time = time.time()
                # set input
                model.get_blank_space_length(global_step)
                model.set_inputs(data)
                # Whether to save eval (i.e., online decoding) result
                do_eval = False
                eval_dir = join(checkpoint_dir, "{}_eval".format(phase))
                os.makedirs(eval_dir, exist_ok=True)
                # Do eval per eval_interval for train
                if train and global_step > 0 \
                        and global_step % hparams.train_eval_interval == 0:
                    do_eval = True
                if not train and not test_evaluated \
                        and global_epoch % hparams.test_eval_epoch_interval == 0 and global_epoch > 0:
                    do_eval = True
                    test_evaluated = True

                if do_eval:
                    print("[{}] Eval at train step {}".format(phase, global_step))
                    #model.eval_model(global_step, eval_dir)
                    if count == 2:
                        model.eval_model_test(global_step, eval_dir)
                        count = 0
                    else:
                        count += 1

                # update global state
                if train:
                    model.train = 1
                    model.optimize_parameters(global_step)
                    global_step += 1
                else:
                    model.train = 0
                    with torch.no_grad():
                        model.test()
                    audio_ebds.append(util.to_np(model.mel_net_norm))
                    image_ebds.append(util.to_np(model.video_net_norm))
                    global_test_step += 1
                model.get_loss_items()

                if global_step > 0:

                    if global_step % hparams.display_freq == 0:
                       show = 1
                       visualizer.display_current_results(model.get_current_visuals(), global_epoch, step=global_step)
                       print('blank_length {}'.format(model.blank_length))
                    if train:
                        if global_step % hparams.print_freq == 0:
                            errors = model.get_current_errors()
                            t = (time.time() - iter_start_time) / hparams.batch_size
                            visualizer.print_current_errors(global_epoch, global_step, errors, t)
                            if hparams.display_id > 0:
                                visualizer.plot_current_errors(global_epoch, float(global_step) / len(data_loader), hparams, errors)

                if train and global_step > 0 and global_step % hparams.checkpoint_interval == 0:
                    model.save_inpainting_checkpoint(global_step, global_test_step, hparams.checkpoint_dir,
                                                     global_epoch, hparams=hparams)

                model.TF_writer(writer, step=global_step)
                if model.update_wavenet:
                    running_loss += model.reconstruct_loss_item
                else:
                    running_loss += 0
                EmbeddingL2_loss += model.EmbeddingL2_item
                mel_running_loss += model.loss_mel_L1_item
                model.del_no_need()
            # log per epoch
            averaged_loss = running_loss / len(data_loader)
            averaged_EmbeddingL2_loss = EmbeddingL2_loss / len(data_loader)
            averaged_loss_mel = mel_running_loss / len(data_loader)
            # print("{} loss at epoch {}: {}".format(phase, global_epoch, averaged_loss))
            writer.add_scalar(hparams.name + "_reconstruction_{} loss (per epoch)".format(phase),
                              averaged_loss, global_epoch)
            writer.add_scalar(hparams.name + "_mel_L1_{} loss (per epoch)".format(phase),
                              averaged_loss_mel, global_epoch)
            writer.add_scalar(hparams.name + "_{} EmbeddingL2loss (per epoch)".format(phase),
                              averaged_EmbeddingL2_loss, global_epoch)
            print("Step {} recontruct_loss [{}] Loss: {}".format(
                global_step, phase, averaged_loss))
            print("Step {} L1_loss [{}] Loss: {}".format(
                global_step, phase, averaged_loss_mel))
            print("{} Step {} [{}] EmbeddingL2_loss: {}".format(hparams.name,
                global_step, phase, averaged_EmbeddingL2_loss))
            if not train:
                audio_ebds = np.concatenate(audio_ebds, axis=0)

                image_ebds = np.concatenate(image_ebds, axis=0)

                metrics = util.L2retrieval(audio_ebds, image_ebds)
                metrics_inv = util.L2retrieval(image_ebds, audio_ebds)
                # -- print log
                writer.add_scalar('val_video_retrieval top1', metrics[0], global_epoch)
                writer.add_scalar('val_audio_retrieval top1', metrics_inv[0], global_epoch)
                info = 'Video Retrieval ({} samples): R@1: {:.2f}, R@5: {:.2f}, R@10: {:.2f}, R@50: {:.2f}, MedR: {:.1f}, MeanR: {:.1f}'
                info_inv = 'Audio Retrieval ({} samples): R@1: {:.2f}, R@5: {:.2f}, R@10: {:.2f}, R@50: {:.2f}, MedR: {:.1f}, MeanR: {:.1f}'
                print(info.format(audio_ebds.shape[0], *metrics))
                print(info_inv.format(image_ebds.shape[0], *metrics_inv))
        global_epoch += 1
        print("current_lr {}".format(model.current_lr))


if __name__ == "__main__":
    # args = docopt(__doc__)
    print()
    log_event_path = hparams.log_event_path
    checkpoint_path = hparams.resume_path
    os.makedirs(hparams.checkpoint_dir, exist_ok=True)

    # Dataloader setup
    data_loaders = data_loader_utils.get_data_loaders(hparams.data_root, hparams.speaker_id, test_shuffle=True)

    # Model
    model = Audio_model.AudioModel(hparams, device=device)

    # Load checkpoints
    if hparams.resume and checkpoint_path is not None:
        global_step, global_epoch, global_test_step = model.load_inpainting_checkpoint(checkpoint_path,
                                                                                       hparams.reset_optimizer)

    if hparams.load_pretrain:
        model.load_part_checkpoint()

    # Setup summary writer for tensorboard
    if log_event_path is None:
        log_event_path = "log/run-test-" + hparams.name + str(datetime.now()).replace(" ", "_")
    print("TensorBoard event log path: {}".format(log_event_path))
    writer = SummaryWriter(log_dir=log_event_path)

    # Train!
    try:
        train_loop(model, data_loaders, writer,
                   checkpoint_dir=hparams.checkpoint_dir)
    except KeyboardInterrupt:
        print("Interrupted!")
        pass
    finally:
        model.save_inpainting_checkpoint(global_step, global_test_step, hparams.checkpoint_dir, global_epoch, hparams=hparams)

    print("Finished")

    sys.exit(0)

