import time
import subprocess
import torch
import pickle
import imageio
import json
import logging
import wandb
import os
import cv2
import collections
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

from matplotlib import animation

class Logger:
    def __init__(self, log_dir, context, modelname):
        '''
        :param log_dir (str): Path to logging folder for this experiment.
        :param context (str): Name of this particular logger instance, for example train / test.
        '''
        self.log_dir = log_dir
        self.context = context
        self.modelname = modelname
        self.log_path = os.path.join("../", self.log_dir, self.modelname, context + '.log')
        self.vis_dir = os.path.join("../", self.log_dir,self.modelname,  'visuals')
        self.npy_dir = os.path.join("../", self.log_dir,self.modelname,  'numpy')
        self.pkl_dir = os.path.join("../", self.log_dir, self.modelname, 'pickle')
        self.aud_dir = os.path.join("../", self.log_dir, self.modelname, 'audio')
        os.makedirs(self.log_dir, 0o777, exist_ok=True)
        os.makedirs(self.vis_dir, 0o777, exist_ok=True)
        os.makedirs(self.npy_dir, 0o777, exist_ok=True)
        os.makedirs(self.pkl_dir, 0o777, exist_ok=True)
        os.makedirs(self.aud_dir, 0o777, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(self.log_path),
                logging.StreamHandler()
            ]
        )

        self.scalar_memory = collections.defaultdict(list)
        self.scalar_memory_hist = dict()
        self.initialized = False

    def save_args(self, args):
        '''
        Records all parameters with which the script was called for reproducibility purposes.
        '''
        args_path = os.path.join(self.log_dir, 'args_' + self.context + '.txt')
        with open(args_path, 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    def init_wandb(self, project, args, networks, group='debug', name=None):
        '''
        Initializes the online dashboard, incorporating all PyTorch modules.
        '''
        if name is None:
            name = args.name
        wandb.init(project=project, group=group, config=args, name=name)
        if not isinstance(networks, collections.abc.Iterable):
            networks = [networks]
        for net in networks:
            if net is not None:
                wandb.watch(net)
        self.initialized = True
    
    def debug(self, *args):
        if args == ():
            args = ['']
        logging.debug(*args)

    def info(self, *args):
        if args == ():
            args = ['']
        logging.info(*args)

    def warning(self, *args):
        if args == ():
            args = ['']
        logging.warning(*args)

    def error(self, *args):
        if args == ():
            args = ['']
        logging.error(*args)

    def critical(self, *args):
        if args == ():
            args = ['']
        logging.critical(*args)

    def exception(self, *args):
        if args == ():
            args = ['']
        logging.exception(*args)

    def line_plot(self,xs,ys,step):
        if self.initialized: 
            wandb.log({"indices" + str(step) : wandb.plot.line_series(
                        xs=xs, 
                        ys=ys,
                        keys=["input", "output", "pred"],
                        title="Codebook values",
                        xname="indices")},step=step)
            
    def report_scalar(self, key, value, step=None, remember=True, commit_histogram=False):
        '''
        Logs a single named value associated with a step.
        If commit_histogram, actual logging is deferred until commit_scalars() is called.
        '''
        if not remember and not commit_histogram:
            if self.initialized:
                wandb.log({key: value}, step=step)
            else:
                self.debug(str(key) + ': ' + str(value))
        else:
            if isinstance(value,list):
                self.scalar_memory[key].extend(value)
            else:
                self.scalar_memory[key].append(value)
            self.scalar_memory_hist[key] = commit_histogram

    def mean_scalar(self, key, value, step=None, remember=False, commit_histogram=False):
        '''
        Logs a single named value associated with a step.
        If commit_histogram, actual logging is deferred until commit_scalars() is called.
        '''
        for key in self.scalar_memory.keys():
            wandb.log({key: torch.mean(self.scalar_memory[key])}, step=step)

    def commit_scalars(self, keys=None, step=None):
        '''
        Aggregates a bunch of report_scalar() calls for one or more named sets of values and records
        their histograms, i.e. statistical properties.
        '''
        if keys is None:
            keys = list(self.scalar_memory.keys())
        returnval = 0
        for key in keys:
            if len(self.scalar_memory[key]) == 0:
                continue

            value = np.mean(self.scalar_memory[key])
            if self.initialized:
                if self.scalar_memory_hist[key]:
                    wandb.log({key: wandb.Histogram(np.array(self.scalar_memory[key]))}, step=step)
                else:
                    print("here")
                    wandb.log({key: value}, step=step)

            else:
                self.debug(str(key) + ': ' + str(value))
            if key == 'val/loss_total':
                returnval=value
            self.scalar_memory[key].clear()
        return returnval
    
    def report_histogram(self, key, value, step=None):
        '''
        Directly logs the statistical properties of a named iterable value, such as a list of
        numbers.
        '''
        if self.initialized:
            wandb.log({key: wandb.Histogram(value)}, step=step)

    def save_image(self, image, cur_step=None, total_step=None, epoch=None, file_name=None, online_name=None, caption=None):
        '''
        Records a single image to a file in visuals and/or the online dashboard.
        '''
        logdir = os.path.join(self.vis_dir,str(epoch),str(cur_step))
        os.makedirs(logdir, 0o777, exist_ok=True)
        if image.dtype == np.float32:
            image = (image * 255.0).astype(np.uint8)
        if file_name is not None:
            plt.figure()
            plt.imshow(image,vmin=0, vmax=255)
            plt.title(caption)
            plt.savefig(os.path.join(logdir, file_name),bbox_inches = 'tight')
        """if online_name is not None and self.initialized:
            wandb.log({online_name: wandb.Image(image)}, step=total_step)"""

    def save_audio(self, audio,  cur_step=None, total_step=None, epoch=None, file_name=None, online_name=None, caption=None):
        '''
        Records a single image to a file in visuals and/or the online dashboard.
        '''
        logdir = os.path.join(self.aud_dir,str(epoch),str(cur_step))
        os.makedirs(logdir, 0o777, exist_ok=True)
        if file_name is not None:
            sf.write(os.path.join(logdir, file_name), audio, 22050)
        """if online_name is not None and self.initialized:
            if caption is not None:
                #pdb.set_trace()
                wandb.log({online_name: wandb.Audio(audio, sample_rate=22050, caption=caption)}, step=total_step)
            else:
                wandb.log({online_name: wandb.Audio(audio, sample_rate=22050)}, step=step)"""

    def save_video(self, frames, step=None, file_name=None, online_name=None, fps=6):
        '''
        Records a single set of frames as a video to a file in visuals and/or the online dashboard.
        '''
        # Duplicate last frame for better visibility.
        last_frame = frames[len(frames) - 1:len(frames)]
        frames = np.concatenate([frames, last_frame], axis=0)
        if frames.dtype == np.float32:
            frames = (frames * 255.0).astype(np.uint8)
        if file_name is not None:
            file_path = os.path.join(self.vis_dir, file_name)
            imageio.mimwrite(file_path, frames, fps=fps)
        if online_name is not None and self.initialized:
            # This is bugged in wandb:
            # wandb.log({online_name: wandb.Video(frames, fps=fps, format='gif')}, step=step)
            assert file_name is not None
            wandb.log({online_name: wandb.Video(file_path, fps=fps, format='gif')}, step=step)

    def save_gallery(self, frames, step=None, file_name=None, online_name=None):
        '''
        Records a single set of frames as a gallery image to a file in visuals and/or the online
        dashboard.
        '''
        if frames.shape[-1] > 3:  # Grayscale: (..., H, W).
            arrangement = frames.shape[:-2]
        else:  # RGB: (..., H, W, 1/3).
            arrangement = frames.shape[:-3]
        if len(arrangement) == 1:  # (A, H, W, 1/3?).
            gallery = np.concatenate(frames, axis=1)  # (H, A*W, 1/3?).
        elif len(arrangement) == 2:  # (A, B, H, W, 1/3?).
            gallery = np.concatenate(frames, axis=1)  # (B, A*H, W, 1/3?).
            gallery = np.concatenate(gallery, axis=1)  # (A*H, B*W, 1/3?).
        else:
            raise ValueError('Too many dimensions to create a gallery.')
        if gallery.dtype == np.float32:
            gallery = (gallery * 255.0).astype(np.uint8)
        if file_name is not None:
            plt.imsave(os.path.join(self.vis_dir, file_name), gallery)
        if online_name is not None and self.initialized:
            wandb.log({online_name: wandb.Image(gallery)}, step=step)

    def save_numpy(self, array, file_name, step=None, folder=None):
        '''
        Stores a numpy object locally, either in pickle or a chosen directory.
        '''
        if folder is None:
            dst_dp = self.npy_dir
        else:
            dst_dp = os.path.join(self.log_dir, folder)
            os.makedirs(dst_dp, exist_ok=True)
        np.save(os.path.join(dst_dp, file_name), array)

    def save_pickle(self, obj, file_name, step=None, folder=None):
        '''
        Stores a pickle object locally, either in pickle or a chosen directory.
        '''
        if folder is None:
            dst_dp = self.pkl_dir
        else:
            dst_dp = os.path.join(self.log_dir, folder)
            os.makedirs(dst_dp, exist_ok=True)
        dst_fp = os.path.join(dst_dp, file_name)
        with open(dst_fp, 'wb') as f:
            pickle.dump(obj, f)

class CustomLogger(Logger):
    def __init__(self, args, argstwo, context):
        super().__init__(args.log_path, context, args.name)
        self.args = args
        self.argstwo = argstwo
        
        self.step_interval = 1
        self.num_exemplars = 4 # To increase simultaneous examples in wandb during train / val
        self.maxemg = 300
    
    def perspective_projection(self, points, rotation, translation, focal_length, camera_center):
        batch_size = points.shape[0]
        K = torch.zeros([batch_size, 3, 3], device=points.device)
        K[:, 0, 0] = focal_length
        K[:, 1, 1] = focal_length
        K[:, 2, 2] = 1.
        K[:, :-1, -1] = camera_center

        # Transform points
        points = torch.einsum('bij, bkj -> bki', rotation.repeat(points.shape[0], 1, 1), points)
        points = points + translation.unsqueeze(1)

        # Apply perspective distortion
        projected_points = points / points[:, :, -1].unsqueeze(-1)

        # Apply camera intrinsics
        projected_points = torch.einsum('bij, bkj -> bki', K, projected_points)

        return projected_points[:, :, :-1], points
    
    def convert_pare_to_full_img_cam(self, pare_cam, bbox_height, bbox_center, img_w, img_h, focal_length, crop_res=224):
        s, tx, ty = pare_cam[:, 0], pare_cam[:, 1], pare_cam[:, 2]
        res = 224
        r = bbox_height / res
        tz = 2 * focal_length / res

        cx = 2 * (bbox_center[:, 0] - (img_w / 2.)) / (s * bbox_height)
        cy = 2 * (bbox_center[:, 1] - (img_h / 2.)) / (s * bbox_height)

        cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)

        return cam_t

    def plot_skel(self, keypoints, pred_keypoints, img, current_path, flag, markersize=3, linewidth=2, alpha=0.7):
        plt.figure()
        keypoints = keypoints[:25]
        pred_keypoints = pred_keypoints[:25]
        plt.imshow(img)

        limb_seq = [[17, 15], [15, 0], [0, 16], [16, 18],
                    [0, 1], [1, 2], [2, 3], [3, 4], 
                            [1, 5], [5, 6], [6, 7], 
                            [1, 8], [8, 9], [9, 10], [10, 24],
                                    [8, 12], [12, 13], [13, 14],
                    [24, 22], [22, 23],
                    [24, 24],
                    [14, 19], [19, 20],
                    [14, 21]]
        
        pred_color = [0, 252, 255]
        gt_color = [255, 21, 0]
        
        # plot limb & keypoints
        if flag == 'pred':
            for vertices in limb_seq: # vertices: limb의 양끝 관절 노드 인덱스, color: rgb 픽셀값
                if pred_keypoints[vertices[0]].mean() != 0 and pred_keypoints[vertices[1]].mean() != 0:
                    plt.plot([pred_keypoints[vertices[0]][0], pred_keypoints[vertices[1]][0]],
                             [pred_keypoints[vertices[0]][1], pred_keypoints[vertices[1]][1]], linewidth=linewidth,
                             color=[j / 255 for j in pred_color], alpha=alpha)
            
            for k in range(len(pred_keypoints)):
                if pred_keypoints.mean() != 0:
                    plt.plot(pred_keypoints[k][0], pred_keypoints[k][1], 'o', markersize=markersize,
                             color=[j / 255 for j in pred_color], alpha=alpha)
                    
        for vertices in limb_seq: # vertices: limb의 양끝 관절 노드 인덱스, color: rgb 픽셀값
            if keypoints[vertices[0]].mean() != 0 and keypoints[vertices[1]].mean() != 0:
                plt.plot([keypoints[vertices[0]][0], keypoints[vertices[1]][0]],
                         [keypoints[vertices[0]][1], keypoints[vertices[1]][1]], linewidth=linewidth,
                         color=[j / 255 for j in gt_color], alpha=alpha)
        
        for k in range(len(keypoints)):
                if keypoints.mean() != 0:
                    plt.plot(keypoints[k][0], keypoints[k][1], 'o', markersize=markersize,
                             color=[j / 255 for j in gt_color], alpha=alpha)
                    
        plt.axis('off')
        plt.savefig(current_path, bbox_inches='tight', pad_inches=0)

    def visualize_video(self, frames, twodskeleton, twodpred, current_path, flag):
        if not os.path.isdir(current_path):
            os.makedirs(current_path, 0o777)
        if not os.path.isdir(current_path + 'skeletonimgs'):
            os.makedirs(current_path + '/skeletonimgs', 0o777)
        if not os.path.isdidr(current_path + '/' + flag + '_frames'):
            os.makedirs(current_path + '/' + flag + '_frames', 0o777)
        
        for i in range(30):
            name = frames[i].split('/')[-1]
            img = cv2.imread(frames[i])
            img = (img * 1.0).astype('int')

            current_skeleton = twodskeleton[i].cpu().numpy()
            pred_skeleton = twodpred[i].cpu().numpy()
            blur_img = img[int(current_skeleton[0][1]) - 100: int(current_skeleton[0][1]) + 100, 
                           int(current_skeleton[0][0]) - 100: int(current_skeleton[0][0]) + 100]
            
            if (int(current_skeleton[0][1]) - 100 > 0 
                and int(current_skeleton[0][1] + 100) < img.shape[0]
                and int(current_skeleton[0][0] - 100 > 0)
                and int(current_skeleton[0][0]) + 100 < img.shape[1]):
                
                blurred_part = cv2.blur(blur_img, ksize=(50, 50))
                img[int(current_skeleton[0][1]) - 100: int(current_skeleton[0][1]) + 100,
                    int(current_skeleton[0][0]) - 100: int(current_skeleton[0][0]) + 100] = blurred_part
                
            self.plot_skel(current_skeleton, pred_skeleton, img[..., ::-1], current_path=f'{current_path}/{flag}_frames/{str(i).zfill(6)}.png', flag=flag)
        
        command = ['ffmpeg', '-framerate', '10', '-i',f'{current_path}' + "/" + flag + '_frames/' + '%06d.png',  '-c:v', 'libx264','-vf' ,'pad=ceil(iw/2)*2:ceil(ih/2)*2',
        '-pix_fmt', 'yuv420p', f'{current_path}/out' + flag + '.mp4']

        print(f'Running \"{" ".join(command)}\"')
        #pdb.set_trace()
        subprocess.call(command)

        return current_path

    def visualize_skeleton(self, threedskeleton, bboxes, predcam, cur_step,ex,phase,current_path, current_path_parent):

        proj =5000.0
        translation = self.convert_pare_to_full_img_cam(predcam,bboxes[:,2:3],bboxes[:,:2],1080,1920,proj)
        #pdb.set_trace()
        twodkpts, skeleton = self.perspective_projection(threedskeleton.detach().cpu(), torch.unsqueeze(torch.eye(3),dim=0), translation[0].float(),torch.tensor([[proj]]), torch.unsqueeze(torch.tensor([1080.0/2, 1920.0/2]),dim=0))
        #twodkpts=twodkpts[0]
        #skeleton=skeleton[0]

        colors=['b','b','r','r','r','g','g','g','y','r','r','r','g','g','g','b','b','b','b','g','g','g','r',
        'r','r','r','r','r','g','g','g','r','r','r','l','l','l','b','b','y','y','y','b','b','b',
        'b','b','b','b']

        fig = plt.figure(figsize=(15,4.8))
        ax = fig.add_subplot(121,projection='3d')
        ax2 = fig.add_subplot(122,projection='3d')
        #ax3 = fig.add_subplot(133)
        #current_path = "../../www-data/mia/muscleresults/" + self.args.name + '_' + phase + '_viz_digitized/'  + movie + "/" + str(cur_step) + "/" + str(ex)
        #current_path = "../../www-data/mia/muscleresults/" + self.args.name + '_' + phase + '_viz_digitized/' + movie + "/" + str(cur_step) + "/" + str(j)
        limb_seq = [([17, 15], [238, 0, 255]),
                    ([15, 0], [255, 0, 166]),
                    ([0, 16], [144, 0, 255]),
                    ([16, 18], [65, 0, 255]),
                    ([0, 1], [255, 0, 59]),
                    ([1, 2], [255, 77, 0]),
                    ([2, 3], [247, 155, 0]),
                    ([3, 4], [255, 255, 0]),
                    ([1, 5], [158, 245, 0]),
                    ([5, 6], [93, 255, 0]),
                    ([6, 7], [0, 255, 0]),
                    ([1, 8], [255, 21, 0]),
                    ([8, 9], [6, 255, 0]),
                    ([9, 10], [0, 255, 117]),
                    # ([10, 11]], [0, 252, 255]),  # See comment above
                    ([10, 24], [0, 252, 255]),
                    ([8, 12], [0, 140, 255]),
                    ([12, 13], [0, 68, 255]),
                    ([13, 14], [0, 14, 255]),
                    # ([11, 22], [0, 252, 255]),
                    # ([11, 24], [0, 252, 255]),
                    ([24, 22], [0, 252, 255]),
                    ([24, 24], [0, 252, 255]),
                    ([22, 23], [0, 252, 255]),
                    ([14, 19], [0, 14, 255]),
                    ([14, 21], [0, 14, 255]),
                    ([19, 20], [0, 14, 255])]
        
        colors_vertices = {0: limb_seq[4][1],
                        1: limb_seq[11][1],
                        2: limb_seq[5][1],
                        3: limb_seq[6][1],
                        4: limb_seq[7][1],
                        5: limb_seq[8][1],
                        6: limb_seq[9][1],
                        7: limb_seq[10][1],
                        8: limb_seq[11][1],
                        9: limb_seq[12][1],
                        10: limb_seq[13][1],
                        11: limb_seq[14][1],
                        12: limb_seq[15][1],
                        13: limb_seq[16][1],
                        14: limb_seq[17][1],
                        15: limb_seq[1][1],
                        16: limb_seq[2][1],
                        17: limb_seq[0][1],
                        18: limb_seq[3][1],
                        19: limb_seq[21][1],
                        20: limb_seq[23][1],
                        21: limb_seq[22][1],
                        22: limb_seq[18][1],
                        23: limb_seq[20][1],
                        24: limb_seq[19][1]}
        for i in range(len(threedskeleton)):
            cur = time.time()
            maxvaly,minvaly = torch.max(skeleton[:,:,2]),torch.min(skeleton[:,:,2])
            maxvalz,minvalz = torch.max(skeleton[:,:,1]),torch.min(skeleton[:,:,1])
            def thistakesalongtime():
                print(i,"i")
                curskeleton = skeleton[i].cpu().numpy()
                
                cur = time.time()
                for j in range(curskeleton.shape[0]):
                    #print(j,"j")
                #plt.figure()
                    c = colors[j]
                    if c == 'b':
                        newc = 'blue' #'#ff0000'
                    elif c=='r':
                        newc= 'red' #'#0000ff'
                    else:
                        newc = '#0f0f0f'
                    if j == 25 or j==30:
                        newc = 'yellow'
                    ax.scatter3D(curskeleton[j][0],curskeleton[j][2],curskeleton[j][1],c=newc)
                    ax2.scatter3D(curskeleton[j][0],curskeleton[j][2],curskeleton[j][1],c=newc)
                for vertices, color in limb_seq:
                    ax.plot3D([curskeleton[vertices[0]][0], curskeleton[vertices[1]][0]],
                            [curskeleton[vertices[0]][1], curskeleton[vertices[1]][1]],
                            [curskeleton[vertices[0]][2], curskeleton[vertices[1]][2]], 
                            linewidth=3,
                            color=[j / 255 for j in colors_vertices[8]], alpha=0.7)
            #
                #ax.view_init(-75, -90,90)
                    #ax.invert_yaxis()
                curtwo = time.time()
                print(curtwo-cur)
                cur = curtwo
                ax.set_xlim3d([-1.0, 2.0])
                ax.set_zlim3d([minvalz, maxvalz])
                ax.set_ylim3d([minvaly, maxvaly])
                curtwo = time.time()
                print(curtwo-cur)
                cur = curtwo
                ax.invert_zaxis()
                ax2.set_xlim3d([-1.5, 2.0])
                ax2.set_zlim3d([minvalz, maxvalz])
                ax2.set_ylim3d([minvaly, maxvaly])
                ax2.invert_zaxis()
                curtwo = time.time()
                print(curtwo-cur)
                cur = curtwo

                ax.set_xlabel("x")
                ax.set_ylabel("z")
                ax.set_zlabel("y")
                
                ax.view_init(0,180)

                ax2.set_xlabel("x")
                ax2.set_ylabel("z")
                ax2.set_zlabel("y")
                ax2.view_init(0,-45)
                curtwo = time.time()
                print(curtwo-cur)
                cur = curtwo
                plt.savefig(current_path + "/3dskeleton" + str(i).zfill(6) + ".png")

                if i == len(threedskeleton)-1:
                    time.sleep(0.5)
                    command = ['ffmpeg', '-framerate', '10', '-i',f'{current_path}/3dskeleton%06d.png',  '-c:v', 'libx264','-pix_fmt', 'yuv420p', f'{current_path}/out3dskeleton.mp4']
                    print(f'Running \"{" ".join(command)}\"')
                    subprocess.call(command)
            proc = mp.Process(target=thistakesalongtime) # , args=(1, 2))
            proc.start()

            # plt.savefig(current_path + "/3dskeleton" + str(i).zfill(6) + ".png")
            curtwo = time.time()
            print(curtwo-cur, "here")
            cur = curtwo

        return current_path

    def animate(self, list_of_data, list_of_pred_data, labels, part, trialnum, current_path, epoch):
            t = np.linspace(0, len(list_of_data[0])/10.0, len(list_of_data[0]))
            numDataPoints = len(t)
            colors = ['g','c']
            colorspred = ['r','b']

            def animate_func(num):
                ax.clear()  # Clear the figure to update the line, point,   
                for i,limb in enumerate(list_of_data):
                    ax.plot(t[:num],limb[:num], c=colors[i], label=labels[i])
                for i,limb in enumerate(list_of_pred_data):
                    ax.plot(t[:num],limb[:num], c=colorspred[i], label=labels[i] + "pred")
                #ax.plot(t[:num],dataSetlefttricep[:num], c='red', label='right tricep')
                ax.legend(loc="upper left")

                # Adding Figure Labels
                ax.set_title('Trajectories of ' + part + ' \nTime = ' + str(np.round(t[num], decimals=2)) + ' sec')
                ax.set_xlabel('x')
                ax.set_ylim([0, self.maxemg])
                
            fig, ax = plt.subplots()
            
            line_ani = animation.FuncAnimation(fig, animate_func, interval=100, frames=numDataPoints)
            print("saving_animation")
            line_ani.save(current_path + "/" + str(part) + '_emg.mp4')

    def handle_train_step(self, epoch, phase, current_step, total_step, steps_per_epoch, data_retval, model_retval, loss_retval):
        if current_step % self. step_interval == 0:
            j = 0
            exemplar_idx = (current_step // self.step_interval) % self.num_exemplars # exemplar_idx는 0 ~ num_exemplars 내에서 순회

            total_loss = loss_retval['total']
            frame_list = [data_retval['frame_paths'][i][j] for i in range(len(data_retval['frame_paths']))]

            current_path = self.visualize_video(frame_list, data_retval['2dskeleton'][j], current_step, j, phase,frame_list[0].split("/")[-2])
            threedskeleton = data_retval['3dskeleton'][j]
            current_path = self.visualize_skeleton(threedskeleton.detach().cpu(), data_retval['bboxes'][j], data_retval['predcam'][j], current_step, j, phase, frame_list[0].split("/")[-2])

            if self.clasif:
                values = data_retval['bined_left_quad'][j] - 1
                bins = data_retval['bins'][j]
                gt_values = torch.index_select(bins.cpu(), 0, values.cpu())
                pred_values = model_retval['emg_output'][j][0].cpu()
                pred_values[pred_values > 1.0] = 1.0
                self.animate([gt_values.numpy()], [pred_values.detach().numpy()], ['left_quad'], 'leftleg', 2, current_path, epoch)
            else:
                gt_values = data_retval['left_quad'][j]
                gt_values[gt_values > 1.0] = 1.0
                pred_values = model_retval['emg_output'][j][0].cpu()
                pred_values[pred_values > 1.0] = 1.0
                self.animate([gt_values.numpy()], [pred_values.detach().numpy()], ['left_quad'], 'leftleg', 2, current_path, epoch)
            
            command = ['ffmpeg', '-i', f'{current_path}/out.mp4', '-i',f'{current_path}/out3dskeleton.mp4',  '-i', f'{current_path}/epoch_159_leftleg_emg.mp4','-filter_complex',
            'hstack=inputs=3', f'{current_path}/total.mp4']
            print(f'Running \"{" ".join(command)}\"')
            subprocess.call(command)
            self.info(f'[Step {current_step} / {steps_per_epoch}]  '
                    f'total_loss: {total_loss:.3f}  ')
            
    def handle_val_step(self, device, epoch, phase, cur_step, total_step, steps_per_epoch, data_retval, model_retval, loss_retval, flag):
        if cur_step % self.step_interval == 0:
            exemplar_idx = (cur_step // self.step_interval) % self.num_exemplars
            
            for j in range(data_retval['2dskeleton'].shape[0]):
                frame_list = [data_retval['frame_paths'][i][j] for i in range(len(data_retval['frame_paths']))]
                filepath = data_retval['filepath']
                movie = filepath.split("/")[-2] + filepath.split("/")[-1]
                
                current_path = "muscleresults/" + self.args.name + '_' + phase + '_viz_digitized/' + movie + "/" + str(cur_step) + "/" + str(j)
                current_path_parent = "muscleresults/" + self.args.name + '_' + phase + '_viz_digitized/' + movie + "/"
                
                if not os.path.isdir(current_path):
                    os.makedirs(current_path, 0o777)
                if not os.path.isdir(current_path_parent):
                    os.makedirs(current_path_parent, 0o777)

                with open(current_path_parent + "file_list.txt","a") as f:
                    f.write(current_path + " \n")

                if self.args.predemg!= 'True' and self.args.threed=='False':
                    pred_pose = model_retval['pose_output'][0].reshape(30,25,2)
                    pred_pose[:,:,0] = pred_pose[:,:,0]*1080
                    pred_pose[:,:,1] = pred_pose[:,:,1]*1920

                    temp = self.visualize_video(frame_list,data_retval['2dskeleton'][j][:,:25,:],pred_pose[:,:25,:],cur_step,j,phase,current_path, 'pred')
                    #threedskeleton = data_retval['3dskeleton'][0][:,:25,:].to(device)
                elif self.args.predemg != 'True' and self.args.threed=='True':

                    threedskeleton = model_retval['pose_output'][0].reshape(30,25,3)
                    bboxes = data_retval['bboxes'][:,:30,:]
                    predcam = data_retval['predcam'][:,:30,:]
                    proj = 5000.0
                    #pdb.set_trace()
                    
                    height= bboxes[:,:,2:3].reshape(bboxes.shape[0]*bboxes.shape[1]).to(device)
                    center = bboxes[:,:,:2].reshape(bboxes.shape[0]*bboxes.shape[1],-1).to(device)
                    focal=torch.tensor([[proj]]).to(device).repeat(height.shape[0],1)
                    predcamelong = predcam.reshape(predcam.shape[0]*predcam.shape[1],-1).to(device)
                    #pdb.set_trace()
                    translation = self.convert_pare_to_full_img_cam(predcamelong,height,center,1080,1920,focal[:,0])
                    #pdb.set_trace()
                    #reshapethreed= threedskeleton.reshape(threedskeleton.shape[0]*threedskeleton.shape[1],threedskeleton.shape[2],threedskeleton.shape[3])
                    rotation=torch.unsqueeze(torch.eye(3),dim=0).repeat(1,1,1).to(device)
                    focal=torch.tensor([[proj]]).to(device).repeat(translation.shape[0],1)
                    imgdimgs=torch.unsqueeze(torch.tensor([1080.0/2, 1920.0/2]),dim=0).repeat(1,1).to(device)
                    #pdb.set_trace()
                    twodkpts, skeleton = self.perspective_projection(threedskeleton, rotation, translation.float(),focal[:,0], imgdimgs)
                    #pdb.set_trace()
                    #current[:,:,0] = current[:,:,0]*1080
                    #current[:,:,1] = current[:,:,1]*1920
                    #pdb.set_trace()
                    np.save(current_path + "/predpose2d" + str(cur_step) + ".npy",twodkpts.detach().cpu().numpy())
                    np.save(current_path + "/gtpose2d" + str(cur_step) + ".npy",data_retval['2dskeleton'][j][:,:25,:].cpu().numpy())
                    np.save(current_path + "/gtpose3d" + str(cur_step) + ".npy",data_retval['3dskeleton'][j][:,:25,:].cpu().numpy())
                    np.save(current_path + "/predpose3d" + str(cur_step) + ".npy",threedskeleton.detach().cpu().numpy())
                    #current_path = self.visualize_video(frame_list,data_retval['2dskeleton'][j][:,:25,:],twodkpts,cur_step,j,phase,current_path, 'pred')
                    #current_path = self.visualize_video(frame_list,data_retval['2dskeleton'][j][:,:25,:],twodkpts,cur_step,j,phase,current_path, 'gt')
                    #pdb.set_trace()
                    temp = self.visualize_video(frame_list,data_retval['2dskeleton'][j][:,:25,:],twodkpts[:,:25,:],cur_step,j,phase,current_path, 'pred')
                    temp = self.visualize_video(frame_list,data_retval['2dskeleton'][j][:,:25,:],twodkpts[:,:25,:],cur_step,j,phase,current_path, 'gt')
                    current_path = self.visualize_skeleton(threedskeleton,data_retval['bboxes'][j],data_retval['predcam'][j],cur_step,j,phase,current_path, current_path_parent)
                    #pdb.set_trace()
                else:
                    frame_list = [data_retval['frame_paths'][i][0] for i in range(len(data_retval['frame_paths']))]
                    filepath = data_retval['filepath']
                    movie = filepath.split("/")[-2] + filepath.split("/")[-1]#frame_list[0].split("/")[-4] + "/" + frame_list[0].split("/")[-3]
                    #movie = frame_list[0].split("/")[-4] + "/" + frame_list[0].split("/")[-3]
                    #current_path = "../../../../local/vondrick/mia/muscleresults/" + self.args.name + '_' + phase + '_viz_digitized/' + movie + "/" + str(cur_step) + "/" + str(0) + "/gt" + flag + "/"
                    current_path = "muscleresults/" + self.args.name + '_' + phase + '_viz_digitized/' + movie + "/" + str(cur_step) + "/" + str(0) + "/gt" + flag + "/"

                    themax = torch.unsqueeze(data_retval['max'],dim=2)
                    themin = torch.unsqueeze(data_retval['min'],dim=2)
                    #pdb.set_trace()
                    normalized_gt_emg = (model_retval['emg_gt'].detach().cpu()-themin)/themax
                    normalized_pred_emg = (model_retval['emg_output'].detach().cpu()-themin)/themax
                    #normalized_pred_emg[normalized_pred_emg<0] = 0.0
                    #normalized_pred_emg[normalized_pred_emg>1] = 1.0
                    #pdb.set_trace()
                    if not os.path.isdir(current_path):
                        os.makedirs(current_path, 0o777)
                    #pdb.set_trace()
                    # self.visualize_mesh_activation(data_retval['2dskeleton'][j],data_retval['verts'][j],data_retval['origcam'][j],frame_list,normalized_gt_emg[j],normalized_pred_emg[j],current_path)
                    #threedskeleton = model_retval['pose_output'][j].reshape(30,25,3)
                    #threedskeleton = data_retval['3dskeleton'][0][:,:25,:].to(device)
                    #current_path = self.visualize_skeleton(threedskeleton,data_retval['bboxes'][j],data_retval['predcam'][j],cur_step,j,phase,current_path, current_path_parent)
                    #if self.args.predemg == 'True':
                    #    self.visualize_mesh_activation(data_retval['2dskeleton'][j],data_retval['verts'][j],data_retval['orig_cam'][j],frame_list, data_retval['emg_values'][j],model_retval['emg_output'][j].cpu().detach(),current_path)"""
    
                gtnp = data_retval['emg_values'].detach().cpu().numpy()
                if self.args.predemg == 'True':
                    prednp = model_retval['emg_output'].detach().cpu().numpy()
                else:
                    prednp = gtnp
                
                np.save(current_path + "/gtnp" + str(cur_step) + ".npy",gtnp)
                np.save(current_path + "/prednp" + str(cur_step) + ".npy",prednp)
                rangeofmuscles=['LeftQuad','LeftHamstring','LeftLateral','LeftBicep','RightQuad','RightHamstring','RightLateral','RightBicep']
                """for i in range(model_retval['emg_gt'].shape[1]):
                    gt_values = model_retval['emg_gt'][j,i,:].cpu()*100
                    #gt_values[gt_values>100.0] = 100.0
                    pred_values = model_retval['emg_output'][j][i].cpu()*100.0
                    #pdb.set_trace()
                    self.animate([gt_values.numpy()],[pred_values.detach().numpy()],[rangeofmuscles[i]],rangeofmuscles[i],2,current_path,epoch)"""
                ###DEBUG
                for i in range(model_retval['emg_gt'].shape[1]):
                    gt_values = model_retval['emg_gt'][j,i,:].cpu()
                    #gt_values = data_retval['old_emg_values'][j,i,:].cpu()
                    if self.args.predemg == 'True':
                        #pred_values = data_retval['emg_values'][j][i].cpu()
                        pred_values = model_retval['emg_output'][j][i].cpu()
                    else:
                        #pred_values = data_retval['emg_values'][j][i].cpu()
                        pred_values = gt_values
                    #pdb.set_trace()
                    self.animate([gt_values.numpy()],[pred_values.detach().numpy()],[rangeofmuscles[i]],rangeofmuscles[i],2,current_path,epoch)
            

                    # Print metrics in console.
                    """command = ['ffmpeg', '-i', f'{current_path}/out.mp4', '-i',f'{current_path}/out3dskeleton.mp4',  '-i', f'{current_path}/epoch_206_leftbicep_emg.mp4',
                    'i', f'{current_path}/epoch_206_rightquad_emg.mp4','-filter_complex',
                    'hstack=inputs=4', f'{current_path}/total.mp4']
                    print(f'Running \"{" ".join(command)}\"')
                    subprocess.call(command)"""

    def epoch_finished(self, epoch):
        returnval = self.commit_scalars(step=epoch)
        
        return returnval

    def handle_test_step(self, cur_step, num_steps, data_retval, inference_retval):

        psnr = inference_retval['psnr']

        # Print metrics in console.
        self.info(f'[Step {cur_step} / {num_steps}]  '
                    f'psnr: {psnr.mean():.2f} ± {psnr.std():.2f}')

        # Save input, prediction, and ground truth images.
        rgb_input = inference_retval['rgb_input']
        rgb_output = inference_retval['rgb_output']
        rgb_target = inference_retval['rgb_target']

        gallery = np.stack([rgb_input, rgb_output, rgb_target])
        gallery = np.clip(gallery, 0.0, 1.0)
        file_name = f'rgb_iogt_s{cur_step}.png'
        online_name = f'rgb_iogt'
        self.save_gallery(gallery, step=cur_step, file_name=file_name, online_name=online_name)