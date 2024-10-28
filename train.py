import argparse
import commentjson as json
import numpy as np
import os
import re
import pickle

import tinycudann as tcnn

from utils import utils
from utils.utils import debatch
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl

#########################################################################################################
################################################ DATASET ################################################
#########################################################################################################

class BundleDataset(Dataset):
    def __init__(self, args, load_volume=False):
        self.args = args
        print("Loading from:", self.args.data_path)
        
        data = np.load(args.data_path, allow_pickle=True)
        
        self.characteristics = data['characteristics'].item() # camera characteristics
        self.motion = data['motion'].item()
        self.frame_timestamps = torch.tensor([data[f'raw_{i}'].item()['timestamp'] for i in range(data['num_raw_frames'])], dtype=torch.float64)
        self.motion_timestamps = torch.tensor(self.motion['timestamp'], dtype=torch.float64)
        self.num_frames = data['num_raw_frames'].item()

        # WXYZ quaternions, remove phase wraps (2pi jumps)
        self.quaternions = utils.unwrap_quaternions(torch.tensor(self.motion['quaternion']).float()) # T',4, has different timestamps from frames
        # flip x/y to match out convention:
        # +y towards selfie camera, +x towards buttons, +z towards scene
        self.quaternions[:,1] *= -1
        self.quaternions[:,2] *= -1

        self.quaternions = utils.multi_interp(self.frame_timestamps, self.motion_timestamps, self.quaternions) # interpolate to frame times

        self.reference_quaternion = self.quaternions[0:1]
        self.quaternion_camera_to_world = utils.quaternion_multiply(utils.quaternion_conjugate(self.reference_quaternion), self.quaternions)
        
        self.intrinsics = torch.tensor(np.array([data[f'raw_{i}'].item()['intrinsics'] for i in range(data['num_raw_frames'])])).float()  # T,3,3  
        # swap cx,cy -> landscape to portrait
        cx, cy = self.intrinsics[:, 2, 1].clone(), self.intrinsics[:, 2, 0].clone()
        self.intrinsics[:, 2, 0], self.intrinsics[:, 2, 1] = cx, cy
        # transpose to put cx,cy in right column
        self.intrinsics = self.intrinsics.transpose(1, 2)
        self.intrinsics_inv = torch.inverse(self.intrinsics)

        self.lens_distortion = torch.tensor(data['raw_0'].item()['lens_distortion'])
        self.tonemap_curve = torch.tensor(data['raw_0'].item()['tonemap_curve'], dtype=torch.float32)
        self.ccm = torch.tensor(data['raw_0'].item()['ccm'], dtype=torch.float32)

        self.img_channels = 3
        self.img_height = data['raw_0'].item()['width'] # rotated 90
        self.img_width = data['raw_0'].item()['height']
        self.rgb_volume = None # placeholder volume for fast loading

        # rolling shutter timing compensation, off by default (can bug for data not from a Pixel 8 Pro)
        self.rolling_shutter_skew = data['raw_0'].item()['android']['sensor.rollingShutterSkew'] / 1e9 # delay between top and bottom row readout, seconds
        self.rolling_shutter_skew_row = self.rolling_shutter_skew / (self.img_height - 1)
        self.row_timestamps = torch.zeros(len(self.frame_timestamps), self.img_height, dtype=torch.float64) # NxH
        for i, frame_timestamp in enumerate(self.frame_timestamps):
            for j in range(self.img_height):
                if args.rolling_shutter:
                    self.row_timestamps[i,j] = frame_timestamp + j * self.rolling_shutter_skew_row
                else:
                    self.row_timestamps[i,j] = frame_timestamp
                    

        self.row_timestamps = self.row_timestamps - self.row_timestamps[0,0] # zero at start
        self.row_timestamps = self.row_timestamps/self.row_timestamps[-1,-1] # normalize to 0-1

        if args.frames is not None:
            # subsample frames
            self.num_frames = len(args.frames)
            self.frame_timestamps = self.frame_timestamps[args.frames]
            self.intrinsics = self.intrinsics[args.frames]
            self.intrinsics_inv = self.intrinsics_inv[args.frames]
            self.quaternions = self.quaternions[args.frames]
            self.reference_quaternion = self.quaternions[0:1]
            self.quaternion_camera_to_world = self.quaternion_camera_to_world[args.frames]

        if load_volume:
            self.load_volume()

        self.frame_batch_size = 2 * (self.args.point_batch_size // self.num_frames // 2) # nearest even cut
        self.point_batch_size = self.frame_batch_size * self.num_frames # nearest multiple of num_frames
        self.num_batches = self.args.num_batches

        self.training_phase = 0.0 # fraction of training complete
        print("Frame Count: ", self.num_frames)
    
    def load_volume(self):
        volume_path = self.args.data_path.replace("frame_bundle.npz", "rgb_volume.npy") 
        if os.path.exists(volume_path):
            print("Loading cached volume from:", volume_path)
            self.rgb_volume = torch.tensor(np.load(volume_path)).float()
        else:
            data = dict(np.load(self.args.data_path, allow_pickle=True))
            utils.de_item(data)
            self.rgb_volume = (utils.raw_to_rgb(data)) # T,C,H,W
            if self.args.cache:
                print("Saving cached volume to:", volume_path)
                np.save(volume_path, self.rgb_volume.numpy())

        if self.args.max_percentile < 100: # cut off highlights (long-tail-distribution)
            self.clip = np.percentile(self.rgb_volume[0], self.args.max_percentile)
            self.rgb_volume = torch.clamp(self.rgb_volume, 0, self.clip)
            self.rgb_volume = self.rgb_volume/self.clip
        else:
            self.clip = 1.0

        self.mean = self.rgb_volume[0].mean()
        self.rgb_volume = (16 * (self.rgb_volume - self.mean)).to(torch.float16)            

        if self.args.frames is not None:
            self.rgb_volume = self.rgb_volume[self.args.frames]  # subsample frames

        self.img_height, self.img_width = self.rgb_volume.shape[2], self.rgb_volume.shape[3]


    def __len__(self):
        return self.num_batches  # arbitrary as we continuously generate random samples
        
    def __getitem__(self, idx):
        uv = torch.rand((self.frame_batch_size * self.num_frames), 2) # uniform random in [0,1]
        uv = uv * torch.tensor([[self.img_width-1, self.img_height-1]]) # scale to image dimensions
        uv = uv.round() # quantize to pixels
        u,v = uv.unbind(-1)

        t = torch.zeros_like(uv[:,0:1])
        for frame in range(self.num_frames):
            # use row to index into row_timestamps
            t[frame * self.frame_batch_size:(frame + 1) * self.frame_batch_size,0] = self.row_timestamps[frame, v[frame * self.frame_batch_size:(frame + 1) * self.frame_batch_size].long()]

        uv = uv / torch.tensor([[self.img_width-1, self.img_height-1]]) # scale back to 0-1
        
        return self.generate_samples(t, uv)

    def generate_samples(self, t, uv):
        """ generate samples from dataset and camera parameters for training
        """
            
        # create frame_batch_size of quaterions for each frame
        quaternion_camera_to_world = (self.quaternion_camera_to_world[:self.num_frames]).repeat_interleave(self.frame_batch_size, dim=0)
        # create frame_batch_size of intrinsics for each frame
        intrinsics = (self.intrinsics[:self.num_frames]).repeat_interleave(self.frame_batch_size, dim=0)
        intrinsics_inv = (self.intrinsics_inv[:self.num_frames]).repeat_interleave(self.frame_batch_size, dim=0)
        
        # sample grid
        u,v = uv.unbind(-1)
        u, v = (u * (self.img_width - 1)).round().long(), (v * (self.img_height - 1)).round().long() # pixel coordinates
        u, v = torch.clamp(u, 0, self.img_width-1), torch.clamp(v, 0, self.img_height-1) # clamp to image bounds
        x, y = u, (self.img_height - 1) - v # convert to array coordinates

        if self.rgb_volume is not None:
            rgb_samples = []
            for frame in range(self.num_frames):
                frame_min, frame_max = frame * self.frame_batch_size, (frame + 1) * self.frame_batch_size
                rgb_samples.append(self.rgb_volume[frame,:,y[frame_min:frame_max],x[frame_min:frame_max]].permute(1,0))
            rgb_samples = torch.cat(rgb_samples, dim=0)
        else:
            rgb_samples = torch.zeros(self.frame_batch_size * self.num_frames, 3)

        return t, uv, quaternion_camera_to_world, intrinsics, intrinsics_inv, rgb_samples
    
    def sample_frame(self, frame, uv):
        """ sample frame [frame] at coordinates u,v
        """

        u,v = uv.unbind(-1)
        u, v = (u * self.img_width).round().long(), (v * self.img_height).round().long() # pixel coordinates
        u, v = torch.clamp(u, 0, self.img_width-1), torch.clamp(v, 0, self.img_height-1) # clamp to image bounds
        x, y = u, (self.img_height - 1) - v # convert to array coordinates

        if self.rgb_volume is not None:
            rgb_samples = self.rgb_volume[frame:frame+1,:,y,x] # frames x 3 x H x W volume
            rgb_samples = rgb_samples.permute(0,2,1).flatten(0,1) # point_batch_size x channels
        else:
            rgb_samples = torch.zeros(u.shape[0], 3)
            
        return rgb_samples

#########################################################################################################
################################################ MODELS #################$###############################
#########################################################################################################

class RotationModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.rotations = nn.Parameter(torch.zeros(1, 3, self.args.num_frames, dtype=torch.float32), requires_grad=True)

    def forward(self, quaternion_camera_to_world, t):
        self.rotations.data[:, :, 0] = 0.0  # zero out first frame's rotation

        rotations = utils.interpolate_params(self.rotations, t)
        rotations = self.args.rotation_weight * rotations
        rx, ry, rz = rotations[:, 0], rotations[:, 1], rotations[:, 2]
        r1 = torch.ones_like(rx)

        rotation_offsets = torch.stack([torch.stack([r1, -rz, ry], dim=-1),
                                        torch.stack([rz, r1, -rx], dim=-1),
                                        torch.stack([-ry, rx, r1], dim=-1)], dim=-1)

        return rotation_offsets @ utils.convert_quaternions_to_rot(quaternion_camera_to_world)

class TranslationModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.translations_coarse = nn.Parameter(torch.rand(1, 3, 7, dtype=torch.float32) * 1e-5, requires_grad=True)
        self.translations_fine = nn.Parameter(torch.rand(1, 3, args.num_frames, dtype=torch.float32) * 1e-5, requires_grad=True)
        self.center = nn.Parameter(torch.zeros(1, 3, dtype=torch.float32), requires_grad=True)

    def forward(self, t, training_phase=1.0):
        self.translations_coarse.data[:, :, 0] = 0.0  # zero out first frame's translation
        self.translations_fine.data[:, :, 0] = 0.0  # zero out first frame's translation

        if training_phase < 0.25:
            translation = utils.interpolate_params(self.translations_coarse, t)
        else:
            translation = utils.interpolate_params(self.translations_coarse, t) + utils.interpolate_params(self.translations_fine, t)

        return self.args.focal_compensation * self.args.translation_weight * (translation + 5 * self.center)

class LightSphereModel(pl.LightningModule):
    
    def __init__(self, args):
        super().__init__()
        self.args = args

        encoding_offset_position_config = json.load(open(f"config/config_{args.encoding_offset_position_config}.json"))["encoding"] 
        encoding_offset_angle_config = json.load(open(f"config/config_{args.encoding_offset_angle_config}.json"))["encoding"] 
        network_offset_config = json.load(open(f"config/config_{args.network_offset_config}.json"))["network"] 

        encoding_color_position_config = json.load(open(f"config/config_{args.encoding_color_position_config}.json"))["encoding"]  
        encoding_color_angle_config = json.load(open(f"config/config_{args.encoding_color_angle_config}.json"))["encoding"]
        network_color_position_config = json.load(open(f"config/config_{args.network_color_position_config}.json"))["network"] 
        network_color_angle_config = json.load(open(f"config/config_{args.network_color_angle_config}.json"))["network"] 

        self.encoding_offset_position = tcnn.Encoding(n_input_dims=3, encoding_config=encoding_offset_position_config)
        self.encoding_offset_angle = tcnn.Encoding(n_input_dims=2, encoding_config=encoding_offset_angle_config)

        self.network_offset = tcnn.Network(n_input_dims=self.encoding_offset_position.n_output_dims + self.encoding_offset_angle.n_output_dims,
                                            n_output_dims=3, network_config=network_offset_config)
        
        self.encoding_color_position = tcnn.Encoding(n_input_dims=3, encoding_config=encoding_color_position_config)
        self.encoding_color_angle = tcnn.Encoding(n_input_dims=2, encoding_config=encoding_color_angle_config)

        self.network_color_position = tcnn.Network(n_input_dims=self.encoding_color_position.n_output_dims, n_output_dims=64, network_config=network_color_position_config)
        self.network_color_angle = tcnn.Network(n_input_dims=self.encoding_color_position.n_output_dims + self.encoding_color_angle.n_output_dims, n_output_dims=64, network_config=network_color_angle_config)
        self.network_color = nn.Linear(64, 3, dtype=torch.float32, bias=False) # faster than tcnn.Network
        self.network_color.weight.data = torch.rand_like(self.network_color.weight.data)
        
        self.initial_rgb = torch.nn.Parameter(data=torch.zeros([1,3], dtype=torch.float16), requires_grad=False)

        self.enc_feat_direction = None
        self.enc_offset_intersection = None
        self.enc_offset_direction = None
        self.enc_image_outer = None

    def mask(self, encoding,  training_phase, initial):
        if self.args.no_mask:
            return encoding
        else:
            return utils.mask(encoding, training_phase, initial)
        
    @torch.jit.script
    def solve_sphere_crossings(ray_origins, ray_directions):
        # Coefficients for the quadratic equation
        b = 2 * torch.sum(ray_origins * ray_directions, dim=1)
        c = torch.sum(ray_origins**2, dim=1) - 1.0
        
        discriminant = b**2 - 4 * c
        sqrt_discriminant = torch.sqrt(discriminant)
        t = (-b + sqrt_discriminant) / 2
        intersections = ray_origins + t.unsqueeze(-1) * ray_directions
        return intersections
                
    def inference(self, t, uv, ray_origins, ray_directions, training_phase=1.0):
        # Slightly slimmed down version of forward() for inference
        uv = uv.clamp(0, 1)

        intersections_sphere = self.solve_sphere_crossings(ray_origins, ray_directions)

        if not self.args.no_offset:
            encoded_offset_position = self.encoding_offset_position((intersections_sphere + 1) / 2)
            encoded_offset_angle = self.encoding_offset_angle(uv)
            encoded_offset = torch.cat((encoded_offset_position, encoded_offset_angle), dim=1)
            
            offset = self.network_offset(encoded_offset).float()
            ray_directions_offset = ray_directions + torch.cross(offset, ray_directions, dim=1)
            intersections_sphere_offset = self.solve_sphere_crossings(ray_origins, ray_directions_offset)
        else:
            intersections_sphere_offset = intersections_sphere

        encoded_color_position = self.encoding_color_position((intersections_sphere_offset + 1) / 2)
        feat_color = self.network_color_position(encoded_color_position).float()

        if not self.args.no_view_color:
            encoded_color_angle = self.encoding_color_angle(uv)
            encoded_color = torch.cat((encoded_color_position, encoded_color_angle), dim=1)
            feat_color_angle = self.network_color_angle(encoded_color)
            feat_color = feat_color + feat_color_angle

        rgb = self.initial_rgb + self.network_color(feat_color)
        return rgb

    def forward(self, t, uv, ray_origins, ray_directions, training_phase):
        uv = uv.clamp(0,1)
        
        if training_phase < 0.2: # Apply random perturbation for training during early epochs
            factor = 0.015 * self.args.focal_compensation
            perturbation = torch.randn_like(ray_origins) * factor / 0.2 * max(0.0, 0.25 - training_phase)
            ray_origins = ray_origins + perturbation

        intersections_sphere = self.solve_sphere_crossings(ray_origins, ray_directions)

        if training_phase > 0.2 and not self.args.no_offset:
            encoded_offset_position = self.mask(self.encoding_offset_position((intersections_sphere + 1) / 2), training_phase, initial=0.2)
            encoded_offset_angle = self.mask(self.encoding_offset_angle(uv), training_phase, initial=0.5)
            encoded_offset = torch.cat((encoded_offset_position, encoded_offset_angle), dim=1)

            offset = self.network_offset(encoded_offset).float()
            ray_directions_offset = ray_directions + torch.cross(offset, ray_directions, dim=1) # linearized rotation
            intersections_sphere_offset = self.solve_sphere_crossings(ray_origins, ray_directions_offset)
        else:
            offset = torch.ones_like(ray_directions)
            intersections_sphere_offset = intersections_sphere

        encoded_color_position = self.mask(self.encoding_color_position((intersections_sphere_offset + 1) / 2), training_phase, initial=0.8)
        feat_color = self.network_color_position(encoded_color_position).float()

        if training_phase > 0.25 and not self.args.no_view_color:
            encoded_color_angle = self.mask(self.encoding_color_angle(uv), training_phase, initial=0.2)
            encoded_color = torch.cat((encoded_color_position, encoded_color_angle), dim=1)
            feat_color_angle = self.network_color_angle(encoded_color)
            feat_color = feat_color + feat_color_angle 
        else:
            feat_color_angle = torch.zeros_like(feat_color)

        rgb = self.initial_rgb + self.network_color(feat_color)
        return rgb, offset, feat_color_angle

    
class DistortionModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, kappa):
        if self.args.no_lens_distortion: # no distortion
            return (kappa * 0.0).to(self.device)
        else:
            return kappa.to(self.device)

        
#########################################################################################################
################################################ NETWORK ################################################
#########################################################################################################
    
class PanoModel(pl.LightningModule):
    def __init__(self, args, cached_data=None):
        super().__init__()
        # load network configs

        self.args = args
        if cached_data is None:
             self.data = BundleDataset(self.args)
        else:
            with open(cached_data, 'rb') as file:
                self.data = pickle.load(file)

        if args.frames is None:
            self.args.frames = list(range(self.data.num_frames))
        self.args.num_frames = self.data.num_frames
            

        # to account for varying focal lengths, scale camera/ray motion to match 82deg Pixel 8 Pro main lens
        self.args.focal_compensation = 1.9236 / (self.data.intrinsics[0,0,0].item()/self.data.intrinsics[0,0,2].item())

        self.model_translation = TranslationModel(self.args)
        self.model_rotation = RotationModel(self.args)
        self.model_distortion = DistortionModel(self.args)
        self.model_light_sphere = LightSphereModel(self.args)

        self.training_phase = 1.0
        self.save_hyperparameters()

    def load_volume(self):
        self.data.load_volume()
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr, betas=[0.9,0.99], eps=1e-9, weight_decay=self.args.weight_decay)
        gamma = 1.0
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        
        return [optimizer], [scheduler]

    def inference(self, *args, **kwargs):
        with torch.no_grad():
            return self.model_light_sphere.inference(*args, **kwargs)
        
    def forward(self, *args, **kwargs):
        return self.model_light_sphere(*args, **kwargs)

    def generate_ray_directions(self, uv, camera_to_world, intrinsics_inv):
        u, v = uv[:, 0:1] * self.data.img_width, uv[:, 1:2] * self.data.img_height
        uv1 = torch.cat([u, v, torch.ones_like(u)], dim=1)  # N x 3
        # Transform pixel coordinates to camera coordinates
        xy1 = torch.bmm(intrinsics_inv, uv1.unsqueeze(2)).squeeze(2)  # N x 3
        xy = xy1[:, 0:2]


        x, y = xy[:, 0:1], xy[:, 1:2]
        r2 = x**2 + y**2
        r4 = r2**2
        r6 = r4 * r2

        kappa1, kappa2, kappa3, kappa4, kappa5 = self.model_distortion(self.data.lens_distortion)

        # Apply radial distortion
        x_distorted = x * (1 + kappa1 * r2 + kappa2 * r4 + kappa3 * r6) + \
                    2 * kappa4 * x * y + kappa5 * (r2 + 2 * x**2)
        y_distorted = y * (1 + kappa1 * r2 + kappa2 * r4 + kappa3 * r6) + \
                    2 * kappa5 * x * y + kappa4 * (r2 + 2 * y**2)

        xy_distorted = torch.cat([x_distorted, y_distorted], dim=1)
        
        # Combine with z = 1 for direction calculation
        ray_directions_unrotated = torch.cat([xy_distorted, torch.ones_like(x)], dim=1)  # N x 3
        
        ray_directions = torch.bmm(camera_to_world, ray_directions_unrotated.unsqueeze(2)).squeeze(2)  # Apply camera rotation

        # Normalize ray directions
        ray_directions = ray_directions / ray_directions.norm(dim=1, keepdim=True)

        return ray_directions

    def training_step(self, train_batch, batch_idx):
        t, uv, quaternion_camera_to_world, intrinsics, intrinsics_inv, rgb_reference = debatch(train_batch) # collapse batch + point dimensions

        camera_to_world = self.model_rotation(quaternion_camera_to_world, t) # apply rotation offset
        ray_origins = self.model_translation(t, self.training_phase) # camera center in world coordinates
        ray_origins.clamp(-0.99,0.99) # bound within the sphere
        ray_directions = self.generate_ray_directions(uv, camera_to_world, intrinsics_inv)

        rgb, offset, feat_color_angle = self.forward(t, uv, ray_origins, ray_directions, self.training_phase)   

        loss = 0.0

        rgb_loss = F.l1_loss(rgb, rgb_reference)
        self.log('loss/rgb', rgb_loss.mean())
        loss += rgb_loss.mean()

        
        factor = 1.2  # Adjust this parameter to control the bend of the curve (1 for linear, 2 for quadratic, etc.)
        normalized_epoch = (self.current_epoch + (batch_idx/self.args.num_batches)) / (self.args.max_epochs - 1)

        # Update the training_phase calculation with the factor
        self.training_phase = min(1.0, 0.05 + (normalized_epoch ** factor))
        self.data.training_phase = self.training_phase

        return loss

    def color_and_tone(self, rgb_samples, height, width):
        """ Apply CCM and tone mapping to raw samples
        """
        
        img = self.color(rgb_samples, height, width)
        img = utils.apply_tonemap(img, self.data.tonemap_curve.to(rgb_samples.device))
            
        return img.clamp(0,1)
    
    def color(self, rgb_samples, height, width):
        """ Apply CCM to raw samples
        """
        
        img = self.data.ccm.to(rgb_samples.device) @  (self.data.mean + rgb_samples.float()/16.0).T
        img = img.reshape(3, height, width)
            
        return img.clamp(0,1)
            
    @torch.no_grad()
    def chunk_forward(self, quaternion_camera_to_world, intrinsics_inv, t, uv, translation, chunk_size=1000000):
        """ Forward model with chunking to avoid OOM
        """
        total_elements = t.shape[0]

        for start_idx in range(0, total_elements, chunk_size):
            end_idx = min(start_idx + chunk_size, total_elements)

            camera_to_world_chunk = self.model_rotation(quaternion_camera_to_world[start_idx:end_idx], t[start_idx:end_idx])
            ray_directions_chunk = self.generate_ray_directions(uv[start_idx:end_idx], camera_to_world_chunk, intrinsics_inv[start_idx:end_idx])
            
            if translation is None:
                ray_origins_chunk = self.model_translation(t[start_idx:end_idx])
            else:
                ray_origins_chunk = torch.zeros_like(ray_directions_chunk) + translation

            chunk_outputs = self.forward(t[start_idx:end_idx], uv[start_idx:end_idx], ray_origins_chunk, ray_directions_chunk, self.training_phase)

            if start_idx == 0:
                num_outputs = len(chunk_outputs)
                final_outputs = [[] for _ in range(num_outputs)]

            for i, output in enumerate(chunk_outputs):
                final_outputs[i].append(output.cpu())

        final_outputs = tuple(torch.cat(outputs, dim=0) for outputs in final_outputs)

        return final_outputs
        
    @torch.no_grad()
    def generate_outputs(self, time, height=720, width=720, u_lims=[0,1], v_lims=[0,1], fov_scale=1.0, quaternion_camera_to_world=None, intrinsics_inv=None, translation=None, sensor_size=None):

        device = self.device

        uv = utils.make_grid(height, width, u_lims, v_lims)
        frame = int(time * (self.data.num_frames - 1))
        t = torch.tensor(time, dtype=torch.float32).repeat(uv.shape[0])[:,None] # num_points x 1
    
        rgb_reference = self.data.sample_frame(frame, uv) # reference rgb samples

        if intrinsics_inv is None :
            intrinsics_inv = self.data.intrinsics_inv[frame:frame+2] # 2 x 3 x 3
            if time <= 0 or time >= 1.0: # select exact frame timestamp
                intrinsics_inv = intrinsics_inv[0:1]
            else: # interpolate between frames
                fraction = time * (self.data.num_frames - 1) - frame
                intrinsics_inv = intrinsics_inv[0:1] * (1 - fraction) + intrinsics_inv[1:2] * fraction

        if quaternion_camera_to_world is None:
            quaternion_camera_to_world = self.data.quaternion_camera_to_world[frame:frame+2] # 2 x 3 x 3
            if time <= 0 or time >= 1.0:
                quaternion_camera_to_world = quaternion_camera_to_world[0:1]
            else: # interpolate between frames
                fraction = time * (self.data.num_frames - 1) - frame
                quaternion_camera_to_world = quaternion_camera_to_world[0:1] * (1 - fraction) + quaternion_camera_to_world[1:2] * fraction
        
        intrinsics_inv = intrinsics_inv.clone()
        intrinsics_inv[:,0] = intrinsics_inv[:,0] * fov_scale
        intrinsics_inv[:,1] = intrinsics_inv[:,1]

        intrinsics_inv = intrinsics_inv.repeat(uv.shape[0],1,1) # num_points x 3 x 3
        quaternion_camera_to_world = quaternion_camera_to_world.repeat(uv.shape[0],1) # num_points x 4

        quaternion_camera_to_world, intrinsics_inv, t, uv = quaternion_camera_to_world.to(device), intrinsics_inv.to(device), t.to(device), uv.to(device)

        rgb, offset, feat_color_angle = self.chunk_forward(quaternion_camera_to_world, intrinsics_inv, t, uv, translation, chunk_size=3000**2) # break into chunks to avoid OOM

        rgb_reference = self.color_and_tone(rgb_reference, height, width)
        rgb = self.color_and_tone(rgb, height, width)

        offset = offset.reshape(height, width, 3).float().permute(2,0,1)

        # Normalize the offset tensor along the axis
        offset = offset
        offset_img = utils.colorize_tensor(offset.mean(dim=0), vmin=offset.min(), vmax=offset.max(), cmap="RdYlBu")
            
        return rgb_reference, rgb, offset, offset_img
    
    
    def save_outputs(self, path, high_res=False):
        os.makedirs(f"outputs/{self.args.name + path}", exist_ok=True)
        if high_res:
            rgb_reference, rgb, offset, offset_img = model.generate_outputs(time=0, height=2560, width=1920)
        else:
            rgb_reference, rgb, offset, offset_img = model.generate_outputs(time=0, height=1080, width=810)

    
#########################################################################################################
############################################### VALIDATION ##############################################
#########################################################################################################
        
class ValidationCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.unlock = False

    def bright(self, rgb):
        return ((rgb / np.percentile(rgb, 95)) ** 0.7).clamp(0,1)
        
    def on_train_epoch_start(self, trainer, model):
        print(f"Training phase (0-1): {model.training_phase}")
        
        if model.current_epoch == 1:
            model.model_translation.translations_coarse.requires_grad_(True)
            model.model_translation.translations_fine.requires_grad_(True)
            model.model_rotation.rotations.requires_grad_(True)

        if model.args.fast: # skip tensorboarding except for beginning and end
            if model.current_epoch == model.args.max_epochs - 1 or model.current_epoch == 0:
                pass
            else:
                return
            
        for i, time in enumerate([0.2, 0.5, 0.8]):
            if model.args.hd:
                rgb_reference, rgb, offset, offset_img = model.generate_outputs(time=time, height=1080, width=1080, fov_scale=1.4)
            else:
                rgb_reference, rgb, offset, offset_img = model.generate_outputs(time=time, fov_scale=2.5)

            model.logger.experiment.add_image(f'pred/{i}_rgb_combined', rgb, global_step=trainer.global_step)
            model.logger.experiment.add_image(f'pred/{i}_rgb_combined_bright', self.bright(rgb), global_step=trainer.global_step)
            model.logger.experiment.add_image(f'pred/{i}_offset', offset_img, global_step=trainer.global_step)

            
    def on_train_start(self, trainer, model):
        pl.seed_everything(42) # the answer to life, the universe, and everything

        # initialize rgb as average color of first frame of data (minimize the amount the rgb models have to learn)
        model.model_light_sphere.initial_rgb.data = torch.mean(model.data.rgb_volume[0], dim=(1,2))[None,:].to(model.device).to(torch.float16)
        
        model.logger.experiment.add_text("args", str(model.args))

        for i, time in enumerate([0, 0.5, 1.0]): 
            rgb_reference, rgb, offset, offset_img = model.generate_outputs(time=time, height=1080, width=810)
            model.logger.experiment.add_image(f'gt/{i}_rgb_reference', rgb_reference, global_step=trainer.global_step)
            model.logger.experiment.add_image(f'gt/{i}_rgb_reference_bright', self.bright(rgb_reference), global_step=trainer.global_step)

        model.training_phase = 0.05
        model.data.training_phase = model.training_phase

            
    def on_train_end(self, trainer, model):
        checkpoint_dir = os.path.join("checkpoints", model.args.name, "last.ckpt")
        data_dir = os.path.join("checkpoints", model.args.name, "data.pkl")
        
        os.makedirs(os.path.dirname(checkpoint_dir), exist_ok=True)
        checkpoint = trainer._checkpoint_connector.dump_checkpoint()
        
        # Forcibly remove optimizer states from the checkpoint
        if 'optimizer_states' in checkpoint:
            del checkpoint['optimizer_states']

        torch.save(checkpoint, checkpoint_dir)
                    
        with open(data_dir, 'wb') as file:
            model.data.rgb_volume = None
            pickle.dump(model.data, file)

if __name__ == "__main__":
    
    # argparse
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--point_batch_size', type=int, default=2**18, help="Number of points to sample per dataloader index.")
    parser.add_argument('--num_batches', type=int, default=200, help="Number of training batches.")
    parser.add_argument('--max_percentile', type=float, default=99.5, help="Percentile of lightest pixels to cut.")
    parser.add_argument('--frames', type=str, help="Which subset of frames to use for training, e.g. 0,10,20,30,40")
    
    # model
    parser.add_argument('--rotation_weight', type=float, default=1e-2, help="Scale learned rotation.")
    parser.add_argument('--translation_weight', type=float, default=1e0, help="Scale learned translation.")
    parser.add_argument('--rolling_shutter', action='store_true', help="Use rolling shutter compensation.")
    parser.add_argument('--no_mask', action='store_true', help="Do not use mask.")
    parser.add_argument('--no_offset', action='store_true', help="Do not use ray offset model.")
    parser.add_argument('--no_view_color', action='store_true', help="Do not use view dependent color model.")
    parser.add_argument('--no_lens_distortion', action='store_true', help="Do not use lens distortion model.")



    # light sphere
    parser.add_argument('--encoding_offset_position_config', type=str, default="small", help="Encoding offset position configuration (tiny, small, medium, large, ultrakill).")
    parser.add_argument('--encoding_offset_angle_config', type=str, default="small", help="Encoding offset angle configuration (tiny, small, medium, large, ultrakill).")
    parser.add_argument('--network_offset_config', type=str, default="large", help="Network offset configuration (tiny, small, medium, large, ultrakill).")

    parser.add_argument('--encoding_color_position_config', type=str, default="large", help="Encoding color position configuration (tiny, small, medium, large, ultrakill).")
    parser.add_argument('--encoding_color_angle_config', type=str, default="small", help="Encoding color angle configuration (tiny, small, medium, large, ultrakill).")
    parser.add_argument('--network_color_position_config', type=str, default="large", help="Network color position configuration (tiny, small, medium, large, ultrakill).")
    parser.add_argument('--network_color_angle_config', type=str, default="large", help="Network color angle configuration (tiny, small, medium, large, ultrakill).")

    # training
    parser.add_argument('--data_path', '--d', type=str, required=True, help="Path to frame_bundle.npz")
    parser.add_argument('--name', type=str, required=True, help="Experiment name for logs and checkpoints.")
    parser.add_argument('--max_epochs', type=int, default=100, help="Number of training epochs.")
    parser.add_argument('--lr', type=float, default=5e-4, help="Learning rate.")
    parser.add_argument('--weight_decay', type=float, default=1e-5, help="Weight decay.")
    parser.add_argument('--save_video', action='store_true', help="Store training outputs at each epoch for visualization.")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument('--debug', action='store_true', help="Debug mode, only use 1 batch.")
    parser.add_argument('--fast', action='store_true', help="Fast mode.")
    parser.add_argument('--cache', action='store_true', help="Cache data.")
    parser.add_argument('--hd', action='store_true', help="Make tensorboard HD.")


    args = parser.parse_args()
        # parse plane args
    print(args)
    if args.frames is not None: 
        args.frames = [int(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", args.frames)]

    # model
    model = PanoModel(args)
    model.load_volume()

    # dataset
    data = model.data
    train_loader = DataLoader(data, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=True, prefetch_factor=1)

    model.model_translation.translations_coarse.requires_grad_(False)
    model.model_translation.translations_fine.requires_grad_(False)
    model.model_rotation.rotations.requires_grad_(False)

    torch.set_float32_matmul_precision('medium')

    # training
    lr_callback = pl.callbacks.LearningRateMonitor()
    logger = pl.loggers.TensorBoardLogger(save_dir=os.getcwd(), version=args.name, name="lightning_logs")
    validation_callback = ValidationCallback()
    trainer = pl.Trainer(accelerator="gpu", devices=torch.cuda.device_count(), num_nodes=1, strategy="auto", max_epochs=args.max_epochs,
                         logger=logger, callbacks=[validation_callback, lr_callback], enable_checkpointing=False, fast_dev_run=args.debug)
    trainer.fit(model, train_loader)
