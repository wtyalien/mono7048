# from __future__ import absolute_import, division, print_function


# import time
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter

# import json

# from utils import *
# from kitti_utils import *
# from layers import *

# import datasets
# import networks
# from linear_warmup_cosine_annealing_warm_restarts_weight_decay import ChainedScheduler


# # torch.backends.cudnn.benchmark = True


# def time_sync():
#     # PyTorch-accurate time
#     if torch.cuda.is_available():
#         torch.cuda.synchronize()
#     return time.time()


# class Trainer:
#     def __init__(self, options):
#         self.opt = options
#         self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

#         # checking height and width are multiples of 32
#         assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
#         assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

#         self.models = {}
#         self.models_pose = {}
#         self.parameters_to_train = []
#         self.parameters_to_train_pose = []

#         self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
#         self.profile = self.opt.profile

#         self.num_scales = len(self.opt.scales)
#         self.frame_ids = len(self.opt.frame_ids)
#         self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

#         assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

#         self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

#         if self.opt.use_stereo:
#             self.opt.frame_ids.append("s")

#         self.models["encoder"] = networks.LiteMono(model=self.opt.model,
#                                                    drop_path_rate=self.opt.drop_path,
#                                                    width=self.opt.width, height=self.opt.height)

#         self.models["encoder"].to(self.device)
#         self.parameters_to_train += list(self.models["encoder"].parameters())

#         self.models["depth"] = networks.DepthDecoder(self.models["encoder"].num_ch_enc,
#                                                      self.opt.scales)
#         self.models["depth"].to(self.device)
#         self.parameters_to_train += list(self.models["depth"].parameters())

#         if self.use_pose_net:
#             if self.opt.pose_model_type == "separate_resnet":
#                 self.models_pose["pose_encoder"] = networks.ResnetEncoder(
#                     self.opt.num_layers,
#                     self.opt.weights_init == "pretrained",
#                     num_input_images=self.num_pose_frames)

#                 self.models_pose["pose_encoder"].to(self.device)
#                 self.parameters_to_train_pose += list(self.models_pose["pose_encoder"].parameters())

#                 self.models_pose["pose"] = networks.PoseDecoder(
#                     self.models_pose["pose_encoder"].num_ch_enc,
#                     num_input_features=1,
#                     num_frames_to_predict_for=2)

#             elif self.opt.pose_model_type == "shared":
#                 self.models_pose["pose"] = networks.PoseDecoder(
#                     self.models["encoder"].num_ch_enc, self.num_pose_frames)

#             elif self.opt.pose_model_type == "posecnn":
#                 self.models_pose["pose"] = networks.PoseCNN(
#                     self.num_input_frames if self.opt.pose_model_input == "all" else 2)

#             self.models_pose["pose"].to(self.device)
#             self.parameters_to_train_pose += list(self.models_pose["pose"].parameters())

#         if self.opt.predictive_mask:
#             assert self.opt.disable_automasking, \
#                 "When using predictive_mask, please disable automasking with --disable_automasking"

#             # Our implementation of the predictive masking baseline has the the same architecture
#             # as our depth decoder. We predict a separate mask for each source frame.
#             self.models["predictive_mask"] = networks.DepthDecoder(
#                 self.models["encoder"].num_ch_enc, self.opt.scales,
#                 num_output_channels=(len(self.opt.frame_ids) - 1))
#             self.models["predictive_mask"].to(self.device)
#             self.parameters_to_train += list(self.models["predictive_mask"].parameters())

#         self.model_optimizer = optim.AdamW(self.parameters_to_train, self.opt.lr[0], weight_decay=self.opt.weight_decay)
#         if self.use_pose_net:
#             self.model_pose_optimizer = optim.AdamW(self.parameters_to_train_pose, self.opt.lr[3], weight_decay=self.opt.weight_decay)

#         self.model_lr_scheduler = ChainedScheduler(
#                             self.model_optimizer,
#                             T_0=int(self.opt.lr[2]),
#                             T_mul=1,
#                             eta_min=self.opt.lr[1],
#                             last_epoch=-1,
#                             max_lr=self.opt.lr[0],
#                             warmup_steps=0,
#                             gamma=0.9
#                         )
#         self.model_pose_lr_scheduler = ChainedScheduler(
#             self.model_pose_optimizer,
#             T_0=int(self.opt.lr[5]),
#             T_mul=1,
#             eta_min=self.opt.lr[4],
#             last_epoch=-1,
#             max_lr=self.opt.lr[3],
#             warmup_steps=0,
#             gamma=0.9
#         )

#         if self.opt.load_weights_folder is not None:
#             self.load_model()

#         if self.opt.mypretrain is not None:
#             self.load_pretrain()

#         print("Training model named:\n  ", self.opt.model_name)
#         print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
#         print("Training is using:\n  ", self.device)

#         # data
#         datasets_dict = {"kitti": datasets.KITTIRAWDataset,
#                          "kitti_odom": datasets.KITTIOdomDataset}
#         self.dataset = datasets_dict[self.opt.dataset]

#         fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

#         train_filenames = readlines(fpath.format("train"))
#         val_filenames = readlines(fpath.format("val"))
#         img_ext = '.png'  

#         num_train_samples = len(train_filenames)
#         self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

#         train_dataset = self.dataset(
#             self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
#             self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
#         self.train_loader = DataLoader(
#             train_dataset, self.opt.batch_size, True,
#             num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
#         val_dataset = self.dataset(
#             self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
#             self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
#         self.val_loader = DataLoader(
#             val_dataset, self.opt.batch_size, True,
#             num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
#         self.val_iter = iter(self.val_loader)

#         self.writers = {}
#         for mode in ["train", "val"]:
#             self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

#         if not self.opt.no_ssim:
#             self.ssim = SSIM()
#             self.ssim.to(self.device)

#         self.backproject_depth = {}
#         self.project_3d = {}
#         for scale in self.opt.scales:
#             h = self.opt.height // (2 ** scale)
#             w = self.opt.width // (2 ** scale)

#             self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
#             self.backproject_depth[scale].to(self.device)

#             self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
#             self.project_3d[scale].to(self.device)

#         self.depth_metric_names = [
#             "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

#         print("Using split:\n  ", self.opt.split)
#         print("There are {:d} training items and {:d} validation items\n".format(
#             len(train_dataset), len(val_dataset)))

#         self.save_opts()

#     def set_train(self):
#         """Convert all models to training mode
#         """
#         for m in self.models.values():
#             m.train()

#     def set_eval(self):
#         """Convert all models to testing/evaluation mode
#         """
#         for m in self.models.values():
#             m.eval()

#     def train(self):
#         """Run the entire training pipeline
#         """
#         self.epoch = 0
#         self.step = 0
#         self.start_time = time.time()
#         for self.epoch in range(self.opt.num_epochs):
#             self.run_epoch()
#             if (self.epoch + 1) % self.opt.save_frequency == 0:
#                 self.save_model()

#     def run_epoch(self):
#         """Run a single epoch of training and validation
#         """

#         print("Training")
#         self.set_train()

#         self.model_lr_scheduler.step()
#         if self.use_pose_net:
#             self.model_pose_lr_scheduler.step()

#         for batch_idx, inputs in enumerate(self.train_loader):

#             before_op_time = time.time()

#             outputs, losses = self.process_batch(inputs)

#             self.model_optimizer.zero_grad()
#             if self.use_pose_net:
#                 self.model_pose_optimizer.zero_grad()
#             losses["loss"].backward()
#             self.model_optimizer.step()
#             if self.use_pose_net:
#                 self.model_pose_optimizer.step()

#             duration = time.time() - before_op_time

#             # log less frequently after the first 2000 steps to save time & disk space
#             early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 20000
#             late_phase = self.step % 2000 == 0

#             if early_phase or late_phase:
#                 self.log_time(batch_idx, duration, losses["loss"].cpu().data)

#                 if "depth_gt" in inputs:
#                     self.compute_depth_losses(inputs, outputs, losses)

#                 self.log("train", inputs, outputs, losses)
#                 self.val()

#             self.step += 1

#     def process_batch(self, inputs):
#         """Pass a minibatch through the network and generate images and losses
#         """
#         for key, ipt in inputs.items():
#             inputs[key] = ipt.to(self.device)

#         if self.opt.pose_model_type == "shared":
#             # If we are using a shared encoder for both depth and pose (as advocated
#             # in monodepthv1), then all images are fed separately through the depth encoder.
#             all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
#             all_features = self.models["encoder"](all_color_aug)
#             all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

#             features = {}
#             for i, k in enumerate(self.opt.frame_ids):
#                 features[k] = [f[i] for f in all_features]

#             outputs = self.models["depth"](features[0])
#         else:
#             # Otherwise, we only feed the image with frame_id 0 through the depth encoder

#             features = self.models["encoder"](inputs["color_aug", 0, 0])

#             outputs = self.models["depth"](features)

#         if self.opt.predictive_mask:
#             outputs["predictive_mask"] = self.models["predictive_mask"](features)

#         if self.use_pose_net:
#             outputs.update(self.predict_poses(inputs, features))

#         self.generate_images_pred(inputs, outputs)
#         losses = self.compute_losses(inputs, outputs)

#         return outputs, losses

#     def predict_poses(self, inputs, features):
#         """Predict poses between input frames for monocular sequences.
#         """
#         outputs = {}
#         if self.num_pose_frames == 2:
#             # In this setting, we compute the pose to each source frame via a
#             # separate forward pass through the pose network.

#             # select what features the pose network takes as input
#             if self.opt.pose_model_type == "shared":
#                 pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
#             else:
#                 pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

#             for f_i in self.opt.frame_ids[1:]:
#                 if f_i != "s":
#                     # To maintain ordering we always pass frames in temporal order
#                     if f_i < 0:
#                         pose_inputs = [pose_feats[f_i], pose_feats[0]]
#                     else:
#                         pose_inputs = [pose_feats[0], pose_feats[f_i]]

#                     if self.opt.pose_model_type == "separate_resnet":
#                         pose_inputs = [self.models_pose["pose_encoder"](torch.cat(pose_inputs, 1))]
#                     elif self.opt.pose_model_type == "posecnn":
#                         pose_inputs = torch.cat(pose_inputs, 1)

#                     axisangle, translation = self.models_pose["pose"](pose_inputs)
#                     outputs[("axisangle", 0, f_i)] = axisangle
#                     outputs[("translation", 0, f_i)] = translation

#                     # Invert the matrix if the frame id is negative
#                     outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
#                         axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

#         else:
#             # Here we input all frames to the pose net (and predict all poses) together
#             if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
#                 pose_inputs = torch.cat(
#                     [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

#                 if self.opt.pose_model_type == "separate_resnet":
#                     pose_inputs = [self.models["pose_encoder"](pose_inputs)]

#             elif self.opt.pose_model_type == "shared":
#                 pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

#             axisangle, translation = self.models_pose["pose"](pose_inputs)

#             for i, f_i in enumerate(self.opt.frame_ids[1:]):
#                 if f_i != "s":
#                     outputs[("axisangle", 0, f_i)] = axisangle
#                     outputs[("translation", 0, f_i)] = translation
#                     outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
#                         axisangle[:, i], translation[:, i])

#         return outputs

#     def val(self):
#         """Validate the model on a single minibatch
#         """
#         self.set_eval()
#         try:
#             inputs = self.val_iter.__next__()  
#         except StopIteration:
#             self.val_iter = iter(self.val_loader)
#             inputs = self.val_iter.__next__() 

#         with torch.no_grad():
#             outputs, losses = self.process_batch(inputs)

#             if "depth_gt" in inputs:
#                 self.compute_depth_losses(inputs, outputs, losses)

#             self.log("val", inputs, outputs, losses)
#             del inputs, outputs, losses

#         self.set_train()

#     def generate_images_pred(self, inputs, outputs):
#         """Generate the warped (reprojected) color images for a minibatch.
#         Generated images are saved into the `outputs` dictionary.
#         """
#         for scale in self.opt.scales:
#             disp = outputs[("disp", scale)]
#             if self.opt.v1_multiscale:
#                 source_scale = scale
#             else:
#                 disp = F.interpolate(
#                     disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
#                 source_scale = 0

#             _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

#             outputs[("depth", 0, scale)] = depth

#             for i, frame_id in enumerate(self.opt.frame_ids[1:]):

#                 if frame_id == "s":
#                     T = inputs["stereo_T"]
#                 else:
#                     T = outputs[("cam_T_cam", 0, frame_id)]

#                 # from the authors of https://arxiv.org/abs/1712.00175
#                 if self.opt.pose_model_type == "posecnn":

#                     axisangle = outputs[("axisangle", 0, frame_id)]
#                     translation = outputs[("translation", 0, frame_id)]

#                     inv_depth = 1 / depth
#                     mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

#                     T = transformation_from_parameters(
#                         axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

#                 cam_points = self.backproject_depth[source_scale](
#                     depth, inputs[("inv_K", source_scale)])
#                 pix_coords = self.project_3d[source_scale](
#                     cam_points, inputs[("K", source_scale)], T)

#                 outputs[("sample", frame_id, scale)] = pix_coords

#                 outputs[("color", frame_id, scale)] = F.grid_sample(
#                     inputs[("color", frame_id, source_scale)],
#                     outputs[("sample", frame_id, scale)],
#                     padding_mode="border", align_corners=True)

#                 if not self.opt.disable_automasking:
#                     outputs[("color_identity", frame_id, scale)] = \
#                         inputs[("color", frame_id, source_scale)]

#     def compute_reprojection_loss(self, pred, target):
#         """Computes reprojection loss between a batch of predicted and target images
#         """
#         abs_diff = torch.abs(target - pred)
#         l1_loss = abs_diff.mean(1, True)

#         if self.opt.no_ssim:
#             reprojection_loss = l1_loss
#         else:
#             ssim_loss = self.ssim(pred, target).mean(1, True)
#             reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

#         return reprojection_loss

#     def compute_losses(self, inputs, outputs):
#         """Compute the reprojection and smoothness losses for a minibatch
#         """

#         losses = {}
#         total_loss = 0

#         for scale in self.opt.scales:
#             loss = 0
#             reprojection_losses = []

#             if self.opt.v1_multiscale:
#                 source_scale = scale
#             else:
#                 source_scale = 0

#             disp = outputs[("disp", scale)]
#             color = inputs[("color", 0, scale)]
#             target = inputs[("color", 0, source_scale)]

#             for frame_id in self.opt.frame_ids[1:]:
#                 pred = outputs[("color", frame_id, scale)]
#                 reprojection_losses.append(self.compute_reprojection_loss(pred, target))

#             reprojection_losses = torch.cat(reprojection_losses, 1)

#             if not self.opt.disable_automasking:
#                 identity_reprojection_losses = []
#                 for frame_id in self.opt.frame_ids[1:]:
#                     pred = inputs[("color", frame_id, source_scale)]
#                     identity_reprojection_losses.append(
#                         self.compute_reprojection_loss(pred, target))

#                 identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

#                 if self.opt.avg_reprojection:
#                     identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
#                 else:
#                     # save both images, and do min all at once below
#                     identity_reprojection_loss = identity_reprojection_losses

#             elif self.opt.predictive_mask:
#                 # use the predicted mask
#                 mask = outputs["predictive_mask"]["disp", scale]
#                 if not self.opt.v1_multiscale:
#                     mask = F.interpolate(
#                         mask, [self.opt.height, self.opt.width],
#                         mode="bilinear", align_corners=False)

#                 reprojection_losses *= mask

#                 # add a loss pushing mask to 1 (using nn.BCELoss for stability)
#                 weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
#                 loss += weighting_loss.mean()

#             if self.opt.avg_reprojection:
#                 reprojection_loss = reprojection_losses.mean(1, keepdim=True)
#             else:
#                 reprojection_loss = reprojection_losses

#             if not self.opt.disable_automasking:
#                 # add random numbers to break ties
#                 identity_reprojection_loss += torch.randn(
#                     identity_reprojection_loss.shape, device=self.device) * 0.00001

#                 combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
#             else:
#                 combined = reprojection_loss

#             if combined.shape[1] == 1:
#                 to_optimise = combined
#             else:
#                 to_optimise, idxs = torch.min(combined, dim=1)

#             if not self.opt.disable_automasking:
#                 outputs["identity_selection/{}".format(scale)] = (
#                     idxs > identity_reprojection_loss.shape[1] - 1).float()

#             loss += to_optimise.mean()

#             mean_disp = disp.mean(2, True).mean(3, True)
#             norm_disp = disp / (mean_disp + 1e-7)
#             smooth_loss = get_smooth_loss(norm_disp, color)

#             loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
#             total_loss += loss
#             losses["loss/{}".format(scale)] = loss

#         total_loss /= self.num_scales
#         losses["loss"] = total_loss
#         return losses

#     def compute_depth_losses(self, inputs, outputs, losses):
#         """Compute depth metrics, to allow monitoring during training

#         This isn't particularly accurate as it averages over the entire batch,
#         so is only used to give an indication of validation performance
#         """
#         depth_pred = outputs[("depth", 0, 0)]
#         depth_pred = torch.clamp(F.interpolate(
#             depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
#         depth_pred = depth_pred.detach()

#         depth_gt = inputs["depth_gt"]
#         mask = depth_gt > 0

#         # garg/eigen crop
#         crop_mask = torch.zeros_like(mask)
#         crop_mask[:, :, 153:371, 44:1197] = 1
#         mask = mask * crop_mask

#         depth_gt = depth_gt[mask]
#         depth_pred = depth_pred[mask]
#         depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

#         depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

#         depth_errors = compute_depth_errors(depth_gt, depth_pred)

#         for i, metric in enumerate(self.depth_metric_names):
#             losses[metric] = np.array(depth_errors[i].cpu())

#     def log_time(self, batch_idx, duration, loss):
#         """Print a logging statement to the terminal
#         """
#         samples_per_sec = self.opt.batch_size / duration
#         time_sofar = time.time() - self.start_time
#         training_time_left = (
#             self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
#         print_string = "epoch {:>3} | lr {:.6f} |lr_p {:.6f} | batch {:>6} | examples/s: {:5.1f}" + \
#             " | loss: {:.5f} | time elapsed: {} | time left: {}"
#         print(print_string.format(self.epoch, self.model_optimizer.state_dict()['param_groups'][0]['lr'],
#                                   self.model_pose_optimizer.state_dict()['param_groups'][0]['lr'],
#                                   batch_idx, samples_per_sec, loss,
#                                   sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

#     def log(self, mode, inputs, outputs, losses):
#         """Write an event to the tensorboard events file
#         """
#         writer = self.writers[mode]
#         for l, v in losses.items():
#             writer.add_scalar("{}".format(l), v, self.step)

#         for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
#             for s in self.opt.scales:
#                 for frame_id in self.opt.frame_ids:
#                     writer.add_image(
#                         "color_{}_{}/{}".format(frame_id, s, j),
#                         inputs[("color", frame_id, s)][j].data, self.step)
#                     if s == 0 and frame_id != 0:
#                         writer.add_image(
#                             "color_pred_{}_{}/{}".format(frame_id, s, j),
#                             outputs[("color", frame_id, s)][j].data, self.step)

#                 writer.add_image(
#                     "disp_{}/{}".format(s, j),
#                     normalize_image(outputs[("disp", s)][j]), self.step)

#                 if self.opt.predictive_mask:
#                     for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
#                         writer.add_image(
#                             "predictive_mask_{}_{}/{}".format(frame_id, s, j),
#                             outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
#                             self.step)

#                 elif not self.opt.disable_automasking:
#                     writer.add_image(
#                         "automask_{}/{}".format(s, j),
#                         outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

#     def save_opts(self):
#         """Save options to disk so we know what we ran this experiment with
#         """
#         models_dir = os.path.join(self.log_path, "models")
#         if not os.path.exists(models_dir):
#             os.makedirs(models_dir)
#         to_save = self.opt.__dict__.copy()

#         with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
#             json.dump(to_save, f, indent=2)

#     def save_model(self):
#         """Save model weights to disk
#         """
#         save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
#         if not os.path.exists(save_folder):
#             os.makedirs(save_folder)

#         for model_name, model in self.models.items():
#             save_path = os.path.join(save_folder, "{}.pth".format(model_name))
#             to_save = model.state_dict()
#             if model_name == 'encoder':
#                 # save the sizes - these are needed at prediction time
#                 to_save['height'] = self.opt.height
#                 to_save['width'] = self.opt.width
#                 to_save['use_stereo'] = self.opt.use_stereo
#             torch.save(to_save, save_path)

#         for model_name, model in self.models_pose.items():
#             save_path = os.path.join(save_folder, "{}.pth".format(model_name))
#             to_save = model.state_dict()
#             torch.save(to_save, save_path)

#         save_path = os.path.join(save_folder, "{}.pth".format("adam"))
#         torch.save(self.model_optimizer.state_dict(), save_path)

#         save_path = os.path.join(save_folder, "{}.pth".format("adam_pose"))
#         if self.use_pose_net:
#             torch.save(self.model_pose_optimizer.state_dict(), save_path)

#     def load_pretrain(self):
#         self.opt.mypretrain = os.path.expanduser(self.opt.mypretrain)
#         path = self.opt.mypretrain
#         model_dict = self.models["encoder"].state_dict()
#         pretrained_dict = torch.load(path)['model']
#         pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and not k.startswith('norm'))}
#         model_dict.update(pretrained_dict)
#         self.models["encoder"].load_state_dict(model_dict)
#         print('mypretrain loaded.')

#     def load_model(self):
#         """Load model(s) from disk
#         """
#         self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

#         assert os.path.isdir(self.opt.load_weights_folder), \
#             "Cannot find folder {}".format(self.opt.load_weights_folder)
#         print("loading model from folder {}".format(self.opt.load_weights_folder))

#         for n in self.opt.models_to_load:
#             print("Loading {} weights...".format(n))
#             path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))

#             if n in ['pose_encoder', 'pose']:
#                 model_dict = self.models_pose[n].state_dict()
#                 pretrained_dict = torch.load(path)
#                 pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#                 model_dict.update(pretrained_dict)
#                 self.models_pose[n].load_state_dict(model_dict)
#             else:
#                 model_dict = self.models[n].state_dict()
#                 pretrained_dict = torch.load(path)
#                 pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#                 model_dict.update(pretrained_dict)
#                 self.models[n].load_state_dict(model_dict)

#         # loading adam state

#         optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
#         optimizer_pose_load_path = os.path.join(self.opt.load_weights_folder, "adam_pose.pth")
#         if os.path.isfile(optimizer_load_path):
#             print("Loading Adam weights")
#             optimizer_dict = torch.load(optimizer_load_path)
#             optimizer_pose_dict = torch.load(optimizer_pose_load_path)
#             self.model_optimizer.load_state_dict(optimizer_dict)
#             self.model_pose_optimizer.load_state_dict(optimizer_pose_dict)
#         else:
#             print("Cannot find Adam weights so Adam is randomly initialized")

# #v1.4.2,加了EMA指数平均
# from __future__ import absolute_import, division, print_function


# import time
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter

# import json

# from utils import *
# from kitti_utils import *
# from layers import *

# import datasets
# import networks
# from linear_warmup_cosine_annealing_warm_restarts_weight_decay import ChainedScheduler

# class EMA():
#     def __init__(self, model, decay):
#         self.model = model
#         self.decay = decay
#         self.shadow = {}
#         self.backup = {}

#     def register(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 self.shadow[name] = param.data.clone()

#     def update(self):
#         if not self.shadow:      # 空字典表示还没注册
#             self.register()
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 if name not in self.shadow:   # 新增参数直接注册
#                     self.shadow[name] = param.data.clone()
#                 assert name in self.shadow

#                 new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
#                 self.shadow[name] = new_average.clone()

#     def apply_shadow(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 assert name in self.shadow
#                 self.backup[name] = param.data
#                 param.data = self.shadow[name]

#     def restore(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 assert name in self.backup
#                 param.data = self.backup[name]
#         self.backup = {}


# # torch.backends.cudnn.benchmark = True


# def time_sync():
#     # PyTorch-accurate time
#     if torch.cuda.is_available():
#         torch.cuda.synchronize()
#     return time.time()


# class Trainer:
#     def __init__(self, options):
#         self.opt = options
#         self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

#         # checking height and width are multiples of 32
#         assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
#         assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

#         self.models = {}
#         self.models_pose = {}
#         self.parameters_to_train = []
#         self.parameters_to_train_pose = []

#         self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
#         self.profile = self.opt.profile

#         self.num_scales = len(self.opt.scales)
#         self.frame_ids = len(self.opt.frame_ids)
#         self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

#         assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

#         self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

#         if self.opt.use_stereo:
#             self.opt.frame_ids.append("s")

#         self.models["encoder"] = networks.LiteMono(model=self.opt.model,
#                                                    drop_path_rate=self.opt.drop_path,
#                                                    width=self.opt.width, height=self.opt.height)

#         self.models["encoder"].to(self.device)
#         self.parameters_to_train += list(self.models["encoder"].parameters())

#         self.models["depth"] = networks.DepthDecoder(self.models["encoder"].num_ch_enc,
#                                                      self.opt.scales)
#         self.models["depth"].to(self.device)
#         self.parameters_to_train += list(self.models["depth"].parameters())

#         if self.use_pose_net:
#             if self.opt.pose_model_type == "separate_resnet":
#                 self.models_pose["pose_encoder"] = networks.ResnetEncoder(
#                     self.opt.num_layers,
#                     self.opt.weights_init == "pretrained",
#                     num_input_images=self.num_pose_frames)

#                 self.models_pose["pose_encoder"].to(self.device)
#                 self.parameters_to_train_pose += list(self.models_pose["pose_encoder"].parameters())

#                 self.models_pose["pose"] = networks.PoseDecoder(
#                     self.models_pose["pose_encoder"].num_ch_enc,
#                     num_input_features=1,
#                     num_frames_to_predict_for=2)

#             elif self.opt.pose_model_type == "shared":
#                 self.models_pose["pose"] = networks.PoseDecoder(
#                     self.models["encoder"].num_ch_enc, self.num_pose_frames)

#             elif self.opt.pose_model_type == "posecnn":
#                 self.models_pose["pose"] = networks.PoseCNN(
#                     self.num_input_frames if self.opt.pose_model_input == "all" else 2)

#             self.models_pose["pose"].to(self.device)
#             self.parameters_to_train_pose += list(self.models_pose["pose"].parameters())

#         if self.opt.predictive_mask:
#             assert self.opt.disable_automasking, \
#                 "When using predictive_mask, please disable automasking with --disable_automasking"

#             # Our implementation of the predictive masking baseline has the the same architecture
#             # as our depth decoder. We predict a separate mask for each source frame.
#             self.models["predictive_mask"] = networks.DepthDecoder(
#                 self.models["encoder"].num_ch_enc, self.opt.scales,
#                 num_output_channels=(len(self.opt.frame_ids) - 1))
#             self.models["predictive_mask"].to(self.device)
#             self.parameters_to_train += list(self.models["predictive_mask"].parameters())

#         self.model_optimizer = optim.AdamW(self.parameters_to_train, self.opt.lr[0], weight_decay=self.opt.weight_decay)
#         if self.use_pose_net:
#             self.model_pose_optimizer = optim.AdamW(self.parameters_to_train_pose, self.opt.lr[3], weight_decay=self.opt.weight_decay)

#         self.model_lr_scheduler = ChainedScheduler(
#                             self.model_optimizer,
#                             T_0=int(self.opt.lr[2]),
#                             T_mul=1,
#                             eta_min=self.opt.lr[1],
#                             last_epoch=-1,
#                             max_lr=self.opt.lr[0],
#                             warmup_steps=0,
#                             gamma=0.9
#                         )
#         self.model_pose_lr_scheduler = ChainedScheduler(
#             self.model_pose_optimizer,
#             T_0=int(self.opt.lr[5]),
#             T_mul=1,
#             eta_min=self.opt.lr[4],
#             last_epoch=-1,
#             max_lr=self.opt.lr[3],
#             warmup_steps=0,
#             gamma=0.9
#         )

#         if self.opt.load_weights_folder is not None:
#             self.load_model()

#         if self.opt.mypretrain is not None:
#             self.load_pretrain()

#         print("Training model named:\n  ", self.opt.model_name)
#         print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
#         print("Training is using:\n  ", self.device)

#         # data
#         datasets_dict = {"kitti": datasets.KITTIRAWDataset,
#                          "kitti_odom": datasets.KITTIOdomDataset}
#         self.dataset = datasets_dict[self.opt.dataset]

#         fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

#         train_filenames = readlines(fpath.format("train"))
#         val_filenames = readlines(fpath.format("val"))
#         img_ext = '.png'  

#         num_train_samples = len(train_filenames)
#         self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

#         train_dataset = self.dataset(
#             self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
#             self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
#         self.train_loader = DataLoader(
#             train_dataset, self.opt.batch_size, True,
#             num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
#         val_dataset = self.dataset(
#             self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
#             self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
#         self.val_loader = DataLoader(
#             val_dataset, self.opt.batch_size, True,
#             num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
#         self.val_iter = iter(self.val_loader)

#         self.writers = {}
#         for mode in ["train", "val"]:
#             self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

#         if not self.opt.no_ssim:
#             self.ssim = SSIM()
#             self.ssim.to(self.device)

#         self.backproject_depth = {}
#         self.project_3d = {}
#         for scale in self.opt.scales:
#             h = self.opt.height // (2 ** scale)
#             w = self.opt.width // (2 ** scale)

#             self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
#             self.backproject_depth[scale].to(self.device)

#             self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
#             self.project_3d[scale].to(self.device)

#         self.depth_metric_names = [
#             "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

#         print("Using split:\n  ", self.opt.split)
#         print("There are {:d} training items and {:d} validation items\n".format(
#             len(train_dataset), len(val_dataset)))

#         self.save_opts()
#         self.ema = EMA(self.models["encoder"], decay=0.999)

#     def set_train(self):
#         """Convert all models to training mode
#         """
#         for m in self.models.values():
#             m.train()

#     def set_eval(self):
#         """Convert all models to testing/evaluation mode
#         """
#         for m in self.models.values():
#             m.eval()

#     def train(self):
#         """Run the entire training pipeline
#         """
#         self.epoch = 0
#         self.step = 0
#         self.start_time = time.time()
#         for self.epoch in range(self.opt.num_epochs):
#             self.run_epoch()
#             if (self.epoch + 1) % self.opt.save_frequency == 0:
#                 self.save_model()

#     def run_epoch(self):
#         """Run a single epoch of training and validation
#         """

#         print("Training")
#         self.set_train()

#         self.model_lr_scheduler.step()
#         if self.use_pose_net:
#             self.model_pose_lr_scheduler.step()

#         for batch_idx, inputs in enumerate(self.train_loader):

#             before_op_time = time.time()

#             outputs, losses = self.process_batch(inputs)

#             self.model_optimizer.zero_grad()
#             if self.use_pose_net:
#                 self.model_pose_optimizer.zero_grad()
#             losses["loss"].backward()
#             self.model_optimizer.step()
#             self.ema.update() 

#             if self.use_pose_net:
#                 self.model_pose_optimizer.step()

#             duration = time.time() - before_op_time

#             # log less frequently after the first 2000 steps to save time & disk space
#             early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 20000
#             late_phase = self.step % 2000 == 0

#             if early_phase or late_phase:
#                 self.log_time(batch_idx, duration, losses["loss"].cpu().data)

#                 if "depth_gt" in inputs:
#                     self.compute_depth_losses(inputs, outputs, losses)

#                 self.log("train", inputs, outputs, losses)
#                 self.val()

#             self.step += 1

#     def process_batch(self, inputs):
#         """Pass a minibatch through the network and generate images and losses
#         """
#         for key, ipt in inputs.items():
#             inputs[key] = ipt.to(self.device)

#         if self.opt.pose_model_type == "shared":
#             # If we are using a shared encoder for both depth and pose (as advocated
#             # in monodepthv1), then all images are fed separately through the depth encoder.
#             all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
#             all_features = self.models["encoder"](all_color_aug)
#             all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

#             features = {}
#             for i, k in enumerate(self.opt.frame_ids):
#                 features[k] = [f[i] for f in all_features]

#             outputs = self.models["depth"](features[0])
#         else:
#             # Otherwise, we only feed the image with frame_id 0 through the depth encoder

#             features = self.models["encoder"](inputs["color_aug", 0, 0])

#             outputs = self.models["depth"](features)

#         if self.opt.predictive_mask:
#             outputs["predictive_mask"] = self.models["predictive_mask"](features)

#         if self.use_pose_net:
#             outputs.update(self.predict_poses(inputs, features))

#         self.generate_images_pred(inputs, outputs)
#         losses = self.compute_losses(inputs, outputs)

#         return outputs, losses

#     def predict_poses(self, inputs, features):
#         """Predict poses between input frames for monocular sequences.
#         """
#         outputs = {}
#         if self.num_pose_frames == 2:
#             # In this setting, we compute the pose to each source frame via a
#             # separate forward pass through the pose network.

#             # select what features the pose network takes as input
#             if self.opt.pose_model_type == "shared":
#                 pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
#             else:
#                 pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

#             for f_i in self.opt.frame_ids[1:]:
#                 if f_i != "s":
#                     # To maintain ordering we always pass frames in temporal order
#                     if f_i < 0:
#                         pose_inputs = [pose_feats[f_i], pose_feats[0]]
#                     else:
#                         pose_inputs = [pose_feats[0], pose_feats[f_i]]

#                     if self.opt.pose_model_type == "separate_resnet":
#                         pose_inputs = [self.models_pose["pose_encoder"](torch.cat(pose_inputs, 1))]
#                     elif self.opt.pose_model_type == "posecnn":
#                         pose_inputs = torch.cat(pose_inputs, 1)

#                     axisangle, translation = self.models_pose["pose"](pose_inputs)
#                     outputs[("axisangle", 0, f_i)] = axisangle
#                     outputs[("translation", 0, f_i)] = translation

#                     # Invert the matrix if the frame id is negative
#                     outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
#                         axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

#         else:
#             # Here we input all frames to the pose net (and predict all poses) together
#             if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
#                 pose_inputs = torch.cat(
#                     [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

#                 if self.opt.pose_model_type == "separate_resnet":
#                     pose_inputs = [self.models["pose_encoder"](pose_inputs)]

#             elif self.opt.pose_model_type == "shared":
#                 pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

#             axisangle, translation = self.models_pose["pose"](pose_inputs)

#             for i, f_i in enumerate(self.opt.frame_ids[1:]):
#                 if f_i != "s":
#                     outputs[("axisangle", 0, f_i)] = axisangle
#                     outputs[("translation", 0, f_i)] = translation
#                     outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
#                         axisangle[:, i], translation[:, i])

#         return outputs

#     def val(self):
#         """Validate the model on a single minibatch
#         """
#         self.set_eval()
#         self.ema.apply_shadow()
#         try:
#             inputs = self.val_iter.__next__()  
#         except StopIteration:
#             self.val_iter = iter(self.val_loader)
#             inputs = self.val_iter.__next__() 

#         with torch.no_grad():
#             outputs, losses = self.process_batch(inputs)

#             if "depth_gt" in inputs:
#                 self.compute_depth_losses(inputs, outputs, losses)

#             self.log("val", inputs, outputs, losses)
#             del inputs, outputs, losses

#         self.ema.restore()
#         self.set_train()

#     def generate_images_pred(self, inputs, outputs):
#         """Generate the warped (reprojected) color images for a minibatch.
#         Generated images are saved into the `outputs` dictionary.
#         """
#         for scale in self.opt.scales:
#             disp = outputs[("disp", scale)]
#             if self.opt.v1_multiscale:
#                 source_scale = scale
#             else:
#                 disp = F.interpolate(
#                     disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
#                 source_scale = 0

#             _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

#             outputs[("depth", 0, scale)] = depth

#             for i, frame_id in enumerate(self.opt.frame_ids[1:]):

#                 if frame_id == "s":
#                     T = inputs["stereo_T"]
#                 else:
#                     T = outputs[("cam_T_cam", 0, frame_id)]

#                 # from the authors of https://arxiv.org/abs/1712.00175
#                 if self.opt.pose_model_type == "posecnn":

#                     axisangle = outputs[("axisangle", 0, frame_id)]
#                     translation = outputs[("translation", 0, frame_id)]

#                     inv_depth = 1 / depth
#                     mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

#                     T = transformation_from_parameters(
#                         axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

#                 cam_points = self.backproject_depth[source_scale](
#                     depth, inputs[("inv_K", source_scale)])
#                 pix_coords = self.project_3d[source_scale](
#                     cam_points, inputs[("K", source_scale)], T)

#                 outputs[("sample", frame_id, scale)] = pix_coords

#                 outputs[("color", frame_id, scale)] = F.grid_sample(
#                     inputs[("color", frame_id, source_scale)],
#                     outputs[("sample", frame_id, scale)],
#                     padding_mode="border", align_corners=True)

#                 if not self.opt.disable_automasking:
#                     outputs[("color_identity", frame_id, scale)] = \
#                         inputs[("color", frame_id, source_scale)]

#     def compute_reprojection_loss(self, pred, target):
#         """Computes reprojection loss between a batch of predicted and target images
#         """
#         abs_diff = torch.abs(target - pred)
#         l1_loss = abs_diff.mean(1, True)

#         if self.opt.no_ssim:
#             reprojection_loss = l1_loss
#         else:
#             ssim_loss = self.ssim(pred, target).mean(1, True)
#             reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

#         return reprojection_loss

#     def compute_losses(self, inputs, outputs):
#         """Compute the reprojection and smoothness losses for a minibatch
#         """

#         losses = {}
#         total_loss = 0

#         for scale in self.opt.scales:
#             loss = 0
#             reprojection_losses = []

#             if self.opt.v1_multiscale:
#                 source_scale = scale
#             else:
#                 source_scale = 0

#             disp = outputs[("disp", scale)]
#             color = inputs[("color", 0, scale)]
#             target = inputs[("color", 0, source_scale)]

#             for frame_id in self.opt.frame_ids[1:]:
#                 pred = outputs[("color", frame_id, scale)]
#                 reprojection_losses.append(self.compute_reprojection_loss(pred, target))

#             reprojection_losses = torch.cat(reprojection_losses, 1)

#             if not self.opt.disable_automasking:
#                 identity_reprojection_losses = []
#                 for frame_id in self.opt.frame_ids[1:]:
#                     pred = inputs[("color", frame_id, source_scale)]
#                     identity_reprojection_losses.append(
#                         self.compute_reprojection_loss(pred, target))

#                 identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

#                 if self.opt.avg_reprojection:
#                     identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
#                 else:
#                     # save both images, and do min all at once below
#                     identity_reprojection_loss = identity_reprojection_losses

#             elif self.opt.predictive_mask:
#                 # use the predicted mask
#                 mask = outputs["predictive_mask"]["disp", scale]
#                 if not self.opt.v1_multiscale:
#                     mask = F.interpolate(
#                         mask, [self.opt.height, self.opt.width],
#                         mode="bilinear", align_corners=False)

#                 reprojection_losses *= mask

#                 # add a loss pushing mask to 1 (using nn.BCELoss for stability)
#                 weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
#                 loss += weighting_loss.mean()

#             if self.opt.avg_reprojection:
#                 reprojection_loss = reprojection_losses.mean(1, keepdim=True)
#             else:
#                 reprojection_loss = reprojection_losses

#             if not self.opt.disable_automasking:
#                 # add random numbers to break ties
#                 identity_reprojection_loss += torch.randn(
#                     identity_reprojection_loss.shape, device=self.device) * 0.00001

#                 combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
#             else:
#                 combined = reprojection_loss

#             if combined.shape[1] == 1:
#                 to_optimise = combined
#             else:
#                 to_optimise, idxs = torch.min(combined, dim=1)

#             if not self.opt.disable_automasking:
#                 outputs["identity_selection/{}".format(scale)] = (
#                     idxs > identity_reprojection_loss.shape[1] - 1).float()

#             loss += to_optimise.mean()

#             mean_disp = disp.mean(2, True).mean(3, True)
#             norm_disp = disp / (mean_disp + 1e-7)
#             smooth_loss = get_smooth_loss(norm_disp, color)

#             loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
#             total_loss += loss
#             losses["loss/{}".format(scale)] = loss

#         total_loss /= self.num_scales
#         losses["loss"] = total_loss
#         return losses

#     def compute_depth_losses(self, inputs, outputs, losses):
#         """Compute depth metrics, to allow monitoring during training

#         This isn't particularly accurate as it averages over the entire batch,
#         so is only used to give an indication of validation performance
#         """
#         depth_pred = outputs[("depth", 0, 0)]
#         depth_pred = torch.clamp(F.interpolate(
#             depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
#         depth_pred = depth_pred.detach()

#         depth_gt = inputs["depth_gt"]
#         mask = depth_gt > 0

#         # garg/eigen crop
#         crop_mask = torch.zeros_like(mask)
#         crop_mask[:, :, 153:371, 44:1197] = 1
#         mask = mask * crop_mask

#         depth_gt = depth_gt[mask]
#         depth_pred = depth_pred[mask]
#         depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

#         depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

#         depth_errors = compute_depth_errors(depth_gt, depth_pred)

#         for i, metric in enumerate(self.depth_metric_names):
#             losses[metric] = np.array(depth_errors[i].cpu())

#     def log_time(self, batch_idx, duration, loss):
#         """Print a logging statement to the terminal
#         """
#         samples_per_sec = self.opt.batch_size / duration
#         time_sofar = time.time() - self.start_time
#         training_time_left = (
#             self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
#         print_string = "epoch {:>3} | lr {:.6f} |lr_p {:.6f} | batch {:>6} | examples/s: {:5.1f}" + \
#             " | loss: {:.5f} | time elapsed: {} | time left: {}"
#         print(print_string.format(self.epoch, self.model_optimizer.state_dict()['param_groups'][0]['lr'],
#                                   self.model_pose_optimizer.state_dict()['param_groups'][0]['lr'],
#                                   batch_idx, samples_per_sec, loss,
#                                   sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

#     def log(self, mode, inputs, outputs, losses):
#         """Write an event to the tensorboard events file
#         """
#         writer = self.writers[mode]
#         for l, v in losses.items():
#             writer.add_scalar("{}".format(l), v, self.step)

#         for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
#             for s in self.opt.scales:
#                 for frame_id in self.opt.frame_ids:
#                     writer.add_image(
#                         "color_{}_{}/{}".format(frame_id, s, j),
#                         inputs[("color", frame_id, s)][j].data, self.step)
#                     if s == 0 and frame_id != 0:
#                         writer.add_image(
#                             "color_pred_{}_{}/{}".format(frame_id, s, j),
#                             outputs[("color", frame_id, s)][j].data, self.step)

#                 writer.add_image(
#                     "disp_{}/{}".format(s, j),
#                     normalize_image(outputs[("disp", s)][j]), self.step)

#                 if self.opt.predictive_mask:
#                     for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
#                         writer.add_image(
#                             "predictive_mask_{}_{}/{}".format(frame_id, s, j),
#                             outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
#                             self.step)

#                 elif not self.opt.disable_automasking:
#                     writer.add_image(
#                         "automask_{}/{}".format(s, j),
#                         outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

#     def save_opts(self):
#         """Save options to disk so we know what we ran this experiment with
#         """
#         models_dir = os.path.join(self.log_path, "models")
#         if not os.path.exists(models_dir):
#             os.makedirs(models_dir)
#         to_save = self.opt.__dict__.copy()

#         with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
#             json.dump(to_save, f, indent=2)

#     def save_model(self):
#         """Save model weights to disk
#         """
#         save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
#         if not os.path.exists(save_folder):
#             os.makedirs(save_folder)

#         for model_name, model in self.models.items():
#             save_path = os.path.join(save_folder, "{}.pth".format(model_name))
#             to_save = model.state_dict()
#             if model_name == 'encoder':
#                 # ===== EMA：先换权重，再保存，再恢复 =====
#                 self.ema.apply_shadow()
#                 to_save = model.state_dict()
#                 torch.save(to_save, os.path.join(save_folder, "encoder_ema.pth"))
#                 self.ema.restore()
#                 # ===== 原逻辑继续 =====
#                 # save the sizes - these are needed at prediction time
#                 to_save['height'] = self.opt.height
#                 to_save['width'] = self.opt.width
#                 to_save['use_stereo'] = self.opt.use_stereo
#             torch.save(to_save, save_path)

#         for model_name, model in self.models_pose.items():
#             save_path = os.path.join(save_folder, "{}.pth".format(model_name))
#             to_save = model.state_dict()
#             torch.save(to_save, save_path)

#         save_path = os.path.join(save_folder, "{}.pth".format("adam"))
#         torch.save(self.model_optimizer.state_dict(), save_path)

#         save_path = os.path.join(save_folder, "{}.pth".format("adam_pose"))
#         if self.use_pose_net:
#             torch.save(self.model_pose_optimizer.state_dict(), save_path)

#     def load_pretrain(self):
#         self.opt.mypretrain = os.path.expanduser(self.opt.mypretrain)
#         path = self.opt.mypretrain
#         model_dict = self.models["encoder"].state_dict()
#         pretrained_dict = torch.load(path)['model']
#         pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and not k.startswith('norm'))}
#         model_dict.update(pretrained_dict)
#         self.models["encoder"].load_state_dict(model_dict)
#         print('mypretrain loaded.')

#     def load_model(self):
#         """Load model(s) from disk
#         """
#         self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

#         assert os.path.isdir(self.opt.load_weights_folder), \
#             "Cannot find folder {}".format(self.opt.load_weights_folder)
#         print("loading model from folder {}".format(self.opt.load_weights_folder))

#         for n in self.opt.models_to_load:
#             print("Loading {} weights...".format(n))
#             path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))

#             if n in ['pose_encoder', 'pose']:
#                 model_dict = self.models_pose[n].state_dict()
#                 pretrained_dict = torch.load(path)
#                 pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#                 model_dict.update(pretrained_dict)
#                 self.models_pose[n].load_state_dict(model_dict)
#             else:
#                 model_dict = self.models[n].state_dict()
#                 pretrained_dict = torch.load(path)
#                 pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#                 model_dict.update(pretrained_dict)
#                 self.models[n].load_state_dict(model_dict)
#                 # 若存在 EMA 权重，加载它（eval 时用）
#                 ema_path = os.path.join(self.opt.load_weights_folder, "encoder_ema.pth")
#                 if n == "encoder" and os.path.exists(ema_path):
#                     self.ema.apply_shadow()
#                     self.models["encoder"].load_state_dict(torch.load(ema_path))
#                     self.ema.restore()
#         # loading adam state

#         optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
#         optimizer_pose_load_path = os.path.join(self.opt.load_weights_folder, "adam_pose.pth")
#         if os.path.isfile(optimizer_load_path):
#             print("Loading Adam weights")
#             optimizer_dict = torch.load(optimizer_load_path)
#             optimizer_pose_dict = torch.load(optimizer_pose_load_path)
#             self.model_optimizer.load_state_dict(optimizer_dict)
#             self.model_pose_optimizer.load_state_dict(optimizer_pose_dict)
#         else:
#             print("Cannot find Adam weights so Adam is randomly initialized")


# #v1.4.2.2, gpt大改
# from __future__ import absolute_import, division, print_function


# import time
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter

# import json

# from utils import *
# from kitti_utils import *
# from layers import *

# import datasets
# import networks
# from linear_warmup_cosine_annealing_warm_restarts_weight_decay import ChainedScheduler

# # + EMA: 改为一次性注册 + 标准 EMA 更新公式 + 安全性小改
# class EMA:
#     def __init__(self, model, decay=0.999):
#         self.model = model
#         self.decay = decay
#         self.shadow = {}
#         self.backup = {}
#         # 一次性注册所有需要跟踪的参数（避免训练中动态注册带来的不稳定）
#         self.register()

#     def register(self):
#         # 将初始参数拷贝到 shadow（只拷贝 requires_grad 的）
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 self.shadow[name] = param.data.clone()

#     def update(self):
#         # 使用标准 EMA 公式： shadow = decay * shadow + (1 - decay) * param
#         # 请确保在 optimizer.step() 之后调用 update()
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 assert name in self.shadow, f"EMA missing key {name}"
#                 self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data

#     def apply_shadow(self):
#         # 将模型参数替换为 shadow（备份原参数）
#         self.backup = {}
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 assert name in self.shadow, f"EMA missing key {name}"
#                 self.backup[name] = param.data.clone()
#                 param.data.copy_(self.shadow[name])

#     def restore(self):
#         # 恢复模型原始参数
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 assert name in self.backup, f"EMA restore missing key {name}"
#                 param.data.copy_(self.backup[name])
#         self.backup = {}


# # torch.backends.cudnn.benchmark = True


# def time_sync():
#     # PyTorch-accurate time
#     if torch.cuda.is_available():
#         torch.cuda.synchronize()
#     return time.time()


# class Trainer:
#     def __init__(self, options):
#         self.opt = options
#         self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

#         # checking height and width are multiples of 32
#         assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
#         assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

#         self.models = {}
#         self.models_pose = {}
#         self.parameters_to_train = []
#         self.parameters_to_train_pose = []

#         self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
#         self.profile = self.opt.profile

#         self.num_scales = len(self.opt.scales)
#         self.frame_ids = len(self.opt.frame_ids)
#         self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

#         assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

#         self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

#         if self.opt.use_stereo:
#             self.opt.frame_ids.append("s")

#         self.models["encoder"] = networks.LiteMono(model=self.opt.model,
#                                                    drop_path_rate=self.opt.drop_path,
#                                                    width=self.opt.width, height=self.opt.height)

#         self.models["encoder"].to(self.device)
#         self.parameters_to_train += list(self.models["encoder"].parameters())

#         self.models["depth"] = networks.DepthDecoder(self.models["encoder"].num_ch_enc,
#                                                      self.opt.scales)
#         self.models["depth"].to(self.device)
#         self.parameters_to_train += list(self.models["depth"].parameters())

#         if self.use_pose_net:
#             if self.opt.pose_model_type == "separate_resnet":
#                 self.models_pose["pose_encoder"] = networks.ResnetEncoder(
#                     self.opt.num_layers,
#                     self.opt.weights_init == "pretrained",
#                     num_input_images=self.num_pose_frames)

#                 self.models_pose["pose_encoder"].to(self.device)
#                 self.parameters_to_train_pose += list(self.models_pose["pose_encoder"].parameters())

#                 self.models_pose["pose"] = networks.PoseDecoder(
#                     self.models_pose["pose_encoder"].num_ch_enc,
#                     num_input_features=1,
#                     num_frames_to_predict_for=2)

#             elif self.opt.pose_model_type == "shared":
#                 self.models_pose["pose"] = networks.PoseDecoder(
#                     self.models["encoder"].num_ch_enc, self.num_pose_frames)

#             elif self.opt.pose_model_type == "posecnn":
#                 self.models_pose["pose"] = networks.PoseCNN(
#                     self.num_input_frames if self.opt.pose_model_input == "all" else 2)

#             self.models_pose["pose"].to(self.device)
#             self.parameters_to_train_pose += list(self.models_pose["pose"].parameters())

#         if self.opt.predictive_mask:
#             assert self.opt.disable_automasking, \
#                 "When using predictive_mask, please disable automasking with --disable_automasking"

#             # Our implementation of the predictive masking baseline has the the same architecture
#             # as our depth decoder. We predict a separate mask for each source frame.
#             self.models["predictive_mask"] = networks.DepthDecoder(
#                 self.models["encoder"].num_ch_enc, self.opt.scales,
#                 num_output_channels=(len(self.opt.frame_ids) - 1))
#             self.models["predictive_mask"].to(self.device)
#             self.parameters_to_train += list(self.models["predictive_mask"].parameters())

#         # # ========== 双卡 DataParallel ==========
#         # if torch.cuda.device_count() > 1:
#         #     for key in list(self.models.keys()):
#         #         self.models[key] = torch.nn.DataParallel(self.models[key], device_ids=[0,1])
#         #     for key in list(self.models_pose.keys()):
#         #         self.models_pose[key] = torch.nn.DataParallel(self.models_pose[key], device_ids=[0,1])
#         # # =======================================

#         self.model_optimizer = optim.AdamW(self.parameters_to_train, self.opt.lr[0], weight_decay=self.opt.weight_decay)
#         if self.use_pose_net:
#             self.model_pose_optimizer = optim.AdamW(self.parameters_to_train_pose, self.opt.lr[3], weight_decay=self.opt.weight_decay)

#         self.model_lr_scheduler = ChainedScheduler(
#                             self.model_optimizer,
#                             T_0=int(self.opt.lr[2]),
#                             T_mul=1,
#                             eta_min=self.opt.lr[1],
#                             last_epoch=-1,
#                             max_lr=self.opt.lr[0],
#                             warmup_steps=0,
#                             gamma=0.9
#                         )
#         self.model_pose_lr_scheduler = ChainedScheduler(
#             self.model_pose_optimizer,
#             T_0=int(self.opt.lr[5]),
#             T_mul=1,
#             eta_min=self.opt.lr[4],
#             last_epoch=-1,
#             max_lr=self.opt.lr[3],
#             warmup_steps=0,
#             gamma=0.9
#         )

#         if self.opt.load_weights_folder is not None:
#             self.load_model()

#         if self.opt.mypretrain is not None:
#             self.load_pretrain()

#         print("Training model named:\n  ", self.opt.model_name)
#         print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
#         print("Training is using:\n  ", self.device)

#         # data
#         datasets_dict = {"kitti": datasets.KITTIRAWDataset,
#                          "kitti_odom": datasets.KITTIOdomDataset}
#         self.dataset = datasets_dict[self.opt.dataset]

#         fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

#         train_filenames = readlines(fpath.format("train"))
#         val_filenames = readlines(fpath.format("val"))
#         img_ext = '.png'  

#         num_train_samples = len(train_filenames)
#         self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

#         train_dataset = self.dataset(
#             self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
#             self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
#         self.train_loader = DataLoader(
#             train_dataset, self.opt.batch_size, True,
#             num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
#         val_dataset = self.dataset(
#             self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
#             self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
#         self.val_loader = DataLoader(
#             val_dataset, self.opt.batch_size, True,
#             num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
#         self.val_iter = iter(self.val_loader)

#         self.writers = {}
#         for mode in ["train", "val"]:
#             self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

#         if not self.opt.no_ssim:
#             self.ssim = SSIM()
#             self.ssim.to(self.device)

#         self.backproject_depth = {}
#         self.project_3d = {}
#         for scale in self.opt.scales:
#             h = self.opt.height // (2 ** scale)
#             w = self.opt.width // (2 ** scale)

#             self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
#             self.backproject_depth[scale].to(self.device)

#             self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
#             self.project_3d[scale].to(self.device)

#         self.depth_metric_names = [
#             "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

#         print("Using split:\n  ", self.opt.split)
#         print("There are {:d} training items and {:d} validation items\n".format(
#             len(train_dataset), len(val_dataset)))

#         self.save_opts()
#         self.ema = EMA(self.models["encoder"], decay=0.999)

#     def set_train(self):
#         """Convert all models to training mode
#         """
#         for m in self.models.values():
#             m.train()

#     def set_eval(self):
#         """Convert all models to testing/evaluation mode
#         """
#         for m in self.models.values():
#             m.eval()

#     def train(self):
#         """Run the entire training pipeline
#         """
#         self.epoch = 0
#         self.step = 0
#         self.start_time = time.time()
#         for self.epoch in range(self.opt.num_epochs):
#             self.run_epoch()
#             if (self.epoch + 1) % self.opt.save_frequency == 0:
#                 self.save_model()

#     def run_epoch(self):
#         """Run a single epoch of training and validation
#         """

#         print("Training")
#         self.set_train()

#         self.model_lr_scheduler.step()
#         if self.use_pose_net:
#             self.model_pose_lr_scheduler.step()

#         for batch_idx, inputs in enumerate(self.train_loader):

#             before_op_time = time.time()

#             outputs, losses = self.process_batch(inputs)

#             self.model_optimizer.zero_grad()
#             if self.use_pose_net:
#                 self.model_pose_optimizer.zero_grad()
#             losses["loss"].backward()
#             self.model_optimizer.step()
#             if self.epoch >= 2:          # 或 ema_start_epoch 配置
#                 self.ema.update()

#             if self.use_pose_net:
#                 self.model_pose_optimizer.step()

#             duration = time.time() - before_op_time

#             # log less frequently after the first 2000 steps to save time & disk space
#             early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 20000
#             late_phase = self.step % 2000 == 0

#             if early_phase or late_phase:
#                 self.log_time(batch_idx, duration, losses["loss"].cpu().data)

#                 if "depth_gt" in inputs:
#                     self.compute_depth_losses(inputs, outputs, losses)

#                 self.log("train", inputs, outputs, losses)
#                 self.val()

#             self.step += 1

#     def process_batch(self, inputs):
#         """Pass a minibatch through the network and generate images and losses
#         """
#         for key, ipt in inputs.items():
#             inputs[key] = ipt.to(self.device)

#         if self.opt.pose_model_type == "shared":
#             # If we are using a shared encoder for both depth and pose (as advocated
#             # in monodepthv1), then all images are fed separately through the depth encoder.
#             all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
#             all_features = self.models["encoder"](all_color_aug)
#             all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

#             features = {}
#             for i, k in enumerate(self.opt.frame_ids):
#                 features[k] = [f[i] for f in all_features]

#             outputs = self.models["depth"](features[0])
#         else:
#             # Otherwise, we only feed the image with frame_id 0 through the depth encoder

#             features = self.models["encoder"](inputs["color_aug", 0, 0])

#             outputs = self.models["depth"](features)

#         if self.opt.predictive_mask:
#             outputs["predictive_mask"] = self.models["predictive_mask"](features)

#         if self.use_pose_net:
#             outputs.update(self.predict_poses(inputs, features))

#         self.generate_images_pred(inputs, outputs)
#         losses = self.compute_losses(inputs, outputs)

#         return outputs, losses

#     def predict_poses(self, inputs, features):
#         """Predict poses between input frames for monocular sequences.
#         """
#         outputs = {}
#         if self.num_pose_frames == 2:
#             # In this setting, we compute the pose to each source frame via a
#             # separate forward pass through the pose network.

#             # select what features the pose network takes as input
#             if self.opt.pose_model_type == "shared":
#                 pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
#             else:
#                 pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

#             for f_i in self.opt.frame_ids[1:]:
#                 if f_i != "s":
#                     # To maintain ordering we always pass frames in temporal order
#                     if f_i < 0:
#                         pose_inputs = [pose_feats[f_i], pose_feats[0]]
#                     else:
#                         pose_inputs = [pose_feats[0], pose_feats[f_i]]

#                     if self.opt.pose_model_type == "separate_resnet":
#                         pose_inputs = [self.models_pose["pose_encoder"](torch.cat(pose_inputs, 1))]
#                     elif self.opt.pose_model_type == "posecnn":
#                         pose_inputs = torch.cat(pose_inputs, 1)

#                     axisangle, translation = self.models_pose["pose"](pose_inputs)
#                     outputs[("axisangle", 0, f_i)] = axisangle
#                     outputs[("translation", 0, f_i)] = translation

#                     # Invert the matrix if the frame id is negative
#                     outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
#                         axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

#         else:
#             # Here we input all frames to the pose net (and predict all poses) together
#             if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
#                 pose_inputs = torch.cat(
#                     [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

#                 if self.opt.pose_model_type == "separate_resnet":
#                     pose_inputs = [self.models["pose_encoder"](pose_inputs)]

#             elif self.opt.pose_model_type == "shared":
#                 pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

#             axisangle, translation = self.models_pose["pose"](pose_inputs)

#             for i, f_i in enumerate(self.opt.frame_ids[1:]):
#                 if f_i != "s":
#                     outputs[("axisangle", 0, f_i)] = axisangle
#                     outputs[("translation", 0, f_i)] = translation
#                     outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
#                         axisangle[:, i], translation[:, i])

#         return outputs

#     def val(self):
#         """Validate the model on a single minibatch
#         """
#         self.set_eval()
#         self.ema.apply_shadow()
#         try:
#             inputs = self.val_iter.__next__()  
#         except StopIteration:
#             self.val_iter = iter(self.val_loader)
#             inputs = self.val_iter.__next__() 

#         with torch.no_grad():
#             outputs, losses = self.process_batch(inputs)

#             if "depth_gt" in inputs:
#                 self.compute_depth_losses(inputs, outputs, losses)

#             self.log("val", inputs, outputs, losses)
#             del inputs, outputs, losses

#         self.ema.restore()
#         self.set_train()

#     def generate_images_pred(self, inputs, outputs):
#         """Generate the warped (reprojected) color images for a minibatch.
#         Generated images are saved into the `outputs` dictionary.
#         """
#         for scale in self.opt.scales:
#             disp = outputs[("disp", scale)]
#             if self.opt.v1_multiscale:
#                 source_scale = scale
#             else:
#                 disp = F.interpolate(
#                     disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
#                 source_scale = 0

#             _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

#             outputs[("depth", 0, scale)] = depth

#             for i, frame_id in enumerate(self.opt.frame_ids[1:]):

#                 if frame_id == "s":
#                     T = inputs["stereo_T"]
#                 else:
#                     T = outputs[("cam_T_cam", 0, frame_id)]

#                 # from the authors of https://arxiv.org/abs/1712.00175
#                 if self.opt.pose_model_type == "posecnn":

#                     axisangle = outputs[("axisangle", 0, frame_id)]
#                     translation = outputs[("translation", 0, frame_id)]

#                     inv_depth = 1 / depth
#                     mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

#                     T = transformation_from_parameters(
#                         axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

#                 cam_points = self.backproject_depth[source_scale](
#                     depth, inputs[("inv_K", source_scale)])
#                 pix_coords = self.project_3d[source_scale](
#                     cam_points, inputs[("K", source_scale)], T)

#                 outputs[("sample", frame_id, scale)] = pix_coords

#                 outputs[("color", frame_id, scale)] = F.grid_sample(
#                     inputs[("color", frame_id, source_scale)],
#                     outputs[("sample", frame_id, scale)],
#                     padding_mode="border", align_corners=True)

#                 if not self.opt.disable_automasking:
#                     outputs[("color_identity", frame_id, scale)] = \
#                         inputs[("color", frame_id, source_scale)]

#     def compute_reprojection_loss(self, pred, target):
#         """Computes reprojection loss between a batch of predicted and target images
#         """
#         abs_diff = torch.abs(target - pred)
#         l1_loss = abs_diff.mean(1, True)

#         if self.opt.no_ssim:
#             reprojection_loss = l1_loss
#         else:
#             ssim_loss = self.ssim(pred, target).mean(1, True)
#             reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

#         return reprojection_loss

#     def compute_losses(self, inputs, outputs):
#         """Compute the reprojection and smoothness losses for a minibatch
#         """

#         losses = {}
#         total_loss = 0

#         for scale in self.opt.scales:
#             loss = 0
#             reprojection_losses = []

#             if self.opt.v1_multiscale:
#                 source_scale = scale
#             else:
#                 source_scale = 0

#             disp = outputs[("disp", scale)]
#             color = inputs[("color", 0, scale)]
#             target = inputs[("color", 0, source_scale)]

#             for frame_id in self.opt.frame_ids[1:]:
#                 pred = outputs[("color", frame_id, scale)]
#                 reprojection_losses.append(self.compute_reprojection_loss(pred, target))

#             reprojection_losses = torch.cat(reprojection_losses, 1)

#             if not self.opt.disable_automasking:
#                 identity_reprojection_losses = []
#                 for frame_id in self.opt.frame_ids[1:]:
#                     pred = inputs[("color", frame_id, source_scale)]
#                     identity_reprojection_losses.append(
#                         self.compute_reprojection_loss(pred, target))

#                 identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

#                 if self.opt.avg_reprojection:
#                     identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
#                 else:
#                     # save both images, and do min all at once below
#                     identity_reprojection_loss = identity_reprojection_losses

#             elif self.opt.predictive_mask:
#                 # use the predicted mask
#                 mask = outputs["predictive_mask"]["disp", scale]
#                 if not self.opt.v1_multiscale:
#                     mask = F.interpolate(
#                         mask, [self.opt.height, self.opt.width],
#                         mode="bilinear", align_corners=False)

#                 reprojection_losses *= mask

#                 # add a loss pushing mask to 1 (using nn.BCELoss for stability)
#                 weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
#                 loss += weighting_loss.mean()

#             if self.opt.avg_reprojection:
#                 reprojection_loss = reprojection_losses.mean(1, keepdim=True)
#             else:
#                 reprojection_loss = reprojection_losses

#             if not self.opt.disable_automasking:
#                 # add random numbers to break ties
#                 identity_reprojection_loss += torch.randn(
#                     identity_reprojection_loss.shape, device=self.device) * 0.00001

#                 combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
#             else:
#                 combined = reprojection_loss

#             if combined.shape[1] == 1:
#                 to_optimise = combined
#             else:
#                 to_optimise, idxs = torch.min(combined, dim=1)

#             if not self.opt.disable_automasking:
#                 outputs["identity_selection/{}".format(scale)] = (
#                     idxs > identity_reprojection_loss.shape[1] - 1).float()

#             loss += to_optimise.mean()

#             mean_disp = disp.mean(2, True).mean(3, True)
#             norm_disp = disp / (mean_disp + 1e-7)
#             smooth_loss = get_smooth_loss(norm_disp, color)

#             loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
#             total_loss += loss
#             losses["loss/{}".format(scale)] = loss

#         total_loss /= self.num_scales
#         losses["loss"] = total_loss
#         return losses

#     def compute_depth_losses(self, inputs, outputs, losses):
#         """Compute depth metrics, to allow monitoring during training

#         This isn't particularly accurate as it averages over the entire batch,
#         so is only used to give an indication of validation performance
#         """
#         depth_pred = outputs[("depth", 0, 0)]
#         depth_pred = torch.clamp(F.interpolate(
#             depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
#         depth_pred = depth_pred.detach()

#         depth_gt = inputs["depth_gt"]
#         mask = depth_gt > 0

#         # garg/eigen crop
#         crop_mask = torch.zeros_like(mask)
#         crop_mask[:, :, 153:371, 44:1197] = 1
#         mask = mask * crop_mask

#         depth_gt = depth_gt[mask]
#         depth_pred = depth_pred[mask]
#         depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

#         depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

#         depth_errors = compute_depth_errors(depth_gt, depth_pred)

#         for i, metric in enumerate(self.depth_metric_names):
#             losses[metric] = np.array(depth_errors[i].cpu())

#     def log_time(self, batch_idx, duration, loss):
#         """Print a logging statement to the terminal
#         """
#         samples_per_sec = self.opt.batch_size / duration
#         time_sofar = time.time() - self.start_time
#         training_time_left = (
#             self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
#         print_string = "epoch {:>3} | lr {:.6f} |lr_p {:.6f} | batch {:>6} | examples/s: {:5.1f}" + \
#             " | loss: {:.5f} | time elapsed: {} | time left: {}"
#         print(print_string.format(self.epoch, self.model_optimizer.state_dict()['param_groups'][0]['lr'],
#                                   self.model_pose_optimizer.state_dict()['param_groups'][0]['lr'],
#                                   batch_idx, samples_per_sec, loss,
#                                   sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

#     def log(self, mode, inputs, outputs, losses):
#         """Write an event to the tensorboard events file
#         """
#         writer = self.writers[mode]
#         for l, v in losses.items():
#             writer.add_scalar("{}".format(l), v, self.step)

#         for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
#             for s in self.opt.scales:
#                 for frame_id in self.opt.frame_ids:
#                     writer.add_image(
#                         "color_{}_{}/{}".format(frame_id, s, j),
#                         inputs[("color", frame_id, s)][j].data, self.step)
#                     if s == 0 and frame_id != 0:
#                         writer.add_image(
#                             "color_pred_{}_{}/{}".format(frame_id, s, j),
#                             outputs[("color", frame_id, s)][j].data, self.step)

#                 writer.add_image(
#                     "disp_{}/{}".format(s, j),
#                     normalize_image(outputs[("disp", s)][j]), self.step)

#                 if self.opt.predictive_mask:
#                     for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
#                         writer.add_image(
#                             "predictive_mask_{}_{}/{}".format(frame_id, s, j),
#                             outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
#                             self.step)

#                 elif not self.opt.disable_automasking:
#                     writer.add_image(
#                         "automask_{}/{}".format(s, j),
#                         outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

#     def save_opts(self):
#         """Save options to disk so we know what we ran this experiment with
#         """
#         models_dir = os.path.join(self.log_path, "models")
#         if not os.path.exists(models_dir):
#             os.makedirs(models_dir)
#         to_save = self.opt.__dict__.copy()

#         with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
#             json.dump(to_save, f, indent=2)

#     def save_model(self):
#         """Save model weights to disk
#         """
#         save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
#         if not os.path.exists(save_folder):
#             os.makedirs(save_folder)

#         for model_name, model in self.models.items():
#             save_path = os.path.join(save_folder, "{}.pth".format(model_name))
#             to_save = model.state_dict()
#             if model_name == 'encoder':
#                 # ===== EMA：先换权重，再保存，再恢复 =====
#                 self.ema.apply_shadow()
#                 to_save = model.state_dict()
#                 torch.save(to_save, os.path.join(save_folder, "encoder_ema.pth"))
#                 self.ema.restore()
#                 # ===== 原逻辑继续 =====
#                 # save the sizes - these are needed at prediction time
#                 to_save['height'] = self.opt.height
#                 to_save['width'] = self.opt.width
#                 to_save['use_stereo'] = self.opt.use_stereo
#             torch.save(to_save, save_path)

#         for model_name, model in self.models_pose.items():
#             save_path = os.path.join(save_folder, "{}.pth".format(model_name))
#             to_save = model.state_dict()
#             torch.save(to_save, save_path)

#         save_path = os.path.join(save_folder, "{}.pth".format("adam"))
#         torch.save(self.model_optimizer.state_dict(), save_path)

#         save_path = os.path.join(save_folder, "{}.pth".format("adam_pose"))
#         if self.use_pose_net:
#             torch.save(self.model_pose_optimizer.state_dict(), save_path)

#     def load_pretrain(self):
#         self.opt.mypretrain = os.path.expanduser(self.opt.mypretrain)
#         path = self.opt.mypretrain
#         model_dict = self.models["encoder"].state_dict()
#         pretrained_dict = torch.load(path)['model']
#         pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and not k.startswith('norm'))}
#         model_dict.update(pretrained_dict)
#         self.models["encoder"].load_state_dict(model_dict)
#         print('mypretrain loaded.')

#     def load_model(self):
#         """Load model(s) from disk
#         """
#         self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

#         assert os.path.isdir(self.opt.load_weights_folder), \
#             "Cannot find folder {}".format(self.opt.load_weights_folder)
#         print("loading model from folder {}".format(self.opt.load_weights_folder))

#         for n in self.opt.models_to_load:
#             print("Loading {} weights...".format(n))
#             path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))

#             if n in ['pose_encoder', 'pose']:
#                 model_dict = self.models_pose[n].state_dict()
#                 pretrained_dict = torch.load(path)
#                 pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#                 model_dict.update(pretrained_dict)
#                 self.models_pose[n].load_state_dict(model_dict)
#             else:
#                 model_dict = self.models[n].state_dict()
#                 pretrained_dict = torch.load(path)
#                 pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#                 model_dict.update(pretrained_dict)
#                 self.models[n].load_state_dict(model_dict)
#                 # 若存在 EMA 权重，加载它（eval 时用）
#                 ema_path = os.path.join(self.opt.load_weights_folder, "encoder_ema.pth")
#                 if n == "encoder" and os.path.exists(ema_path):
#                     # ⑤ 把 encoder_ema.pth 填入 shadow，而非直接替换模型
#                     ema_state = torch.load(ema_path, map_location="cpu")
#                     for k, v in ema_state.items():
#                         if k in self.ema.shadow:
#                             self.ema.shadow[k] = v.to(self.device)
#                     print("Loaded encoder_ema.pth into EMA shadow")
#         # loading adam state

#         optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
#         optimizer_pose_load_path = os.path.join(self.opt.load_weights_folder, "adam_pose.pth")
#         if os.path.isfile(optimizer_load_path):
#             print("Loading Adam weights")
#             optimizer_dict = torch.load(optimizer_load_path)
#             optimizer_pose_dict = torch.load(optimizer_pose_load_path)
#             self.model_optimizer.load_state_dict(optimizer_dict)
#             self.model_pose_optimizer.load_state_dict(optimizer_pose_dict)
#         else:
#             print("Cannot find Adam weights so Adam is randomly initialized")

#v1.4.2.3, gpt大改
# from __future__ import absolute_import, division, print_function


# import time
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter

# import json

# from utils import *
# from kitti_utils import *
# from layers import *

# import datasets
# import networks
# from linear_warmup_cosine_annealing_warm_restarts_weight_decay import ChainedScheduler

# class EMA():
#     def __init__(self, model, decay):
#         self.model = model
#         self.decay = decay
#         self.shadow = {}
#         self.backup = {}

#     def register(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 self.shadow[name] = param.data.clone()

#     def update(self):
#         if not self.shadow:      # 空字典表示还没注册
#             self.register()
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 if name not in self.shadow:   # 新增参数直接注册
#                     self.shadow[name] = param.data.clone()
#                 assert name in self.shadow

#                 new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
#                 self.shadow[name] = new_average.clone()

#     def apply_shadow(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 assert name in self.shadow
#                 self.backup[name] = param.data
#                 param.data = self.shadow[name]

#     def restore(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 assert name in self.backup
#                 param.data = self.backup[name]
#         self.backup = {}


# # torch.backends.cudnn.benchmark = True


# def time_sync():
#     # PyTorch-accurate time
#     if torch.cuda.is_available():
#         torch.cuda.synchronize()
#     return time.time()


# class Trainer:
#     def __init__(self, options):
#         self.opt = options
#         self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

#         # checking height and width are multiples of 32
#         assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
#         assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

#         self.models = {}
#         self.models_pose = {}
#         self.parameters_to_train = []
#         self.parameters_to_train_pose = []

#         self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
#         self.profile = self.opt.profile

#         self.num_scales = len(self.opt.scales)
#         self.frame_ids = len(self.opt.frame_ids)
#         self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

#         assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

#         self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

#         if self.opt.use_stereo:
#             self.opt.frame_ids.append("s")

#         self.models["encoder"] = networks.LiteMono(model=self.opt.model,
#                                                    drop_path_rate=self.opt.drop_path,
#                                                    width=self.opt.width, height=self.opt.height)

#         self.models["encoder"].to(self.device)
#         self.parameters_to_train += list(self.models["encoder"].parameters())

#         self.models["depth"] = networks.DepthDecoder(self.models["encoder"].num_ch_enc,
#                                                      self.opt.scales)
#         self.models["depth"].to(self.device)
#         self.parameters_to_train += list(self.models["depth"].parameters())

#         if self.use_pose_net:
#             if self.opt.pose_model_type == "separate_resnet":
#                 self.models_pose["pose_encoder"] = networks.ResnetEncoder(
#                     self.opt.num_layers,
#                     self.opt.weights_init == "pretrained",
#                     num_input_images=self.num_pose_frames)

#                 self.models_pose["pose_encoder"].to(self.device)
#                 self.parameters_to_train_pose += list(self.models_pose["pose_encoder"].parameters())

#                 self.models_pose["pose"] = networks.PoseDecoder(
#                     self.models_pose["pose_encoder"].num_ch_enc,
#                     num_input_features=1,
#                     num_frames_to_predict_for=2)

#             elif self.opt.pose_model_type == "shared":
#                 self.models_pose["pose"] = networks.PoseDecoder(
#                     self.models["encoder"].num_ch_enc, self.num_pose_frames)

#             elif self.opt.pose_model_type == "posecnn":
#                 self.models_pose["pose"] = networks.PoseCNN(
#                     self.num_input_frames if self.opt.pose_model_input == "all" else 2)

#             self.models_pose["pose"].to(self.device)
#             self.parameters_to_train_pose += list(self.models_pose["pose"].parameters())

#         if self.opt.predictive_mask:
#             assert self.opt.disable_automasking, \
#                 "When using predictive_mask, please disable automasking with --disable_automasking"

#             # Our implementation of the predictive masking baseline has the the same architecture
#             # as our depth decoder. We predict a separate mask for each source frame.
#             self.models["predictive_mask"] = networks.DepthDecoder(
#                 self.models["encoder"].num_ch_enc, self.opt.scales,
#                 num_output_channels=(len(self.opt.frame_ids) - 1))
#             self.models["predictive_mask"].to(self.device)
#             self.parameters_to_train += list(self.models["predictive_mask"].parameters())

#         self.model_optimizer = optim.AdamW(self.parameters_to_train, self.opt.lr[0], weight_decay=self.opt.weight_decay)
#         if self.use_pose_net:
#             self.model_pose_optimizer = optim.AdamW(self.parameters_to_train_pose, self.opt.lr[3], weight_decay=self.opt.weight_decay)

#         self.model_lr_scheduler = ChainedScheduler(
#                             self.model_optimizer,
#                             T_0=int(self.opt.lr[2]),
#                             T_mul=1,
#                             eta_min=self.opt.lr[1],
#                             last_epoch=-1,
#                             max_lr=self.opt.lr[0],
#                             warmup_steps=0,
#                             gamma=0.9
#                         )
#         self.model_pose_lr_scheduler = ChainedScheduler(
#             self.model_pose_optimizer,
#             T_0=int(self.opt.lr[5]),
#             T_mul=1,
#             eta_min=self.opt.lr[4],
#             last_epoch=-1,
#             max_lr=self.opt.lr[3],
#             warmup_steps=0,
#             gamma=0.9
#         )

#         if self.opt.load_weights_folder is not None:
#             self.load_model()

#         if self.opt.mypretrain is not None:
#             self.load_pretrain()

#         print("Training model named:\n  ", self.opt.model_name)
#         print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
#         print("Training is using:\n  ", self.device)

#         # data
#         datasets_dict = {"kitti": datasets.KITTIRAWDataset,
#                          "kitti_odom": datasets.KITTIOdomDataset}
#         self.dataset = datasets_dict[self.opt.dataset]

#         fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

#         train_filenames = readlines(fpath.format("train"))
#         val_filenames = readlines(fpath.format("val"))
#         img_ext = '.png'  

#         num_train_samples = len(train_filenames)
#         self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

#         train_dataset = self.dataset(
#             self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
#             self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
#         self.train_loader = DataLoader(
#             train_dataset, self.opt.batch_size, True,
#             num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
#         val_dataset = self.dataset(
#             self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
#             self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
#         self.val_loader = DataLoader(
#             val_dataset, self.opt.batch_size, True,
#             num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
#         self.val_iter = iter(self.val_loader)

#         self.writers = {}
#         for mode in ["train", "val"]:
#             self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

#         if not self.opt.no_ssim:
#             self.ssim = SSIM()
#             self.ssim.to(self.device)

#         self.backproject_depth = {}
#         self.project_3d = {}
#         for scale in self.opt.scales:
#             h = self.opt.height // (2 ** scale)
#             w = self.opt.width // (2 ** scale)

#             self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
#             self.backproject_depth[scale].to(self.device)

#             self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
#             self.project_3d[scale].to(self.device)

#         self.depth_metric_names = [
#             "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

#         print("Using split:\n  ", self.opt.split)
#         print("There are {:d} training items and {:d} validation items\n".format(
#             len(train_dataset), len(val_dataset)))

#         self.save_opts()
#         self.ema = EMA(self.models["encoder"], decay=0.999)

#     def set_train(self):
#         """Convert all models to training mode
#         """
#         for m in self.models.values():
#             m.train()

#     def set_eval(self):
#         """Convert all models to testing/evaluation mode
#         """
#         for m in self.models.values():
#             m.eval()

#     def train(self):
#         """Run the entire training pipeline
#         """
#         self.epoch = 0
#         self.step = 0
#         self.start_time = time.time()
#         for self.epoch in range(self.opt.num_epochs):
#             self.run_epoch()
#             if (self.epoch + 1) % self.opt.save_frequency == 0:
#                 self.save_model()

#     def run_epoch(self):
#         """Run a single epoch of training and validation
#         """

#         print("Training")
#         self.set_train()

#         self.model_lr_scheduler.step()
#         if self.use_pose_net:
#             self.model_pose_lr_scheduler.step()

#         for batch_idx, inputs in enumerate(self.train_loader):

#             before_op_time = time.time()

#             outputs, losses = self.process_batch(inputs)

#             self.model_optimizer.zero_grad()
#             if self.use_pose_net:
#                 self.model_pose_optimizer.zero_grad()
#             losses["loss"].backward()
#             self.model_optimizer.step()
#             self.ema.update() 

#             if self.use_pose_net:
#                 self.model_pose_optimizer.step()

#             duration = time.time() - before_op_time

#             # log less frequently after the first 2000 steps to save time & disk space
#             early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 20000
#             late_phase = self.step % 2000 == 0

#             if early_phase or late_phase:
#                 self.log_time(batch_idx, duration, losses["loss"].cpu().data)

#                 if "depth_gt" in inputs:
#                     self.compute_depth_losses(inputs, outputs, losses)

#                 self.log("train", inputs, outputs, losses)
#                 self.val()

#             self.step += 1

#     def process_batch(self, inputs):
#         """Pass a minibatch through the network and generate images and losses
#         """
#         for key, ipt in inputs.items():
#             inputs[key] = ipt.to(self.device)

#         if self.opt.pose_model_type == "shared":
#             # If we are using a shared encoder for both depth and pose (as advocated
#             # in monodepthv1), then all images are fed separately through the depth encoder.
#             all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
#             all_features = self.models["encoder"](all_color_aug)
#             all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

#             features = {}
#             for i, k in enumerate(self.opt.frame_ids):
#                 features[k] = [f[i] for f in all_features]

#             outputs = self.models["depth"](features[0])
#         else:
#             # Otherwise, we only feed the image with frame_id 0 through the depth encoder

#             features = self.models["encoder"](inputs["color_aug", 0, 0])

#             outputs = self.models["depth"](features)

#         if self.opt.predictive_mask:
#             outputs["predictive_mask"] = self.models["predictive_mask"](features)

#         if self.use_pose_net:
#             outputs.update(self.predict_poses(inputs, features))

#         self.generate_images_pred(inputs, outputs)
#         losses = self.compute_losses(inputs, outputs)

#         return outputs, losses

#     def predict_poses(self, inputs, features):
#         """Predict poses between input frames for monocular sequences.
#         """
#         outputs = {}
#         if self.num_pose_frames == 2:
#             # In this setting, we compute the pose to each source frame via a
#             # separate forward pass through the pose network.

#             # select what features the pose network takes as input
#             if self.opt.pose_model_type == "shared":
#                 pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
#             else:
#                 pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

#             for f_i in self.opt.frame_ids[1:]:
#                 if f_i != "s":
#                     # To maintain ordering we always pass frames in temporal order
#                     if f_i < 0:
#                         pose_inputs = [pose_feats[f_i], pose_feats[0]]
#                     else:
#                         pose_inputs = [pose_feats[0], pose_feats[f_i]]

#                     if self.opt.pose_model_type == "separate_resnet":
#                         pose_inputs = [self.models_pose["pose_encoder"](torch.cat(pose_inputs, 1))]
#                     elif self.opt.pose_model_type == "posecnn":
#                         pose_inputs = torch.cat(pose_inputs, 1)

#                     axisangle, translation = self.models_pose["pose"](pose_inputs)
#                     outputs[("axisangle", 0, f_i)] = axisangle
#                     outputs[("translation", 0, f_i)] = translation

#                     # Invert the matrix if the frame id is negative
#                     outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
#                         axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

#         else:
#             # Here we input all frames to the pose net (and predict all poses) together
#             if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
#                 pose_inputs = torch.cat(
#                     [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

#                 if self.opt.pose_model_type == "separate_resnet":
#                     pose_inputs = [self.models["pose_encoder"](pose_inputs)]

#             elif self.opt.pose_model_type == "shared":
#                 pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

#             axisangle, translation = self.models_pose["pose"](pose_inputs)

#             for i, f_i in enumerate(self.opt.frame_ids[1:]):
#                 if f_i != "s":
#                     outputs[("axisangle", 0, f_i)] = axisangle
#                     outputs[("translation", 0, f_i)] = translation
#                     outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
#                         axisangle[:, i], translation[:, i])

#         return outputs

#     def val(self):
#         """Validate the model on a single minibatch
#         """
#         self.set_eval()
#         self.ema.apply_shadow()
#         try:
#             inputs = self.val_iter.__next__()  
#         except StopIteration:
#             self.val_iter = iter(self.val_loader)
#             inputs = self.val_iter.__next__() 

#         with torch.no_grad():
#             outputs, losses = self.process_batch(inputs)

#             if "depth_gt" in inputs:
#                 self.compute_depth_losses(inputs, outputs, losses)

#             self.log("val", inputs, outputs, losses)
#             del inputs, outputs, losses

#         self.ema.restore()
#         self.set_train()

#     def generate_images_pred(self, inputs, outputs):
#         """Generate the warped (reprojected) color images for a minibatch.
#         Generated images are saved into the `outputs` dictionary.
#         """
#         for scale in self.opt.scales:
#             disp = outputs[("disp", scale)]
#             if self.opt.v1_multiscale:
#                 source_scale = scale
#             else:
#                 disp = F.interpolate(
#                     disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
#                 source_scale = 0

#             _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

#             outputs[("depth", 0, scale)] = depth

#             for i, frame_id in enumerate(self.opt.frame_ids[1:]):

#                 if frame_id == "s":
#                     T = inputs["stereo_T"]
#                 else:
#                     T = outputs[("cam_T_cam", 0, frame_id)]

#                 # from the authors of https://arxiv.org/abs/1712.00175
#                 if self.opt.pose_model_type == "posecnn":

#                     axisangle = outputs[("axisangle", 0, frame_id)]
#                     translation = outputs[("translation", 0, frame_id)]

#                     inv_depth = 1 / depth
#                     mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

#                     T = transformation_from_parameters(
#                         axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

#                 cam_points = self.backproject_depth[source_scale](
#                     depth, inputs[("inv_K", source_scale)])
#                 pix_coords = self.project_3d[source_scale](
#                     cam_points, inputs[("K", source_scale)], T)

#                 outputs[("sample", frame_id, scale)] = pix_coords

#                 outputs[("color", frame_id, scale)] = F.grid_sample(
#                     inputs[("color", frame_id, source_scale)],
#                     outputs[("sample", frame_id, scale)],
#                     padding_mode="border", align_corners=True)

#                 if not self.opt.disable_automasking:
#                     outputs[("color_identity", frame_id, scale)] = \
#                         inputs[("color", frame_id, source_scale)]

#     def compute_reprojection_loss(self, pred, target):
#         """Computes reprojection loss between a batch of predicted and target images
#         """
#         abs_diff = torch.abs(target - pred)
#         l1_loss = abs_diff.mean(1, True)

#         if self.opt.no_ssim:
#             reprojection_loss = l1_loss
#         else:
#             ssim_loss = self.ssim(pred, target).mean(1, True)
#             reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

#         return reprojection_loss

#     def compute_losses(self, inputs, outputs):
#         """Compute the reprojection and smoothness losses for a minibatch
#         """

#         losses = {}
#         total_loss = 0

#         for scale in self.opt.scales:
#             loss = 0
#             reprojection_losses = []

#             if self.opt.v1_multiscale:
#                 source_scale = scale
#             else:
#                 source_scale = 0

#             disp = outputs[("disp", scale)]
#             color = inputs[("color", 0, scale)]
#             target = inputs[("color", 0, source_scale)]

#             for frame_id in self.opt.frame_ids[1:]:
#                 pred = outputs[("color", frame_id, scale)]
#                 reprojection_losses.append(self.compute_reprojection_loss(pred, target))

#             reprojection_losses = torch.cat(reprojection_losses, 1)

#             if not self.opt.disable_automasking:
#                 identity_reprojection_losses = []
#                 for frame_id in self.opt.frame_ids[1:]:
#                     pred = inputs[("color", frame_id, source_scale)]
#                     identity_reprojection_losses.append(
#                         self.compute_reprojection_loss(pred, target))

#                 identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

#                 if self.opt.avg_reprojection:
#                     identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
#                 else:
#                     # save both images, and do min all at once below
#                     identity_reprojection_loss = identity_reprojection_losses

#             elif self.opt.predictive_mask:
#                 # use the predicted mask
#                 mask = outputs["predictive_mask"]["disp", scale]
#                 if not self.opt.v1_multiscale:
#                     mask = F.interpolate(
#                         mask, [self.opt.height, self.opt.width],
#                         mode="bilinear", align_corners=False)

#                 reprojection_losses *= mask

#                 # add a loss pushing mask to 1 (using nn.BCELoss for stability)
#                 weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
#                 loss += weighting_loss.mean()

#             if self.opt.avg_reprojection:
#                 reprojection_loss = reprojection_losses.mean(1, keepdim=True)
#             else:
#                 reprojection_loss = reprojection_losses

#             if not self.opt.disable_automasking:
#                 # add random numbers to break ties
#                 identity_reprojection_loss += torch.randn(
#                     identity_reprojection_loss.shape, device=self.device) * 0.00001

#                 combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
#             else:
#                 combined = reprojection_loss

#             if combined.shape[1] == 1:
#                 to_optimise = combined
#             else:
#                 to_optimise, idxs = torch.min(combined, dim=1)

#             if not self.opt.disable_automasking:
#                 outputs["identity_selection/{}".format(scale)] = (
#                     idxs > identity_reprojection_loss.shape[1] - 1).float()

#             loss += to_optimise.mean()

#             mean_disp = disp.mean(2, True).mean(3, True)
#             norm_disp = disp / (mean_disp + 1e-7)
#             smooth_loss = get_smooth_loss(norm_disp, color)

#             loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
#             total_loss += loss
#             losses["loss/{}".format(scale)] = loss

#         total_loss /= self.num_scales
#         losses["loss"] = total_loss
#         return losses

#     def compute_depth_losses(self, inputs, outputs, losses):
#         """Compute depth metrics, to allow monitoring during training

#         This isn't particularly accurate as it averages over the entire batch,
#         so is only used to give an indication of validation performance
#         """
#         depth_pred = outputs[("depth", 0, 0)]
#         depth_pred = torch.clamp(F.interpolate(
#             depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
#         depth_pred = depth_pred.detach()

#         depth_gt = inputs["depth_gt"]
#         mask = depth_gt > 0

#         # garg/eigen crop
#         crop_mask = torch.zeros_like(mask)
#         crop_mask[:, :, 153:371, 44:1197] = 1
#         mask = mask * crop_mask

#         depth_gt = depth_gt[mask]
#         depth_pred = depth_pred[mask]
#         depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

#         depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

#         depth_errors = compute_depth_errors(depth_gt, depth_pred)

#         for i, metric in enumerate(self.depth_metric_names):
#             losses[metric] = np.array(depth_errors[i].cpu())

#     def log_time(self, batch_idx, duration, loss):
#         """Print a logging statement to the terminal
#         """
#         samples_per_sec = self.opt.batch_size / duration
#         time_sofar = time.time() - self.start_time
#         training_time_left = (
#             self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
#         print_string = "epoch {:>3} | lr {:.6f} |lr_p {:.6f} | batch {:>6} | examples/s: {:5.1f}" + \
#             " | loss: {:.5f} | time elapsed: {} | time left: {}"
#         print(print_string.format(self.epoch, self.model_optimizer.state_dict()['param_groups'][0]['lr'],
#                                   self.model_pose_optimizer.state_dict()['param_groups'][0]['lr'],
#                                   batch_idx, samples_per_sec, loss,
#                                   sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

#     def log(self, mode, inputs, outputs, losses):
#         """Write an event to the tensorboard events file
#         """
#         writer = self.writers[mode]
#         for l, v in losses.items():
#             writer.add_scalar("{}".format(l), v, self.step)

#         for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
#             for s in self.opt.scales:
#                 for frame_id in self.opt.frame_ids:
#                     writer.add_image(
#                         "color_{}_{}/{}".format(frame_id, s, j),
#                         inputs[("color", frame_id, s)][j].data, self.step)
#                     if s == 0 and frame_id != 0:
#                         writer.add_image(
#                             "color_pred_{}_{}/{}".format(frame_id, s, j),
#                             outputs[("color", frame_id, s)][j].data, self.step)

#                 writer.add_image(
#                     "disp_{}/{}".format(s, j),
#                     normalize_image(outputs[("disp", s)][j]), self.step)

#                 if self.opt.predictive_mask:
#                     for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
#                         writer.add_image(
#                             "predictive_mask_{}_{}/{}".format(frame_id, s, j),
#                             outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
#                             self.step)

#                 elif not self.opt.disable_automasking:
#                     writer.add_image(
#                         "automask_{}/{}".format(s, j),
#                         outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

#     def save_opts(self):
#         """Save options to disk so we know what we ran this experiment with
#         """
#         models_dir = os.path.join(self.log_path, "models")
#         if not os.path.exists(models_dir):
#             os.makedirs(models_dir)
#         to_save = self.opt.__dict__.copy()

#         with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
#             json.dump(to_save, f, indent=2)

#     def save_model(self):
#         """Save model weights to disk
#         """
#         save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
#         if not os.path.exists(save_folder):
#             os.makedirs(save_folder)

#         for model_name, model in self.models.items():
#             save_path = os.path.join(save_folder, "{}.pth".format(model_name))
#             to_save = model.state_dict()
#             if model_name == 'encoder':
#                 # ===== EMA：先换权重，再保存，再恢复 =====
#                 self.ema.apply_shadow()
#                 to_save = model.state_dict()
#                 torch.save(to_save, os.path.join(save_folder, "encoder_ema.pth"))
#                 self.ema.restore()
#                 # ===== 原逻辑继续 =====
#                 # save the sizes - these are needed at prediction time
#                 to_save['height'] = self.opt.height
#                 to_save['width'] = self.opt.width
#                 to_save['use_stereo'] = self.opt.use_stereo
#             torch.save(to_save, save_path)

#         for model_name, model in self.models_pose.items():
#             save_path = os.path.join(save_folder, "{}.pth".format(model_name))
#             to_save = model.state_dict()
#             torch.save(to_save, save_path)

#         save_path = os.path.join(save_folder, "{}.pth".format("adam"))
#         torch.save(self.model_optimizer.state_dict(), save_path)

#         save_path = os.path.join(save_folder, "{}.pth".format("adam_pose"))
#         if self.use_pose_net:
#             torch.save(self.model_pose_optimizer.state_dict(), save_path)

#     def load_pretrain(self):
#         self.opt.mypretrain = os.path.expanduser(self.opt.mypretrain)
#         path = self.opt.mypretrain
#         model_dict = self.models["encoder"].state_dict()
#         pretrained_dict = torch.load(path)['model']
#         pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and not k.startswith('norm'))}
#         model_dict.update(pretrained_dict)
#         self.models["encoder"].load_state_dict(model_dict)
#         print('mypretrain loaded.')

#     def load_model(self):
#         """Load model(s) from disk
#         """
#         self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

#         assert os.path.isdir(self.opt.load_weights_folder), \
#             "Cannot find folder {}".format(self.opt.load_weights_folder)
#         print("loading model from folder {}".format(self.opt.load_weights_folder))

#         for n in self.opt.models_to_load:
#             print("Loading {} weights...".format(n))
#             path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))

#             if n in ['pose_encoder', 'pose']:
#                 model_dict = self.models_pose[n].state_dict()
#                 pretrained_dict = torch.load(path)
#                 pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#                 model_dict.update(pretrained_dict)
#                 self.models_pose[n].load_state_dict(model_dict)
#             else:
#                 model_dict = self.models[n].state_dict()
#                 pretrained_dict = torch.load(path)
#                 pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#                 model_dict.update(pretrained_dict)
#                 self.models[n].load_state_dict(model_dict)
#                 # 若存在 EMA 权重，加载它（eval 时用）
#                 ema_path = os.path.join(self.opt.load_weights_folder, "encoder_ema.pth")
#                 if n == "encoder" and os.path.exists(ema_path):
#                     self.ema.apply_shadow()
#                     self.models["encoder"].load_state_dict(torch.load(ema_path))
#                     self.ema.restore()
#         # loading adam state

#         optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
#         optimizer_pose_load_path = os.path.join(self.opt.load_weights_folder, "adam_pose.pth")
#         if os.path.isfile(optimizer_load_path):
#             print("Loading Adam weights")
#             optimizer_dict = torch.load(optimizer_load_path)
#             optimizer_pose_dict = torch.load(optimizer_pose_load_path)
#             self.model_optimizer.load_state_dict(optimizer_dict)
#             self.model_pose_optimizer.load_state_dict(optimizer_pose_dict)
#         else:
#             print("Cannot find Adam weights so Adam is randomly initialized")

# from __future__ import absolute_import, division, print_function

# # 基于原版。v1.4.2.4
# import time
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter

# import json
# from copy import deepcopy
# from utils import *
# from kitti_utils import *
# from layers import *

# import datasets
# import networks
# from linear_warmup_cosine_annealing_warm_restarts_weight_decay import ChainedScheduler


# # torch.backends.cudnn.benchmark = True


# def time_sync():
#     # PyTorch-accurate time
#     if torch.cuda.is_available():
#         torch.cuda.synchronize()
#     return time.time()


# class Trainer:
#     def __init__(self, options):
#         self.opt = options
#         self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

#         # checking height and width are multiples of 32
#         assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
#         assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

#         self.models = {}
#         self.models_pose = {}
#         self.parameters_to_train = []
#         self.parameters_to_train_pose = []

#         self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
#         self.profile = self.opt.profile

#         self.num_scales = len(self.opt.scales)
#         self.frame_ids = len(self.opt.frame_ids)
#         self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

#         assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

#         self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

#         if self.opt.use_stereo:
#             self.opt.frame_ids.append("s")

#         self.models["encoder"] = networks.LiteMono(model=self.opt.model,
#                                                    drop_path_rate=self.opt.drop_path,
#                                                    width=self.opt.width, height=self.opt.height)

#         self.models["encoder"].to(self.device)
#         self.parameters_to_train += list(self.models["encoder"].parameters())

#         self.models["depth"] = networks.DepthDecoder(self.models["encoder"].num_ch_enc,
#                                                      self.opt.scales)
#         self.models["depth"].to(self.device)
#         self.parameters_to_train += list(self.models["depth"].parameters())

#         if self.use_pose_net:
#             if self.opt.pose_model_type == "separate_resnet":
#                 self.models_pose["pose_encoder"] = networks.ResnetEncoder(
#                     self.opt.num_layers,
#                     self.opt.weights_init == "pretrained",
#                     num_input_images=self.num_pose_frames)

#                 self.models_pose["pose_encoder"].to(self.device)
#                 self.parameters_to_train_pose += list(self.models_pose["pose_encoder"].parameters())

#                 self.models_pose["pose"] = networks.PoseDecoder(
#                     self.models_pose["pose_encoder"].num_ch_enc,
#                     num_input_features=1,
#                     num_frames_to_predict_for=2)

#             elif self.opt.pose_model_type == "shared":
#                 self.models_pose["pose"] = networks.PoseDecoder(
#                     self.models["encoder"].num_ch_enc, self.num_pose_frames)

#             elif self.opt.pose_model_type == "posecnn":
#                 self.models_pose["pose"] = networks.PoseCNN(
#                     self.num_input_frames if self.opt.pose_model_input == "all" else 2)

#             self.models_pose["pose"].to(self.device)
#             self.parameters_to_train_pose += list(self.models_pose["pose"].parameters())

#         if self.opt.predictive_mask:
#             assert self.opt.disable_automasking, \
#                 "When using predictive_mask, please disable automasking with --disable_automasking"

#             # Our implementation of the predictive masking baseline has the the same architecture
#             # as our depth decoder. We predict a separate mask for each source frame.
#             self.models["predictive_mask"] = networks.DepthDecoder(
#                 self.models["encoder"].num_ch_enc, self.opt.scales,
#                 num_output_channels=(len(self.opt.frame_ids) - 1))
#             self.models["predictive_mask"].to(self.device)
#             self.parameters_to_train += list(self.models["predictive_mask"].parameters())

#         self.model_optimizer = optim.AdamW(self.parameters_to_train, self.opt.lr[0], weight_decay=self.opt.weight_decay)
#         if self.use_pose_net:
#             self.model_pose_optimizer = optim.AdamW(self.parameters_to_train_pose, self.opt.lr[3], weight_decay=self.opt.weight_decay)

#         self.model_lr_scheduler = ChainedScheduler(
#                             self.model_optimizer,
#                             T_0=int(self.opt.lr[2]),
#                             T_mul=1,
#                             eta_min=self.opt.lr[1],
#                             last_epoch=-1,
#                             max_lr=self.opt.lr[0],
#                             warmup_steps=0,
#                             gamma=0.9
#                         )
#         self.model_pose_lr_scheduler = ChainedScheduler(
#             self.model_pose_optimizer,
#             T_0=int(self.opt.lr[5]),
#             T_mul=1,
#             eta_min=self.opt.lr[4],
#             last_epoch=-1,
#             max_lr=self.opt.lr[3],
#             warmup_steps=0,
#             gamma=0.9
#         )

#         if self.opt.load_weights_folder is not None:
#             self.load_model()

#         if self.opt.mypretrain is not None:
#             self.load_pretrain()

#         print("Training model named:\n  ", self.opt.model_name)
#         print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
#         print("Training is using:\n  ", self.device)

#         # data
#         datasets_dict = {"kitti": datasets.KITTIRAWDataset,
#                          "kitti_odom": datasets.KITTIOdomDataset}
#         self.dataset = datasets_dict[self.opt.dataset]

#         fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

#         train_filenames = readlines(fpath.format("train"))
#         val_filenames = readlines(fpath.format("val"))
#         img_ext = '.png'  

#         num_train_samples = len(train_filenames)
#         self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

#         train_dataset = self.dataset(
#             self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
#             self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
#         self.train_loader = DataLoader(
#             train_dataset, self.opt.batch_size, True,
#             num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
#         val_dataset = self.dataset(
#             self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
#             self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
#         self.val_loader = DataLoader(
#             val_dataset, self.opt.batch_size, True,
#             num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
#         self.val_iter = iter(self.val_loader)

#         self.writers = {}
#         for mode in ["train", "val"]:
#             self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

#         if not self.opt.no_ssim:
#             self.ssim = SSIM()
#             self.ssim.to(self.device)

#         self.backproject_depth = {}
#         self.project_3d = {}
#         for scale in self.opt.scales:
#             h = self.opt.height // (2 ** scale)
#             w = self.opt.width // (2 ** scale)

#             self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
#             self.backproject_depth[scale].to(self.device)

#             self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
#             self.project_3d[scale].to(self.device)

#         self.depth_metric_names = [
#             "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

#         print("Using split:\n  ", self.opt.split)
#         print("There are {:d} training items and {:d} validation items\n".format(
#             len(train_dataset), len(val_dataset)))

#         # ===== EMA 初始化 =====
#         self.ema_decay = 0.9995
#         self.ema_shadow = {}
#         for name in ('encoder', 'depth'):
#             self.ema_shadow[name] = deepcopy(self.models[name].state_dict())

#         self.save_opts()

#     def set_train(self):
#         """Convert all models to training mode
#         """
#         for m in self.models.values():
#             m.train()
#         for name in ('encoder', 'depth'):
#             for m in self.models[name].modules():
#                 if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
#                     m.track_running_stats = True  # 恢复

#     def set_eval(self):
#         """Convert all models to testing/evaluation mode
#         """
#         for m in self.models.values():
#             m.eval()
#         # 额外加：冻结BN统计量
#         for mm in m.modules():
#             if isinstance(mm, (nn.BatchNorm2d, nn.SyncBatchNorm)):
#                 mm.track_running_stats = False

#     def train(self):
#         """Run the entire training pipeline
#         """
#         self.epoch = 0
#         self.step = 0
#         self.start_time = time.time()
#         for self.epoch in range(self.opt.num_epochs):
#             self.run_epoch()
#             if (self.epoch + 1) % self.opt.save_frequency == 0:
#                 self.save_model()

#     def run_epoch(self):
#         """Run a single epoch of training and validation
#         """

#         print("Training")
#         self.set_train()

#         self.model_lr_scheduler.step()
#         if self.use_pose_net:
#             self.model_pose_lr_scheduler.step()

#         for batch_idx, inputs in enumerate(self.train_loader):

#             before_op_time = time.time()

#             outputs, losses = self.process_batch(inputs)

#             self.model_optimizer.zero_grad()
#             if self.use_pose_net:
#                 self.model_pose_optimizer.zero_grad()
#             losses["loss"].backward()
#             self.model_optimizer.step()
#             if self.use_pose_net:
#                 self.model_pose_optimizer.step()

#             # ===== EMA 更新 =====
#             self._ema_update()

#             duration = time.time() - before_op_time

#             # log less frequently after the first 2000 steps to save time & disk space
#             early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 20000
#             late_phase = self.step % 2000 == 0

#             if early_phase or late_phase:
#                 self.log_time(batch_idx, duration, losses["loss"].cpu().data)

#                 if "depth_gt" in inputs:
#                     self.compute_depth_losses(inputs, outputs, losses)

#                 self.log("train", inputs, outputs, losses)
#                 self.val()

#             self.step += 1

#     def process_batch(self, inputs):
#         """Pass a minibatch through the network and generate images and losses
#         """
#         for key, ipt in inputs.items():
#             inputs[key] = ipt.to(self.device)

#         if self.opt.pose_model_type == "shared":
#             # If we are using a shared encoder for both depth and pose (as advocated
#             # in monodepthv1), then all images are fed separately through the depth encoder.
#             all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
#             all_features = self.models["encoder"](all_color_aug)
#             all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

#             features = {}
#             for i, k in enumerate(self.opt.frame_ids):
#                 features[k] = [f[i] for f in all_features]

#             outputs = self.models["depth"](features[0])
#         else:
#             # Otherwise, we only feed the image with frame_id 0 through the depth encoder

#             features = self.models["encoder"](inputs["color_aug", 0, 0])

#             outputs = self.models["depth"](features)

#         if self.opt.predictive_mask:
#             outputs["predictive_mask"] = self.models["predictive_mask"](features)

#         if self.use_pose_net:
#             outputs.update(self.predict_poses(inputs, features))

#         self.generate_images_pred(inputs, outputs)
#         losses = self.compute_losses(inputs, outputs)

#         return outputs, losses

#     def predict_poses(self, inputs, features):
#         """Predict poses between input frames for monocular sequences.
#         """
#         outputs = {}
#         if self.num_pose_frames == 2:
#             # In this setting, we compute the pose to each source frame via a
#             # separate forward pass through the pose network.

#             # select what features the pose network takes as input
#             if self.opt.pose_model_type == "shared":
#                 pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
#             else:
#                 pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

#             for f_i in self.opt.frame_ids[1:]:
#                 if f_i != "s":
#                     # To maintain ordering we always pass frames in temporal order
#                     if f_i < 0:
#                         pose_inputs = [pose_feats[f_i], pose_feats[0]]
#                     else:
#                         pose_inputs = [pose_feats[0], pose_feats[f_i]]

#                     if self.opt.pose_model_type == "separate_resnet":
#                         pose_inputs = [self.models_pose["pose_encoder"](torch.cat(pose_inputs, 1))]
#                     elif self.opt.pose_model_type == "posecnn":
#                         pose_inputs = torch.cat(pose_inputs, 1)

#                     axisangle, translation = self.models_pose["pose"](pose_inputs)
#                     outputs[("axisangle", 0, f_i)] = axisangle
#                     outputs[("translation", 0, f_i)] = translation

#                     # Invert the matrix if the frame id is negative
#                     outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
#                         axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

#         else:
#             # Here we input all frames to the pose net (and predict all poses) together
#             if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
#                 pose_inputs = torch.cat(
#                     [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

#                 if self.opt.pose_model_type == "separate_resnet":
#                     pose_inputs = [self.models["pose_encoder"](pose_inputs)]

#             elif self.opt.pose_model_type == "shared":
#                 pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

#             axisangle, translation = self.models_pose["pose"](pose_inputs)

#             for i, f_i in enumerate(self.opt.frame_ids[1:]):
#                 if f_i != "s":
#                     outputs[("axisangle", 0, f_i)] = axisangle
#                     outputs[("translation", 0, f_i)] = translation
#                     outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
#                         axisangle[:, i], translation[:, i])

#         return outputs

#     def val(self):
#         """Validate the model on a single minibatch
#         """
#         self.set_eval()

#         self._swap_ema_shadow(swap_back=False)   # 验证前换 ema

#         try:
#             inputs = self.val_iter.__next__()  
#         except StopIteration:
#             self.val_iter = iter(self.val_loader)
#             inputs = self.val_iter.__next__() 

#         with torch.no_grad():
#             outputs, losses = self.process_batch(inputs)

#             if "depth_gt" in inputs:
#                 self.compute_depth_losses(inputs, outputs, losses)

#             self.log("val", inputs, outputs, losses)

#         self._swap_ema_shadow(swap_back=True)    # 验证完换回来
#         self.set_train()

#     def generate_images_pred(self, inputs, outputs):
#         """Generate the warped (reprojected) color images for a minibatch.
#         Generated images are saved into the `outputs` dictionary.
#         """
#         for scale in self.opt.scales:
#             disp = outputs[("disp", scale)]
#             if self.opt.v1_multiscale:
#                 source_scale = scale
#             else:
#                 disp = F.interpolate(
#                     disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
#                 source_scale = 0

#             _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

#             outputs[("depth", 0, scale)] = depth

#             for i, frame_id in enumerate(self.opt.frame_ids[1:]):

#                 if frame_id == "s":
#                     T = inputs["stereo_T"]
#                 else:
#                     T = outputs[("cam_T_cam", 0, frame_id)]

#                 # from the authors of https://arxiv.org/abs/1712.00175
#                 if self.opt.pose_model_type == "posecnn":

#                     axisangle = outputs[("axisangle", 0, frame_id)]
#                     translation = outputs[("translation", 0, frame_id)]

#                     inv_depth = 1 / depth
#                     mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

#                     T = transformation_from_parameters(
#                         axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

#                 cam_points = self.backproject_depth[source_scale](
#                     depth, inputs[("inv_K", source_scale)])
#                 pix_coords = self.project_3d[source_scale](
#                     cam_points, inputs[("K", source_scale)], T)

#                 outputs[("sample", frame_id, scale)] = pix_coords

#                 outputs[("color", frame_id, scale)] = F.grid_sample(
#                     inputs[("color", frame_id, source_scale)],
#                     outputs[("sample", frame_id, scale)],
#                     padding_mode="border", align_corners=True)

#                 if not self.opt.disable_automasking:
#                     outputs[("color_identity", frame_id, scale)] = \
#                         inputs[("color", frame_id, source_scale)]

#     def compute_reprojection_loss(self, pred, target):
#         """Computes reprojection loss between a batch of predicted and target images
#         """
#         abs_diff = torch.abs(target - pred)
#         l1_loss = abs_diff.mean(1, True)

#         if self.opt.no_ssim:
#             reprojection_loss = l1_loss
#         else:
#             ssim_loss = self.ssim(pred, target).mean(1, True)
#             reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

#         return reprojection_loss

#     def compute_losses(self, inputs, outputs):
#         """Compute the reprojection and smoothness losses for a minibatch
#         """

#         losses = {}
#         total_loss = 0

#         for scale in self.opt.scales:
#             loss = 0
#             reprojection_losses = []

#             if self.opt.v1_multiscale:
#                 source_scale = scale
#             else:
#                 source_scale = 0

#             disp = outputs[("disp", scale)]
#             color = inputs[("color", 0, scale)]
#             target = inputs[("color", 0, source_scale)]

#             for frame_id in self.opt.frame_ids[1:]:
#                 pred = outputs[("color", frame_id, scale)]
#                 reprojection_losses.append(self.compute_reprojection_loss(pred, target))

#             reprojection_losses = torch.cat(reprojection_losses, 1)

#             if not self.opt.disable_automasking:
#                 identity_reprojection_losses = []
#                 for frame_id in self.opt.frame_ids[1:]:
#                     pred = inputs[("color", frame_id, source_scale)]
#                     identity_reprojection_losses.append(
#                         self.compute_reprojection_loss(pred, target))

#                 identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

#                 if self.opt.avg_reprojection:
#                     identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
#                 else:
#                     # save both images, and do min all at once below
#                     identity_reprojection_loss = identity_reprojection_losses

#             elif self.opt.predictive_mask:
#                 # use the predicted mask
#                 mask = outputs["predictive_mask"]["disp", scale]
#                 if not self.opt.v1_multiscale:
#                     mask = F.interpolate(
#                         mask, [self.opt.height, self.opt.width],
#                         mode="bilinear", align_corners=False)

#                 reprojection_losses *= mask

#                 # add a loss pushing mask to 1 (using nn.BCELoss for stability)
#                 weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
#                 loss += weighting_loss.mean()

#             if self.opt.avg_reprojection:
#                 reprojection_loss = reprojection_losses.mean(1, keepdim=True)
#             else:
#                 reprojection_loss = reprojection_losses

#             if not self.opt.disable_automasking:
#                 # add random numbers to break ties
#                 identity_reprojection_loss += torch.randn(
#                     identity_reprojection_loss.shape, device=self.device) * 0.00001

#                 combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
#             else:
#                 combined = reprojection_loss

#             if combined.shape[1] == 1:
#                 to_optimise = combined
#             else:
#                 to_optimise, idxs = torch.min(combined, dim=1)

#             if not self.opt.disable_automasking:
#                 outputs["identity_selection/{}".format(scale)] = (
#                     idxs > identity_reprojection_loss.shape[1] - 1).float()

#             loss += to_optimise.mean()

#             mean_disp = disp.mean(2, True).mean(3, True)
#             norm_disp = disp / (mean_disp + 1e-7)
#             smooth_loss = get_smooth_loss(norm_disp, color)

#             loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
#             total_loss += loss
#             losses["loss/{}".format(scale)] = loss

#         total_loss /= self.num_scales
#         losses["loss"] = total_loss
#         return losses

#     def compute_depth_losses(self, inputs, outputs, losses):
#         """Compute depth metrics, to allow monitoring during training

#         This isn't particularly accurate as it averages over the entire batch,
#         so is only used to give an indication of validation performance
#         """
#         depth_pred = outputs[("depth", 0, 0)]
#         depth_pred = torch.clamp(F.interpolate(
#             depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
#         depth_pred = depth_pred.detach()

#         depth_gt = inputs["depth_gt"]
#         mask = depth_gt > 0

#         # garg/eigen crop
#         crop_mask = torch.zeros_like(mask)
#         crop_mask[:, :, 153:371, 44:1197] = 1
#         mask = mask * crop_mask

#         depth_gt = depth_gt[mask]
#         depth_pred = depth_pred[mask]
#         depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

#         depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

#         depth_errors = compute_depth_errors(depth_gt, depth_pred)

#         for i, metric in enumerate(self.depth_metric_names):
#             losses[metric] = np.array(depth_errors[i].cpu())

#     def log_time(self, batch_idx, duration, loss):
#         """Print a logging statement to the terminal
#         """
#         samples_per_sec = self.opt.batch_size / duration
#         time_sofar = time.time() - self.start_time
#         training_time_left = (
#             self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
#         print_string = "epoch {:>3} | lr {:.6f} |lr_p {:.6f} | batch {:>6} | examples/s: {:5.1f}" + \
#             " | loss: {:.5f} | time elapsed: {} | time left: {}"
#         print(print_string.format(self.epoch, self.model_optimizer.state_dict()['param_groups'][0]['lr'],
#                                   self.model_pose_optimizer.state_dict()['param_groups'][0]['lr'],
#                                   batch_idx, samples_per_sec, loss,
#                                   sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

#     def log(self, mode, inputs, outputs, losses):
#         """Write an event to the tensorboard events file
#         """
#         writer = self.writers[mode]
#         for l, v in losses.items():
#             writer.add_scalar("{}".format(l), v, self.step)

#         for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
#             for s in self.opt.scales:
#                 for frame_id in self.opt.frame_ids:
#                     writer.add_image(
#                         "color_{}_{}/{}".format(frame_id, s, j),
#                         inputs[("color", frame_id, s)][j].data, self.step)
#                     if s == 0 and frame_id != 0:
#                         writer.add_image(
#                             "color_pred_{}_{}/{}".format(frame_id, s, j),
#                             outputs[("color", frame_id, s)][j].data, self.step)

#                 writer.add_image(
#                     "disp_{}/{}".format(s, j),
#                     normalize_image(outputs[("disp", s)][j]), self.step)

#                 if self.opt.predictive_mask:
#                     for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
#                         writer.add_image(
#                             "predictive_mask_{}_{}/{}".format(frame_id, s, j),
#                             outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
#                             self.step)

#                 elif not self.opt.disable_automasking:
#                     writer.add_image(
#                         "automask_{}/{}".format(s, j),
#                         outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

#     def save_opts(self):
#         """Save options to disk so we know what we ran this experiment with
#         """
#         models_dir = os.path.join(self.log_path, "models")
#         if not os.path.exists(models_dir):
#             os.makedirs(models_dir)
#         to_save = self.opt.__dict__.copy()

#         with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
#             json.dump(to_save, f, indent=2)

#     def _ema_update(self):
#         if self.step < 2000:          # 前2k步不做EMA
#             return
#         with torch.no_grad():
#             for name in ('encoder', 'depth'):
#                 online_m = self.models[name]
#                 shadow_sd = self.ema_shadow[name]
#                 for (k, v_online) in online_m.state_dict().items():
#                     if v_online.dtype not in (torch.float32, torch.float16):
#                         shadow_sd[k].copy_(v_online)          # buffer直接拷贝
#                     elif 'num_batches_tracked' in k:          # BN计数器也直接拷贝
#                         shadow_sd[k].copy_(v_online)
#                     else:
#                         shadow_sd[k].lerp_(v_online, 1. - self.ema_decay)
                            
#     def _swap_ema_shadow(self, swap_back=False):
#         """False: shadow→online；True: 换回来"""
#         for name in ('encoder', 'depth'):
#             online_sd = self.models[name].state_dict()
#             shadow_sd = self.ema_shadow[name]
#             if not swap_back:
#                 tmp = deepcopy(online_sd)
#                 online_sd.update(shadow_sd)
#                 shadow_sd.update(tmp)
#             else:
#                 online_sd.update(shadow_sd)
#                 shadow_sd.update(online_sd)

#     def save_model(self):
#         """Save model weights to disk
#         """
#         save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
#         if not os.path.exists(save_folder):
#             os.makedirs(save_folder)

#         for model_name, model in self.models.items():
#             save_path = os.path.join(save_folder, "{}.pth".format(model_name))
#             to_save = model.state_dict()
#             if model_name == 'encoder':
#                 # save the sizes - these are needed at prediction time
#                 to_save['height'] = self.opt.height
#                 to_save['width'] = self.opt.width
#                 to_save['use_stereo'] = self.opt.use_stereo
#             torch.save(to_save, save_path)

#         for model_name, model in self.models_pose.items():
#             save_path = os.path.join(save_folder, "{}.pth".format(model_name))
#             to_save = model.state_dict()
#             torch.save(to_save, save_path)

#         save_path = os.path.join(save_folder, "{}.pth".format("adam"))
#         torch.save(self.model_optimizer.state_dict(), save_path)

#         save_path = os.path.join(save_folder, "{}.pth".format("adam_pose"))
#         if self.use_pose_net:
#             torch.save(self.model_pose_optimizer.state_dict(), save_path)

#     def load_pretrain(self):
#         self.opt.mypretrain = os.path.expanduser(self.opt.mypretrain)
#         path = self.opt.mypretrain
#         model_dict = self.models["encoder"].state_dict()
#         pretrained_dict = torch.load(path)['model']
#         pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and not k.startswith('norm'))}
#         model_dict.update(pretrained_dict)
#         self.models["encoder"].load_state_dict(model_dict)
#         print('mypretrain loaded.')

#     def load_model(self):
#         """Load model(s) from disk
#         """
#         self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

#         assert os.path.isdir(self.opt.load_weights_folder), \
#             "Cannot find folder {}".format(self.opt.load_weights_folder)
#         print("loading model from folder {}".format(self.opt.load_weights_folder))

#         for n in self.opt.models_to_load:
#             print("Loading {} weights...".format(n))
#             path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))

#             if n in ['pose_encoder', 'pose']:
#                 model_dict = self.models_pose[n].state_dict()
#                 pretrained_dict = torch.load(path)
#                 pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#                 model_dict.update(pretrained_dict)
#                 self.models_pose[n].load_state_dict(model_dict)
#             else:
#                 model_dict = self.models[n].state_dict()
#                 pretrained_dict = torch.load(path)
#                 pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#                 model_dict.update(pretrained_dict)
#                 self.models[n].load_state_dict(model_dict)

#         # loading adam state

#         optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
#         optimizer_pose_load_path = os.path.join(self.opt.load_weights_folder, "adam_pose.pth")
#         if os.path.isfile(optimizer_load_path):
#             print("Loading Adam weights")
#             optimizer_dict = torch.load(optimizer_load_path)
#             optimizer_pose_dict = torch.load(optimizer_pose_load_path)
#             self.model_optimizer.load_state_dict(optimizer_dict)
#             self.model_pose_optimizer.load_state_dict(optimizer_pose_dict)
#         else:
#             print("Cannot find Adam weights so Adam is randomly initialized")

# # v1.4.2.5 == v1.4.2.4 +TTA ,只修改val函数
# v3.0 = v1.4+扩散模型

"""
完整修正版 trainer.py - 集成扩散深度细化
所有方法完整，使用option.py参数控制
"""
from __future__ import absolute_import, division, print_function

import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import json

from utils import *
from kitti_utils import *
from layers import *
import datasets
import networks
from linear_warmup_cosine_annealing_warm_restarts_weight_decay import ChainedScheduler

# ===== 新增：导入扩散模块 =====
from diffusion_depth_module import DiffusionDepthRefiner


def time_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.models_pose = {}
        self.parameters_to_train = []
        self.parameters_to_train_pose = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
        self.profile = self.opt.profile

        self.num_scales = len(self.opt.scales)
        self.frame_ids = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        # ===== 原有encoder和depth模型 =====
        self.models["encoder"] = networks.LiteMono(model=self.opt.model,
                                                   drop_path_rate=self.opt.drop_path,
                                                   width=self.opt.width, height=self.opt.height)
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = networks.DepthDecoder(self.models["encoder"].num_ch_enc,
                                                     self.opt.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        # ===== 新增：扩散模块（根据option.py参数） =====
        if self.opt.use_diffusion:
            print("=" * 60)
            print("Initializing Diffusion Refiner...")
            print(f"  - Feature channels: {self.opt.diffusion_channels}")
            print(f"  - Inference steps: {self.opt.diffusion_steps}")
            print(f"  - Loss weight: {self.opt.diffusion_loss_weight}")
            print("=" * 60)
            
            self.models["diffusion"] = DiffusionDepthRefiner(
                depth_channels=1,
                feature_channels=self.opt.diffusion_channels,
                time_emb_dim=128,
                num_steps=self.opt.diffusion_steps
            )
            self.models["diffusion"].to(self.device)
            self.parameters_to_train += list(self.models["diffusion"].parameters())

        # ===== 原有pose网络 =====
        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models_pose["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)

                self.models_pose["pose_encoder"].to(self.device)
                self.parameters_to_train_pose += list(self.models_pose["pose_encoder"].parameters())

                self.models_pose["pose"] = networks.PoseDecoder(
                    self.models_pose["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "shared":
                self.models_pose["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)

            elif self.opt.pose_model_type == "posecnn":
                self.models_pose["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            self.models_pose["pose"].to(self.device)
            self.parameters_to_train_pose += list(self.models_pose["pose"].parameters())

        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())

        # ===== 优化器 =====
        self.model_optimizer = optim.AdamW(self.parameters_to_train, self.opt.lr[0], 
                                           weight_decay=self.opt.weight_decay)
        if self.use_pose_net:
            self.model_pose_optimizer = optim.AdamW(self.parameters_to_train_pose, self.opt.lr[3], 
                                                    weight_decay=self.opt.weight_decay)

        self.model_lr_scheduler = ChainedScheduler(
            self.model_optimizer,
            T_0=int(self.opt.lr[2]),
            T_mul=1,
            eta_min=self.opt.lr[1],
            last_epoch=-1,
            max_lr=self.opt.lr[0],
            warmup_steps=0,
            gamma=0.9
        )
        
        if self.use_pose_net:
            self.model_pose_lr_scheduler = ChainedScheduler(
                self.model_pose_optimizer,
                T_0=int(self.opt.lr[5]),
                T_mul=1,
                eta_min=self.opt.lr[4],
                last_epoch=-1,
                max_lr=self.opt.lr[3],
                warmup_steps=0,
                gamma=0.9
            )

        if self.opt.load_weights_folder is not None:
            self.load_model()

        if self.opt.mypretrain is not None:
            self.load_pretrain()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # ===== 数据加载 =====
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        for m in self.models.values():
            m.train()

    def set_eval(self):
        for m in self.models.values():
            m.eval()

    def train(self):
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        print("Training")
        self.set_train()

        self.model_lr_scheduler.step()
        if self.use_pose_net:
            self.model_pose_lr_scheduler.step()

        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            if self.use_pose_net:
                self.model_pose_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()
            if self.use_pose_net:
                self.model_pose_optimizer.step()

            duration = time.time() - before_op_time

            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 20000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1

    def process_batch(self, inputs):
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        # ===== 编码-解码 =====
        if self.opt.pose_model_type == "shared":
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]
            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]
            outputs = self.models["depth"](features[0])
        else:
            features = self.models["encoder"](inputs["color_aug", 0, 0])
            outputs = self.models["depth"](features)
        # ===== 关键修改：先完成光度损失计算 =====
        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))

        self.generate_images_pred(inputs, outputs)  # 用原始depth做重投影
        
        # ===== 扩散细化（epoch>=20且训练时） =====
        if self.opt.use_diffusion and self.epoch >= 20 and self.models["diffusion"].training:
            for scale in self.opt.scales:
                # 关键修改：只在训练时detach一次，避免重复计算
                depth_init = outputs[("depth", 0, scale)].detach()
                
                # 扩散细化
                depth_refined, depth_noisy, noise_gt, t = self.models["diffusion"](depth_init)
                
                # 存储用于损失计算
                outputs[("depth_refined", 0, scale)] = depth_refined
                outputs[("depth_init", 0, scale)] = depth_init  # 存储用于对比
                outputs[("noise_gt", scale)] = noise_gt  # 注意：改为noise_gt
                outputs[("noise_t", scale)] = t
        
        # ===== 测试时细化 =====
        elif self.opt.use_diffusion and not self.models["diffusion"].training:
            for scale in self.opt.scales:
                depth_init = outputs[("depth", 0, scale)]
                depth_refined = self.models["diffusion"](
                    depth_init, 
                    num_inference_steps=self.opt.diffusion_steps,
                    deterministic=True
                )
                outputs[("depth", 0, scale)] = depth_refined
        
        losses = self.compute_losses(inputs, outputs)
        return outputs, losses
        

    def predict_poses(self, inputs, features):
        outputs = {}
        if self.num_pose_frames == 2:
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models_pose["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models_pose["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
        else:
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)
                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]
            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models_pose["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    def val(self):
        self.set_eval()
        try:
            inputs = self.val_iter.__next__()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.__next__()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):
                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                if self.opt.pose_model_type == "posecnn":
                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]
                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)
                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords
                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True)

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)
                reprojection_losses *= mask
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape, device=self.device) * 0.00001
                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales

        # ===== 扩散损失（简化版，避免重投影计算） =====
        if self.opt.use_diffusion and self.epoch >= 20 and self.models["diffusion"].training:
            diffusion_loss = 0
            
            for scale in self.opt.scales:
                if ("depth_refined", 0, scale) in outputs:
                    depth_init = outputs[("depth_init", 0, scale)]
                    depth_refined = outputs[("depth_refined", 0, scale)]
                    noise_gt = outputs[("noise_gt", scale)]
                    color = inputs[("color", 0, scale)]
                    
                    # ① 去噪损失：让细化深度接近初始深度
                    denoise_loss = F.l1_loss(depth_refined, depth_init)
                    
                    # ② 边缘一致性损失
                    def compute_grad(img):
                        grad_x = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])
                        grad_y = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])
                        return grad_x.mean() + grad_y.mean()
                    
                    grad_loss = compute_grad(depth_refined) - compute_grad(depth_init).detach()
                    grad_loss = torch.abs(grad_loss)
                    
                    diffusion_loss += denoise_loss + 0.1 * grad_loss
            
            diffusion_loss = (diffusion_loss / self.num_scales) * 0.05  # 极低权重
            losses["diffusion_loss"] = diffusion_loss
            total_loss += diffusion_loss

        losses["loss"] = total_loss
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | lr {:.6f} |lr_p {:.6f} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, 
                                  self.model_optimizer.state_dict()['param_groups'][0]['lr'],
                                  self.model_pose_optimizer.state_dict()['param_groups'][0]['lr'],
                                  batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                elif not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        for model_name, model in self.models_pose.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam_pose"))
        if self.use_pose_net:
            torch.save(self.model_pose_optimizer.state_dict(), save_path)

    def load_pretrain(self):
        self.opt.mypretrain = os.path.expanduser(self.opt.mypretrain)
        path = self.opt.mypretrain
        model_dict = self.models["encoder"].state_dict()
        pretrained_dict = torch.load(path)['model']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and not k.startswith('norm'))}
        model_dict.update(pretrained_dict)
        self.models["encoder"].load_state_dict(model_dict)
        print('mypretrain loaded.')

    def load_model(self):
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))

            if n in ['pose_encoder', 'pose']:
                model_dict = self.models_pose[n].state_dict()
                pretrained_dict = torch.load(path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models_pose[n].load_state_dict(model_dict)
            else:
                model_dict = self.models[n].state_dict()
                pretrained_dict = torch.load(path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models[n].load_state_dict(model_dict)

        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        optimizer_pose_load_path = os.path.join(self.opt.load_weights_folder, "adam_pose.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            optimizer_pose_dict = torch.load(optimizer_pose_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
            self.model_pose_optimizer.load_state_dict(optimizer_pose_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")