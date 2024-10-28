import sys
import os
import torch
import pygame
import argparse

from utils import utils
from train import PanoModel, BundleDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Interactive Rendering")
    parser.add_argument("-d", "--checkpoint", required=True, help="Path to the checkpoint file")
    return parser.parse_args()

args = parse_args()
chkpt_path = args.checkpoint
chkpt_path = os.path.join(os.path.dirname(chkpt_path), "last.ckpt")
data_path = os.path.join(os.path.dirname(chkpt_path), "data.pkl")
model = PanoModel.load_from_checkpoint(chkpt_path, device="cuda", cached_data=data_path)
model = model.to("cuda")
model = model.eval()
model.args.no_offset = False
model.args.no_view_color = False

class InteractiveWindow:
    def __init__(self):
        self.offset = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32).to("cuda")
        self.fov_factor = 1.0
        self.width = 1440
        self.height = 1080
        self.t = 0.5
        self.brightness = 1
        self.t_tensor = torch.full((self.width * self.height,), self.t, device="cuda", dtype=torch.float32)

        self.generate_camera()
            

        pygame.init()
        pygame.font.init()  # Initialize font module
        self.font = pygame.font.SysFont(None, 36)  # Set up the font

        self.screen = pygame.display.set_mode((self.width, self.height), pygame.DOUBLEBUF | pygame.HWSURFACE | pygame.SRCALPHA)
        pygame.display.set_caption("NeuLS Interactive Rendering")

        self.last_mouse_position = None
        self.update_image()
    
    def generate_camera(self):
        self.intrinsics_inv = model.data.intrinsics_inv[int(self.t * model.args.num_frames - 1)].clone()
        self.intrinsics_inv[0] *= 1.77  # Widen FOV
        self.intrinsics_inv = self.intrinsics_inv[None, :, :].repeat(self.width * self.height, 1, 1).to("cuda")

        self.quaternion_camera_to_world = model.data.quaternion_camera_to_world[int(self.t * model.args.num_frames - 1)].to("cuda")
        self.quaternion_camera_to_world_offset = self.quaternion_camera_to_world.clone()
        self.camera_to_world = model.model_rotation(self.quaternion_camera_to_world, self.t_tensor).to("cuda")

        self.ray_origins = model.model_translation(self.t_tensor, 1.0)

        self.uv = utils.make_grid(self.height, self.width, [0, 1], [0, 1]).to("cuda")
        self.ray_directions = model.generate_ray_directions(self.uv, self.camera_to_world, self.intrinsics_inv)



    def generate_image(self):
        with torch.no_grad():
            self.camera_to_world = utils.convert_quaternions_to_rot(self.quaternion_camera_to_world_offset).repeat(self.width * self.height, 1, 1).to("cuda")
            self.ray_directions = model.generate_ray_directions(self.uv * self.fov_factor + 0.5 * (1 - self.fov_factor), self.camera_to_world, self.intrinsics_inv)
            
        rgb_transmission = model.inference(self.t_tensor, self.uv * self.fov_factor + 0.5 * (1 - self.fov_factor), self.ray_origins + self.offset, self.ray_directions, 1.0)

        rgb_transmission = model.color(rgb_transmission, self.height, self.width).permute(2, 1, 0)
        return (rgb_transmission ** 0.7 * 255 * self.brightness).clamp(0, 255).byte().cpu().numpy()

    def update_image(self):
        image = self.generate_image()
        surf = pygame.surfarray.make_surface(image)
        self.screen.blit(surf, (0, 0))

        # Render and display status text
        self.render_status_text()

        pygame.display.flip()

    def render_status_text(self):
        # Double the font size
        large_font = pygame.font.SysFont(None, int(self.font.get_height() * 1.4))
        
        view_color_status = "View-Dependent Color: On" if not model.args.no_view_color else "View-Dependent Color: Off"
        ray_offset_status = "Ray Offset: On" if not model.args.no_offset else "Ray Offset: Off"

        # Render the text in white
        view_color_text = large_font.render(view_color_status, True, (255, 255, 255))
        ray_offset_text = large_font.render(ray_offset_status, True, (255, 255, 255))

        # Calculate the size of the black box
        box_width = max(view_color_text.get_width(), ray_offset_text.get_width()) + 20
        box_height = view_color_text.get_height() + ray_offset_text.get_height() + 20

        # Position for the box
        box_x = 10
        box_y = self.height - box_height - 10

        # Draw the black box
        pygame.draw.rect(self.screen, (0, 0, 0), (box_x, box_y, box_width, box_height))

        # Draw the white text on top of the black box
        self.screen.blit(view_color_text, (box_x + 10, box_y + 10))
        self.screen.blit(ray_offset_text, (box_x + 10, box_y + view_color_text.get_height() + 10))


    def handle_keys(self, event):
        fov_step = 1.05

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_o:
                model.args.no_offset = not model.args.no_offset
            elif event.key == pygame.K_p:
                model.args.no_view_color = not model.args.no_view_color
            elif event.key == pygame.K_EQUALS:
                self.brightness *= 1.05
            elif event.key == pygame.K_MINUS:
                self.brightness *= 0.9523
            elif event.key == pygame.K_LEFT:
                self.t = max(0, self.t - 0.01)
                self.t_tensor.fill_(self.t)
                self.generate_camera()
            elif event.key == pygame.K_RIGHT:
                self.t = min(1, self.t + 0.01)
                self.t_tensor.fill_(self.t)
                self.generate_camera()
            elif event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()
            elif event.key == pygame.K_r:
                self.offset.zero_()
                self.quaternion_camera_to_world_offset = self.quaternion_camera_to_world.clone()
                self.fov_factor = 1.0

    def handle_mouse(self, event):
        shift_pressed = pygame.key.get_mods() & pygame.KMOD_SHIFT

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # Left button
            self.last_mouse_position = pygame.mouse.get_pos()
        elif event.type == pygame.MOUSEMOTION and self.last_mouse_position is not None:
            x, y = pygame.mouse.get_pos()
            dx, dy = x - self.last_mouse_position[0], y - self.last_mouse_position[1]
            sensitivity = 0.001 * self.fov_factor * model.args.focal_compensation

            if event.buttons[0]:  # Left button held down
                if shift_pressed:
                    step = 0.5
                    movement_camera_space = torch.tensor([-dx * step * sensitivity, dy * step * sensitivity, 0.0]).to("cuda")
                    movement_world_space = self.camera_to_world[0] @ movement_camera_space
                    self.offset[0] += movement_world_space
                else:
                    pitch = torch.tensor(-dy * sensitivity, dtype=torch.float32).to("cuda")
                    yaw = torch.tensor(-dx * sensitivity, dtype=torch.float32).to("cuda")

                    # Compute the pitch and yaw quaternions (small rotations)
                    pitch_quat = torch.tensor([torch.cos(pitch / 2), torch.sin(pitch / 2), 0, 0]).to("cuda")
                    yaw_quat = torch.tensor([torch.cos(yaw / 2), 0, torch.sin(yaw / 2), 0]).to("cuda")

                    # Combine the pitch and yaw quaternions by multiplying them
                    rotation_quat = utils.quaternion_multiply(yaw_quat, pitch_quat)

                    # Apply the rotation incrementally
                    self.quaternion_camera_to_world_offset = utils.quaternion_multiply(
                        self.quaternion_camera_to_world_offset, rotation_quat
                    )

                    # Normalize the quaternion to avoid drift
                    self.quaternion_camera_to_world_offset = self.quaternion_camera_to_world_offset / torch.norm(self.quaternion_camera_to_world_offset)


            self.last_mouse_position = (x, y)
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:  # Left button
            self.last_mouse_position = None
        elif event.type == pygame.MOUSEWHEEL:
            fov_step = 1.02
            if event.y > 0:  # Scroll up
                self.fov_factor /= fov_step
            elif event.y < 0:  # Scroll down
                self.fov_factor *= fov_step

    def run(self):
        clock = pygame.time.Clock()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type in [pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION, pygame.MOUSEWHEEL]:
                    self.handle_mouse(event)
                else:
                    self.handle_keys(event)

            self.update_image()
            clock.tick(30)  # Limit the frame rate to 30 FPS

if __name__ == "__main__":
    window = InteractiveWindow()
    window.run()
