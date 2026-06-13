import blenderproc as bproc
import bpy
import argparse
import os
import numpy as np
import cv2
from tqdm import tqdm

assets_path = 'assets'

bproc.init()

# load bop objects into the scene
target_objs = bproc.loader.load_blend(path=os.path.join(assets_path, 'probe/probe.blend'))

for i, obj in enumerate(target_objs):
    obj.set_cp("category_id", 1) 
    obj.set_cp("bop_dataset_name", "probe")
    obj.set_name(f"probe_{i}")


# load BOP datset intrinsics
import json

with open("camera.json", "r") as f:
    cam_data = json.load(f)

fx = cam_data["fx"]
fy = cam_data["fy"]
cx = cam_data["cx"]
cy = cam_data["cy"]
width = cam_data["width"]
height = cam_data["height"]

K = np.array([
    [fx, 0,  cx],
    [0,  fy, cy],
    [0,  0,  1]
])

bproc.camera.set_resolution(width, height)
bproc.camera.set_intrinsics_from_K_matrix(K, width, height)


# set shading and hide objects
for obj in target_objs:
    obj.set_shading_mode('auto')
    obj.hide(True)
    

def load_custom_pbr_materials(base_path):
    materials = []
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        if not os.path.isdir(folder_path):
            continue
            
        mat = bproc.material.create(f"custom_mat_{folder_name}")
        files = os.listdir(folder_path)
        
        color_file = next((f for f in files if 'color' in f.lower() or 'albedo' in f.lower()), None)
        normal_file = next((f for f in files if 'normal' in f.lower()), None)
        bump_file = next((f for f in files if 'bump' in f.lower() or 'roughness' in f.lower()), None)
        
        if color_file:
            color_image = bpy.data.images.load(os.path.join(folder_path, color_file))
            mat.set_principled_shader_value("Base Color", color_image)
            
        if normal_file:
            normal_image = bpy.data.images.load(os.path.join(folder_path, normal_file))
            mat.set_principled_shader_value("Normal", normal_image)
            
        if bump_file:
            bump_image = bpy.data.images.load(os.path.join(folder_path, bump_file))
            mat.set_principled_shader_value("Roughness", bump_image)
        else:
            mat.set_principled_shader_value("Roughness", np.random.uniform(0.4, 0.9))
            
        materials.append(mat)
        
    return materials

terrain_dir = os.path.join(assets_path, 'terrain')
terrain_images = [os.path.join(terrain_dir, f"{i}.jpg") for i in range(9)]

custom_textures_path = os.path.join(terrain_dir, 'full-textures')
cc_textures = load_custom_pbr_materials(custom_textures_path)

room_material = bproc.material.create('room_terrain_material')

room_planes = [bproc.object.create_primitive('PLANE', scale=[2, 2, 1]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, -2, 2], rotation=[-1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, 2, 2], rotation=[1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[2, 0, 2], rotation=[0, -1.570796, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[-2, 0, 2], rotation=[0, 1.570796, 0])]
for plane in room_planes:
    plane.enable_rigidbody(False, collision_shape='BOX', mass=1.0, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)

light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 10])
light_plane.set_name('light_plane')
light_plane_material = bproc.material.create('light_material')

light_point = bproc.types.Light()
light_point.set_energy(200)

def sample_pose_func(obj: bproc.types.MeshObject):
    min = np.random.uniform([-0.3, -0.3, 0.0], [-0.2, -0.2, 0.0])
    max = np.random.uniform([0.2, 0.2, 0.4], [0.3, 0.3, 0.6])
    obj.set_location(np.random.uniform(min, max))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())
    
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.set_max_amount_of_samples(50)


def apply_sim2real_noise(img_array):
    noisy_images = []
    for img in img_array:
        img_f = img.astype(np.float32)

        row, col, ch = img_f.shape
        mean = 0
        sigma = np.random.uniform(5, 15) # Siła szumu
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        img_f = img_f + gauss

        if np.random.rand() > 0.5:
            blur_size = np.random.choice([3, 5])
            img_f = cv2.GaussianBlur(img_f, (blur_size, blur_size), 0)

        alpha = np.random.uniform(0.8, 1.2)
        beta = np.random.uniform(-20, 20)
        img_f = cv2.convertScaleAbs(img_f, alpha=alpha, beta=beta).astype(np.float32)

        img_f[:, :, 0] *= np.random.uniform(0.9, 1.1) # Kanał R
        img_f[:, :, 1] *= np.random.uniform(0.9, 1.1) # Kanał G
        img_f[:, :, 2] *= np.random.uniform(0.9, 1.1) # Kanał B

        img_final = np.clip(img_f, 0, 255).astype(np.uint8)
        noisy_images.append(img_final)
        
    return noisy_images


for i in tqdm(range(100)):
    random_cc_texture = np.random.choice(cc_textures)
    for plane in room_planes:
        plane.replace_materials(random_cc_texture)

    sampled_target_bop_objs = list(np.random.choice(target_objs, size=1, replace=False))

    for obj in (sampled_target_bop_objs):        
        mat = obj.get_materials()[0]

        if obj.get_cp("bop_dataset_name") in ['itodd', 'tless']:
            grey_col = np.random.uniform(0.1, 0.9)   
            mat.set_principled_shader_value("Base Color", [grey_col, grey_col, grey_col, 1])   

        mat.set_principled_shader_value("Metallic", np.random.uniform(0, 0.8))
        mat.set_principled_shader_value("Roughness", np.random.uniform(0.1, 0.9))
        mat.set_principled_shader_value("Specular IOR Level", np.random.uniform(0.1, 1.0))
        
        obj.enable_rigidbody(True, mass=1.0, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)
        obj.hide(False)
    

    light_plane_material.make_emissive(emission_strength=np.random.uniform(2, 10), 
                                    emission_color=np.random.uniform([0.4, 0.4, 0.4, 1.0], [1.0, 1.0, 1.0, 1.0]))  
    light_plane.replace_materials(light_plane_material)
    
    light_point.set_color(np.random.uniform([0.3, 0.3, 0.3], [1.0, 1.0, 1.0]))
    light_point.set_energy(np.random.uniform(100, 500))

    bproc.object.sample_poses(objects_to_sample = sampled_target_bop_objs,
                            sample_pose_func = sample_pose_func, 
                            max_tries = 1000)
            
    bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=3,
                                                    max_simulation_time=10,
                                                    check_object_interval=1,
                                                    substeps_per_frame = 20,
                                                    solver_iters=25)

    bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(sampled_target_bop_objs)

    cam_poses = 0
    while cam_poses < 5:
        location = bproc.sampler.shell(center = [0, 0, 0],
                                radius_min = 0.1,
                                radius_max = 0.3,
                                elevation_min = 5,
                                elevation_max = 89)
        poi = bproc.object.compute_poi(np.random.choice(sampled_target_bop_objs, size=1, replace=False))

        poi_offset = np.random.uniform([-0.2, -0.2, -0.2], [0.2, 0.2, 0.2])
        shifted_poi = poi + poi_offset

        rotation_matrix = bproc.camera.rotation_from_forward_vec(shifted_poi - location, inplane_rot=np.random.uniform(-3.14159, 3.14159))
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        
        if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.3}, bop_bvh_tree):
            bproc.camera.add_camera_pose(cam2world_matrix, frame=cam_poses)
            cam_poses += 1

    data = bproc.renderer.render()

    data["colors"] = apply_sim2real_noise(data["colors"])

    # Write data in bop format
    bproc.writer.write_bop(os.path.join('output', 'bop_data'),
                           target_objects = sampled_target_bop_objs,
                           dataset = 'probe',
                           depth_scale=0.1,
                           depths = data["depth"],
                           colors = data["colors"], 
                           color_file_format = "JPEG",
                           ignore_dist_thres = 10)
    
    for obj in (sampled_target_bop_objs):      
        obj.disable_rigidbody()
        obj.hide(True)