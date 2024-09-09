import os
import numpy as np
from tqdm import tqdm
import sys
from datasets import point_cloud_to_range_image

def cartesian_to_spherical(cartesian_points):
    """
    Convert Cartesian coordinates to spherical coordinates (r, azimuth, elevation).
    
    Parameters:
        cartesian_points: numpy array of shape (N, 3) representing N points in Cartesian coordinates.
        
    Returns:
        spherical_points: numpy array of shape (N, 3) representing N points in spherical coordinates (r, azimuth, elevation).
    """
    x, y, z = cartesian_points[:, 0], cartesian_points[:, 1], cartesian_points[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    azimuth = -np.arctan2(y, x) # azimuth angle
    elevation = np.arcsin(z / r) # elevation angle
    return np.column_stack((r, azimuth, elevation))


def projection_bbox(yaw, pitch):
    fov_up = 3.0 / 180.0 * np.pi   
    fov_down = -25.0 / 180.0 * np.pi
    fov = abs(fov_down) + abs(fov_up)

    proj_x = 0.5 * (yaw / np.pi + 1.0)          
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov

    proj_x *= 2048                              
    proj_y *= 64             

    proj_x = np.floor(proj_x).astype(np.int32)
    proj_x = np.minimum(2048 - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)  
    
    proj_y = np.floor(proj_y).astype(np.int32)
    proj_y = np.minimum(64 - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)  

    return proj_x, proj_y




def extract_detection(file_name):
    kitti_gt = 'path/velodyne'
    calib_path = 'path/calib'
    label_path = 'path/label_2'
    
    bin_real = os.path.join(kitti_gt, file_name + '.bin')
    bin_real_pcd = np.fromfile(bin_real, dtype=np.float32)
    points = bin_real_pcd.reshape((-1, 4))[:, 0:3]
    points2 = bin_real_pcd.reshape((-1, 4))
    points2 = points2[np.any(points2 != 0, axis=1)]
    points2 = points2[(points2[:,2]>=-3)]
    points2 = points2[np.sqrt(points2[:,0]**2 + points2[:,1]**2 + points2[:,2]**2) > 3.0]




    with open(os.path.join(calib_path, file_name + '.txt'), 'r') as f:
        calib = f.readlines()
        P2 = np.array([float(x) for x in calib[2].strip().split()[1:]]).reshape(3, 4)
        R0_rect = np.eye(4) 
        R0_rect[:3, :3] = np.array([float(x) for x in calib[4].strip().split()[1:]]).reshape(3, 3)
        Tr_velo_to_cam = np.eye(4) 
        Tr_velo_to_cam[:3, :4] = np.array([float(x) for x in calib[5].strip().split()[1:]]).reshape(3, 4)

    with open(os.path.join(label_path, file_name + '.txt'), 'r') as f:
        bboxes = [list(map(float, line.split()[4:8])) for line in f if line.split()[0] == 'Car']
        
        

    points_hom = np.hstack((points2[:, :3], np.ones((points2.shape[0], 1))))

    cam_coords = P2 @ R0_rect @ Tr_velo_to_cam @ points_hom.T

    cam_coords[:2] /= cam_coords[2]

    front_points_mask = cam_coords[2] > 0
    points1 = points2[front_points_mask, :4]
    cam_coords = cam_coords[:, front_points_mask]
    image_area_points = []
    new_bbox = []
    img_data = {
            'image': [],
            'coordinates': []
        }
    for i, bbox in enumerate(bboxes):
        xmin, ymin, xmax, ymax = bbox
        mask = (cam_coords[0] >= xmin) & (cam_coords[0] <= xmax) & \
            (cam_coords[1] >= ymin) & (cam_coords[1] <= ymax)
        filtered_points = points1[np.where(mask)[0], :4]
        sp_points = cartesian_to_spherical(filtered_points[:,:3])
        if len(sp_points) == 0:
            continue

        box_phi_min = np.min(sp_points[:,1])
        box_phi_max = np.max(sp_points[:,1])
        box_theta_min = np.min(sp_points[:,2])
        box_theta_max = np.max(sp_points[:,2])

        i_box_phi_min, i_box_theta_min = projection_bbox(box_phi_min, box_theta_min)
        i_box_phi_max, i_box_theta_max = projection_bbox(box_phi_max, box_theta_max)
        if (i_box_phi_min == i_box_phi_max) or (i_box_theta_min == i_box_theta_max):
            continue            
        
        
        a = point_cloud_to_range_image(points2[:,:3], points2[:,3], True, return_remission=True)
        b = np.stack(a, axis=0)
        k = np.zeros_like(b)
        def adjust_min_to_multiple_of_16(value):
            return (value // 8) * 8

        def adjust_max_to_multiple_of_16(value, upper_limit):
            adjusted_value = ((value + 7) // 8) * 8
            return min(adjusted_value, upper_limit)


        x_min = adjust_min_to_multiple_of_16(i_box_phi_min)
        x_max = adjust_max_to_multiple_of_16(i_box_phi_max, 2048)


        y_min = adjust_min_to_multiple_of_16(max(i_box_theta_max - 1, 0))
        y_max = adjust_max_to_multiple_of_16(min(i_box_theta_min + 1, 64), 64)
        
        k[:, y_min:y_max, x_min:x_max] = b[:, y_min:y_max, x_min:x_max]
        
        
        img = k[:, y_min:y_max, x_min:x_max]
        img = np.where(img <= 0, 0, img)
        img_data['image'].append(img)
        img_data['coordinates'].append([x_min, x_max, y_min, y_max])
        assert len(img_data['image']) == len(img_data['coordinates'])
    return img_data