
import numpy as np




class LaserScan:
  """Class that contains LaserScan with x,y,z,r"""


  def __init__(self, project=True, H=64, W=1024, fov_up=3.0, fov_down=-25.0):
    self.project = project
    self.proj_H = H
    self.proj_W = W
    self.proj_fov_up = fov_up
    self.proj_fov_down = fov_down
    self.reset()


  def reset(self):
    """ Reset scan members. """
    self.points = np.zeros((0, 3), dtype=np.float32)        # [m, 3]: x, y, z
    self.remissions = np.zeros((0, 1), dtype=np.float32)    # [m ,1]: remission


    # projected range image - [H,W] range (-1 is no data)
    self.proj_range = np.full((self.proj_H, self.proj_W), -1,
                              dtype=np.float32)


    # unprojected range (list of depths for each point)
    self.unproj_range = np.zeros((0, 1), dtype=np.float32)


    # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
    self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1,
                            dtype=np.float32)


    # projected remission - [H,W] intensity (-1 is no data)
    self.proj_remission = np.full((self.proj_H, self.proj_W), -1,
                                  dtype=np.float32)


    # projected index (for each pixel, what I am in the pointcloud)
    # [H,W] index (-1 is no data)
    self.proj_idx = np.full((self.proj_H, self.proj_W), -1,
                            dtype=np.int32)


    # for each point, where it is in the range image
    self.proj_x = np.zeros((0, 1), dtype=np.int32)        # [m, 1]: x
    self.proj_y = np.zeros((0, 1), dtype=np.int32)        # [m, 1]: y


    # mask containing for each pixel, if it contains a point or not
    self.proj_mask = np.zeros((self.proj_H, self.proj_W),
                              dtype=np.int32)       # [H,W] mask


  def size(self):
    """ Return the size of the point cloud. """
    return self.points.shape[0]


  def __len__(self):
    return self.size()


  def open_scan(self, scan):
    """ Open raw scan and fill in attributes
    """
    # reset just in case there was an open structure
    self.reset()
    scan = np.asarray(scan, dtype=np.float32).reshape((-1, 4))


    # put in attribute
    points = scan[:, 0:3]    # get xyz
    remissions = scan[:, 3]  # get remission
    out_dict = self.set_points(points, remissions, scan)
    return out_dict


  def set_points(self, points, remissions=None, scan=None):
    """ Set scan attributes (instead of opening from file)
    """
    # reset just in case there was an open structure
    self.reset()


    # # check scan makes sense
    # if not isinstance(points, np.ndarray):
    #   raise TypeError("Scan should be numpy array")


    # # check remission makes sense
    # if remissions is not None and not isinstance(remissions, np.ndarray):
    #   raise TypeError("Remissions should be numpy array")


    self.scan = scan


    # put in attribute
    self.points = points    # get xyz
    if remissions is not None:
      self.remissions = remissions  # get remission
    else:
      self.remissions = np.zeros((points.shape[0]), dtype=np.float32)


    # if projection is wanted, then do it and fill in the structure
    if self.project:
      out_dict = self.do_range_projection()


    return out_dict


  def do_range_projection(self):
    """ Project a pointcloud into a spherical projection image.projection.
        Function takes no arguments because it can be also called externally
        if the value of the constructor was not set (in case you change your
        mind about wanting the projection)
    """
    # laser parameters
    fov_up = self.proj_fov_up / 180.0 * np.pi      # field of view up in rad
    fov_down = self.proj_fov_down / 180.0 * np.pi  # field of view down in rad
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad


    # get depth of all points
    depth = np.linalg.norm(self.points, 2, axis=1) # r


    # get scan components
    scan_x = self.points[:, 0]
    scan_y = self.points[:, 1]
    scan_z = self.points[:, 2]


    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)


    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]


    # scale to image size using angular resolution
    proj_x *= self.proj_W                              # in [0.0, W]
    proj_y *= self.proj_H                              # in [0.0, H]




    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(self.proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]
    self.proj_x = np.copy(proj_x)  # store a copy in orig order


    proj_y = np.floor(proj_y)
    proj_y = np.minimum(self.proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]
    self.proj_y = np.copy(proj_y)  # stope a copy in original order


    # copy of depth in original order
    self.unproj_range = np.copy(depth)


    # order in decreasing depth，从远到近排序
    indices = np.arange(depth.shape[0]) # 固定步长的排列，[0,shape[0]-1],步长1
    order = np.argsort(depth)[::-1] # 从大到小,远2近
    depth = depth[order]
    indices = indices[order]
    points = self.points[order]
    remission = self.remissions[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]
    scan = self.scan[order]


    # assing to images
    self.proj_range[proj_y, proj_x] = depth
    # print(self.proj_range[0,0])
    a = np.expand_dims(self.proj_range, axis=2)


    self.proj_xyz[proj_y, proj_x] = points
    b = self.proj_xyz


    self.proj_remission[proj_y, proj_x] = remission
    c = np.expand_dims(self.proj_remission, axis=2)


    self.proj_idx[proj_y, proj_x] = indices
    self.proj_mask = (self.proj_idx > 0).astype(np.int32)


    self.proj_i = np.concatenate((a,b,c), 2)




    out_dict = {
      'range': self.proj_range.astype(np.float32),
      'ori_xyz': self.proj_xyz.astype(np.float32),
      'ori_r': self.proj_remission.astype(np.float32),
      'idx': self.proj_idx.astype(np.int32),
      'mask': self.proj_mask.astype(np.int32),
      'range_in': self.proj_i.astype(np.float32),
      'y': proj_y.astype(np.float32),
      'x': proj_x.astype(np.float32),
      'points': scan.astype(np.float32),
    }


    return out_dict
