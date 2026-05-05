import numpy as np

PointXYZ = np.dtype([
    ("x", np.float32),
    ("y", np.float32),
    ("z", np.float32),
    ("_stride_0", np.float32),
])
PointXYZI = np.dtype([
    ("x", np.float32),
    ("y", np.float32),
    ("z", np.float32),
    ("_stride_0", np.float32),
    ("intensity", np.float32),
    ("_stride_1", np.uint32),
    ("_stride_2", np.uint32),
    ("_stride_3", np.uint32),
])
PointXYZICT = np.dtype([
    ("x", np.float32),
    ("y", np.float32),
    ("z", np.float32),
    ("_stride_0", np.float32),
    ("intensity", np.float32),
    ("channel", np.uint16),
    ("_stride_1", np.uint16),
    ("time", np.float32),
    ("_stride_2", np.uint32),
])
PointXYZL = np.dtype([
    ("x", np.float32),
    ("y", np.float32),
    ("z", np.float32),
    ("_stride_0", np.float32),
    ("label", np.uint32),
    ("_stride_1", np.uint32),
    ("_stride_2", np.uint32),
    ("_stride_3", np.uint32),
])
PointXYZTMRV = np.dtype([
    ("x", np.float32),
    ("y", np.float32),
    ("z", np.float32),
    ("time", np.float32),
    ("moving", np.uint8),
    ("_stride_0", np.uint8),
    ("_stride_1", np.uint16),
    ("range_var", np.float32),
    ("vel_var", np.float32),
    ("_stride_2", np.uint32),
])
PointXYZIVT = np.dtype([
    ("x", np.float32),
    ("y", np.float32),
    ("z", np.float32),
    ("_stride_0", np.float32),
    ("intensity", np.float32),
    ("vel_var", np.float32),
    ("time", np.float32),
    ("_stride_1", np.uint32),
])
PointXYZICTRGB = np.dtype([
    ("x", np.float32),
    ("y", np.float32),
    ("z", np.float32),
    ("_stride_0", np.float32),
    ("intensity", np.float32),
    ("channel", np.uint16),
    ("_stride_1", np.uint16),
    ("time", np.float32),
    ("r", np.float32),
    ("g", np.float32),
    ("b", np.float32),
    ("_stride_2", np.uint32),
    ("_stride_3", np.uint32),
])

POINT_IDX = {
    0: PointXYZ,
    1: PointXYZI,
    2: PointXYZICT,
    3: PointXYZL,
    4: PointXYZTMRV,
    5: PointXYZIVT,
    6: PointXYZICTRGB,
}

POINT_TYPE = {
    PointXYZ: "PointXYZ",
    PointXYZI: "PointXYZI",
    PointXYZICT: "PointXYZICT",
    PointXYZL: "PointXYZL",
    PointXYZTMRV: "PointXYZTMRV",
    PointXYZIVT: "PointXYZIVT",
    PointXYZICTRGB: "PointXYZICTRGB",
}


def protoCloudToNumpy(msg):
    data = np.frombuffer(msg.data, dtype=POINT_IDX[msg.type], count=msg.height)
    fields = [name for name in data.dtype.names if 'stride' not in name]
    return np.array([[float(x) for x in tup] for tup in data[fields]])


def typeToPoint(typept):
    return POINT_TYPE[POINT_IDX[typept]]


def protoCloudToPcdFile(msg):
    points = protoCloudToNumpy(msg)
    xyz = points[:, :3].tobytes()
	# All the pointcloud types have x y z fields and they are all 4 bytes
    return (
        "# .PCD v0.7 - Point Cloud Data file format\n"
		"VERSION 0.7\n"
		"FIELDS x y z\n"
		"SIZE 4 4 4\n"
		"TYPE F F F\n"
		"COUNT 1 1 1\n"
		f"WIDTH {msg.height}\n"
		"HEIGHT 1\n"
		"VIEWPOINT 0 0 0 1 0 0 0\n"
		f"POINTS {msg.height}\n"
		"DATA binary\n"
	).encode() + xyz
