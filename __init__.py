NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def remote_control():
	global NODE_CLASS_MAPPINGS
	global NODE_DISPLAY_NAME_MAPPINGS
	from .nodes.remote_control import QueueRemote, FetchRemote
	NODE_CLASS_MAPPINGS.update({
		"QueueRemote": QueueRemote,
		"FetchRemote": FetchRemote,
	})
	NODE_DISPLAY_NAME_MAPPINGS.update({
		"QueueRemote": "Queue on remote",
		"FetchRemote": "Fetch from remote",
	})

def remote_images():
	global NODE_CLASS_MAPPINGS
	global NODE_DISPLAY_NAME_MAPPINGS
	from .nodes.remote_images import LoadImageUrl, SaveImageUrl, SaveImageToS3, SaveVideoToS3, VideoCombineForS3, SimulateMask
	NODE_CLASS_MAPPINGS.update({
		"LoadImageUrl": LoadImageUrl,
		"SaveImageUrl": SaveImageUrl,
		"SaveImageToS3": SaveImageToS3,
		"SaveVideoToS3": SaveVideoToS3,
		"VideoCombineForS3": VideoCombineForS3,
		"SimulateMask": SimulateMask,
	})
	NODE_DISPLAY_NAME_MAPPINGS.update({
		"LoadImageUrl": "Load Image (URL)",
		"SaveImageUrl": "Save Image (URL)",
		"SaveImageToS3": "Save Image to S3",
		"SaveVideoToS3": "Save Video to S3",
		"VideoCombineForS3": "Combine Video for S3",
		"SimulateMask": "Create simulate mask when null",
	})

def remote_misc():
	global NODE_CLASS_MAPPINGS
	global NODE_DISPLAY_NAME_MAPPINGS
	from .nodes.misc import CombineImageBatch
	NODE_CLASS_MAPPINGS.update({
		"CombineImageBatch": CombineImageBatch,
	})
	NODE_DISPLAY_NAME_MAPPINGS.update({
		"CombineImageBatch": "Combine images",
	})

print("Loading network distribution node pack")
remote_control()
remote_images()
remote_misc()
