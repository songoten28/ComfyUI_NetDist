import os
import json
import torch
import requests
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from base64 import b64encode
from io import BytesIO
import boto3

s3_client = boto3.client('s3', aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'), aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'))

ffmpeg_path = shutil.which("ffmpeg")
if ffmpeg_path is None:
	logger.info("ffmpeg could not be found. Using ffmpeg from imageio-ffmpeg.")
	from imageio_ffmpeg import get_ffmpeg_exe
	try:
		ffmpeg_path = get_ffmpeg_exe()
	except:
		logger.warning("ffmpeg could not be found. Outputs that require it have been disabled")

class LoadImageUrl:
	def __init__(self):
		pass

	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"url": ("STRING", { "multiline": False, })
			}
		}

	RETURN_TYPES = ("IMAGE", "MASK")
	FUNCTION = "load_image_url"
	CATEGORY = "remote"

	def load_image_url(self, url):
		with requests.get(url, stream=True) as r:
			r.raise_for_status()
			i = Image.open(r.raw)
		image = i.convert("RGB")
		image = np.array(image).astype(np.float32) / 255.0
		image = torch.from_numpy(image)[None,]
		if 'A' in i.getbands():
			mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
			mask = 1. - torch.from_numpy(mask)
		else:
			mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
		return (image, mask)

class SaveImageUrl:
	def __init__(self):
		pass

	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"images": ("IMAGE", ),
				"url": ("STRING", { "multiline": False, }),
				"filename_prefix": ("STRING", {"default": "ComfyUI"}),
				"data_format": (["HTML_image", "Raw_data"],)
			},
			"hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
		}

	RETURN_TYPES = ()
	OUTPUT_NODE = True
	FUNCTION = "save_images"
	CATEGORY = "remote"
	
	def save_images(self, images, url, data_format, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
		filename = os.path.basename(os.path.normpath(filename_prefix))

		counter = 1
		data = {}
		for image in images:
			i = 255. * image.cpu().numpy()
			img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
			meta = PngInfo()
			if prompt is not None:
				meta.add_text("prompt", json.dumps(prompt))
			if extra_pnginfo is not None:
				for x in extra_pnginfo:
					meta.add_text(x, json.dumps(extra_pnginfo[x]))
		
			file = f"{filename}_{counter:05}.png"

			buffer = BytesIO()
			img.save(buffer, "png", pnginfo=meta, compress_level=4)
			buffer.seek(0)
			encoded = b64encode(buffer.read()).decode('utf-8')
			data[file] = f"data:image/png;base64,{encoded}" if data_format == "HTML_image" else encoded
			counter += 1

		with requests.post(url, json=data) as r:
			r.raise_for_status()
		return ()

class SaveImageToS3:
	def __init__(self):
		pass

	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"images": ("IMAGE", ),
				"bucket": ("STRING", { "multiline": False, }),
				"filename_prefix": ("STRING", {"default": "ComfyUI"}),
				"folder": ("STRING", { "multiline": False, }),
			},
			"hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
		}

	RETURN_TYPES = ()
	OUTPUT_NODE = True
	FUNCTION = "upload_images"
	CATEGORY = "s3"

	def upload_images(self, images, bucket, folder, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
		filename = os.path.basename(os.path.normpath(filename_prefix))

		counter = 1
		for image in images:
			i = 255. * image.cpu().numpy()
			img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
			meta = PngInfo()
			if prompt is not None:
				meta.add_text("prompt", json.dumps(prompt))
			if extra_pnginfo is not None:
				for x in extra_pnginfo:
					meta.add_text(x, json.dumps(extra_pnginfo[x]))

			file = f"{filename}_{counter:05}.png"

			buffer = BytesIO()
			img.save(buffer, "png", pnginfo=meta, compress_level=4)
			buffer.seek(0)
			counter += 1
			s3_client.put_object(Body=buffer, Bucket=bucket, Key=f"{folder}/{file}", ContentType='image/png', ContentEncoding='base64')

		return ()

class VideoCombineForS3:
	@classmethod
	def INPUT_TYPES(s):
		#Hide ffmpeg formats if ffmpeg isn't available
		if ffmpeg_path is not None:
			ffmpeg_formats = ["video/"+x[:-5] for x in folder_paths.get_filename_list("video_formats")]
		else:
			ffmpeg_formats = []
		return {
			"required": {
				"images": ("IMAGE",),
				"frame_rate": (
					"INT",
					{"default": 8, "min": 1, "step": 1},
				),
				"loop_count": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
				"filename_prefix": ("STRING", {"default": "AnimateDiff"}),
				"format": (["image/gif", "image/webp"] + ffmpeg_formats,),
				"pingpong": ("BOOLEAN", {"default": False}),
				"save_image": ("BOOLEAN", {"default": True}),
				"crf": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1}),
			},
			"hidden": {
				"prompt": "PROMPT",
				"extra_pnginfo": "EXTRA_PNGINFO",
			},
		}

	RETURN_TYPES = ("INFORMATION",)
	OUTPUT_NODE = True
	CATEGORY = "s3"
	FUNCTION = "combine_video"

	def save_with_tempfile(self, args, metadata, file_path, frames, env, crf):
		#Ensure temp directory exists
		os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)

		metadata_path = os.path.join(folder_paths.get_temp_directory(), "metadata.txt")
		#metadata from file should  escape = ; # \ and newline
		#From my testing, though, only backslashes need escapes and = in particular causes problems
		#It is likely better to prioritize future compatibility with containers that don't support
		#or shouldn't use the comment tag for embedding metadata
		metadata = metadata.replace("\\","\\\\")
		metadata = metadata.replace(";","\\;")
		metadata = metadata.replace("#","\\#")
		#metadata = metadata.replace("=","\\=")
		metadata = metadata.replace("\n","\\\n")
		with open(metadata_path, "w") as f:
			f.write(";FFMETADATA1\n")
			f.write(metadata)
		args = args[:1] + ["-i", metadata_path] + args[1:] + [file_path]
		with subprocess.Popen(args, stdin=subprocess.PIPE, env=env) as proc:
			for frame in frames:
				proc.stdin.write(frame.tobytes())

	def combine_video(
			self,
			images,
			crf,
			frame_rate: int,
			loop_count: int,
			filename_prefix="AnimateDiff",
			format="image/gif",
			pingpong=False,
			save_image=True,
			prompt=None,
			extra_pnginfo=None,
	):
		# convert images to numpy
		frames: List[Image.Image] = []
		for image in images:
			img = 255.0 * image.cpu().numpy()
			img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
			frames.append(img)

		# get output information
		output_dir = (
			folder_paths.get_output_directory()
			if save_image
			else folder_paths.get_temp_directory()
		)
		(
			full_output_folder,
			filename,
			_,
			subfolder,
			_,
		) = folder_paths.get_save_image_path(filename_prefix, output_dir)

		metadata = PngInfo()
		video_metadata = {}
		if prompt is not None:
			metadata.add_text("prompt", json.dumps(prompt))
			video_metadata["prompt"] = prompt
		if extra_pnginfo is not None:
			for x in extra_pnginfo:
				metadata.add_text(x, json.dumps(extra_pnginfo[x]))
				video_metadata[x] = extra_pnginfo[x]

		# comfy counter workaround
		max_counter = 0

		# Loop through the existing files
		matcher = re.compile(f"{re.escape(filename)}_(\d+)_?\.[a-zA-Z0-9]+")
		for existing_file in os.listdir(full_output_folder):
			# Check if the file matches the expected format
			match = matcher.fullmatch(existing_file)
			if match:
				# Extract the numeric portion of the filename
				file_counter = int(match.group(1))
				# Update the maximum counter value if necessary
				if file_counter > max_counter:
					max_counter = file_counter

		# Increment the counter by 1 to get the next available value
		counter = max_counter + 1

		# save first frame as png to keep metadata
		file = f"{filename}_{counter:05}.png"
		file_path = os.path.join(full_output_folder, file)
		frames[0].save(
			file_path,
			pnginfo=metadata,
			compress_level=4,
		)
		if pingpong:
			frames = frames + frames[-2:0:-1]

		format_type, format_ext = format.split("/")
		file = f"{filename}_{counter:05}.{format_ext}"
		file_path = os.path.join(full_output_folder, file)
		if format_type == "image":
			# Use pillow directly to save an animated image
			frames[0].save(
				file_path,
				format=format_ext.upper(),
				save_all=True,
				append_images=frames[1:],
				duration=round(1000 / frame_rate),
				loop=loop_count,
				compress_level=4,
			)
		else:
			# Use ffmpeg to save a video
			if ffmpeg_path is None:
				#Should never be reachable
				raise ProcessLookupError("Could not find ffmpeg")

			video_format_path = folder_paths.get_full_path("video_formats", format_ext + ".json")
			with open(video_format_path, 'r') as stream:
				video_format = json.load(stream)
			file = f"{filename}_{counter:05}.{video_format['extension']}"
			file_path = os.path.join(full_output_folder, file)
			dimensions = f"{frames[0].width}x{frames[0].height}"
			metadata_args = ["-metadata", "comment=" + json.dumps(video_metadata)]
			args = [ffmpeg_path, "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
					"-s", dimensions, "-r", str(frame_rate), "-i", "-", "-crf", str(crf) ] \
				   + video_format['main_pass']
			# On linux, max arg length is Pagesize * 32 -> 131072
			# On windows, this around 32767 but seems to vary wildly by > 500
			# in a manor not solely related to other arguments
			if os.name == 'posix':
				max_arg_length = 4096*32
			else:
				max_arg_length = 32767 - len(" ".join(args + [metadata_args[0]] + [file_path])) - 1
			#test max limit
			#metadata_args[1] = metadata_args[1] + "a"*(max_arg_length - len(metadata_args[1])-1)

			env=os.environ.copy()
			if  "environment" in video_format:
				env.update(video_format["environment"])
			if len(metadata_args[1]) >= max_arg_length:
				logger.info(f"Using fallback file for long metadata: {len(metadata_args[1])}/{max_arg_length}")
				self.save_with_tempfile(args, metadata_args[1], file_path, frames, env, crf)
			else:
				try:
					with subprocess.Popen(args + metadata_args + [file_path],
										  stdin=subprocess.PIPE, env=env) as proc:
						for frame in frames:
							proc.stdin.write(frame.tobytes())
				except FileNotFoundError as e:
					if "winerror" in dir(e) and e.winerror == 206:
						logger.warn("Metadata was too long. Retrying with fallback file")
						self.save_with_tempfile(args, metadata_args[1], file_path, frames, env, crf)
					else:
						raise
				except OSError as e:
					if "errno" in dir(e) and e.errno == 7:
						logger.warn("Metadata was too long. Retrying with fallback file")
						self.save_with_tempfile(args, metadata_args[1], file_path, frames, env, crf)
					else:
						raise

		previews = [
			{
				"filename": file,
				"subfolder": subfolder,
				"type": "output" if save_image else "temp",
				"format": format,
			}
		]
		return {"ui": {"gifs": previews}}


class SaveVideoToS3:
	def __init__(self):
		pass

	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"gif": ("INFORMATION", ),
				"bucket": ("STRING", { "multiline": False, }),
				"filename_prefix": ("STRING", {"default": "ComfyUI"}),
				"folder": ("STRING", { "multiline": False, }),
			},
			"hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
		}

	RETURN_TYPES = ()
	OUTPUT_NODE = True
	FUNCTION = "save_video"
	CATEGORY = "s3"

	def save_video(self, gif, bucket, folder, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
		filename = os.path.basename(os.path.normpath(filename_prefix))
		print("gif", gif)

		return ()