02_Huvle — Automated Wallpaper Generation Project

A project building a pipeline that generates new wallpapers automatically from user-uploaded images, using image segmentation and image-editing (compositing) models.

Implemented FC-CLIP directly: cloned the repo, added a helper function based on demo.py, and wrote a new extract.py to build a loadable segmentation pipeline

Replaced the Gemini API with Qwen (Qwen/Qwen2.5-VL-7B-Instruct) for the feature that takes a wallpaper image and extracts a title, description, tags, and category

Replaced the segmentation model from FC-CLIP with EOV-Seg
Re-implemented it based on the structure of extract.py / extract.sh from the FC-CLIP folder

1. Installed ComfyUI on a remote server
2. Loaded the Qwen Image-Edit model into the pipeline
3. Pre-placed segments on an empty canvas to determine layout (e.g., placing a dog segment on top of an ocean segment)
4. Composited the final image using a text prompt

Goal: feed actual segment images into the model instead of example images
Currently segments are fed in without compositing → plan to use each segment's mask metadata to make the black background transparent
Improve compositing quality (currently random/unintended people sometimes appear)
Evaluate applying the Qwen-Image-Edit Lightning LoRA (lightx2v/Qwen-Image-Edit-2511-Lightning)
Compare runtime and output quality across methods
Base: 40 / 20 / 10 steps
Lightning: 4 steps


