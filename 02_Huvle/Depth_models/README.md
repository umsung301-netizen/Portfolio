Building the training dataset
Step 1: given a single wallpaper, load its segments and sort them by depth
Compared 3 depth estimation models — Depth Anything V2, Marigold, and MiDaS — on the same image (201701crop_ib0s05n74_4C9vPbFFxd.jpg); code for each model is under search/depth_models

Depth Anything V2 and Marigold agreed on the background-to-foreground order (2 → 1 → 3)
MiDaS produced a different order (1 → 2 → 3)
