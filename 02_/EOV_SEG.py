# 세그먼트 모델 교체
# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import subprocess
import json


class QwenServer:
    def __init__(self):
        env = os.environ.copy()
        # 외부 터미널 설정을 최우선으로 따르고, 설정이 없으면 기본값 '0'번 GPU 사용
        env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        print(f"Qwen3-VL 서버를 독립된 환경(GPU: {env['CUDA_VISIBLE_DEVICES']})에서 시작합니다...")
        
        # 하드코딩된 절대 경로 제거 (sys.executable로 현재 가상환경 파이썬 자동 인식)
        self.process = subprocess.Popen(
            [
                sys.executable,
                "qwen_server.py",  # qwen_server.py가 현재 실행 위치와 같은 폴더에 있다고 가정
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            env=env,
        )
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("Qwen 서버 초기화 실패")
            if "READY" in line:
                break
        print("Qwen 서버 준비 완료!")

    def process_image(self, path):
        self.process.stdin.write(path + "\n")
        self.process.stdin.flush()
        while True:
            line = self.process.stdout.readline()
            if not line:
                return {"error": "Server connection lost"}
            if line.startswith("RESULT:"):
                data = json.loads(line[len("RESULT:") :])
                if data.get("status") == "success":
                    return data["data"]
                else:
                    return {"error": data.get("message")}

    def close(self):
        self.process.stdin.write("QUIT\n")
        self.process.stdin.flush()
        self.process.wait()


HEAVY_DEPS_AVAILABLE = True
try:
    import cv2
    import numpy as np
    import tqdm
    from matplotlib import pyplot as plt

    from detectron2.config import get_cfg
    from detectron2.data.detection_utils import read_image
    from detectron2.projects.deeplab import add_deeplab_config
    from detectron2.utils.logger import setup_logger

    from eov_seg import add_eov_config
    from eov_seg.config import add_maskformer2_config  # <== 이 부분을 추가
    from demo.predictor import VisualizationDemo
except Exception:
    HEAVY_DEPS_AVAILABLE = False
    cv2 = None
    np = None
    tqdm = None
    plt = None
    get_cfg = None
    read_image = None
    add_deeplab_config = None
    setup_logger = None
    add_eov_config = None
    add_maskformer2_config = None
    VisualizationDemo = None


def create_cityscapes_label_colormap():
    """Creates a label colormap used in CITYSCAPES segmentation benchmark.
    Returns:
        A colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap[0] = [128, 64, 128]
    colormap[1] = [244, 35, 232]
    colormap[2] = [70, 70, 70]
    colormap[3] = [102, 102, 156]
    colormap[4] = [190, 153, 153]
    colormap[5] = [153, 153, 153]
    colormap[6] = [250, 170, 30]
    colormap[7] = [220, 220, 0]
    colormap[8] = [107, 142, 35]
    colormap[9] = [152, 251, 152]
    colormap[10] = [70, 130, 180]
    colormap[11] = [220, 20, 60]
    colormap[12] = [255, 0, 0]
    colormap[13] = [0, 0, 142]
    colormap[14] = [0, 0, 70]
    colormap[15] = [0, 60, 100]
    colormap[16] = [0, 80, 100]
    colormap[17] = [0, 0, 230]
    colormap[18] = [119, 11, 32]
    return colormap


def label_to_color_image(label):

    colormap = create_cityscapes_label_colormap()
    return colormap[label]


# constants
WINDOW_NAME = "mask2former demo"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_eov_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/eov_seg/maskformer2_R50_bs16_50ep.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--webcam", action="store_true", help="Take inputs from webcam."
    )
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", type=str)
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def is_blank_meta(gm):
    if gm is None:
        return True
    if not isinstance(gm, dict):
        return True
    # consider blank if all values are empty strings/lists or None
    for v in gm.values():
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        if isinstance(v, (list, dict)) and len(v) == 0:
            continue
        return False
    return True


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


# predictions에서 segmentmap 꺼내서 임베딩으로 바꾸는 함수

import torch
from PIL import Image

# clip is optional for regen-only runs
try:
    import clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = clip.load("ViT-B/16", device=device)
    clip_model.eval()
except Exception:
    clip = None
    clip_model = None
    clip_preprocess = None


def extract_and_save(img, predictions, output_dir, path, global_meta, rel_dir="."):
    panoptic_seg, segments_info = predictions["panoptic_seg"]
    panoptic_map = panoptic_seg.cpu().numpy()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    crops = []
    base_name = os.path.basename(path)

    for seg in segments_info:
        seg_id = seg["id"]
        seg["source_image"] = (
            base_name  # 세그먼트가 원본 이미지를 추적할 수 있도록 파일명 저장
        )
        mask = panoptic_map == seg_id

        if not np.any(mask):
            continue

        rows, cols = np.where(mask)
        if rows.size == 0 or cols.size == 0:
            continue

        y1, y2 = rows.min(), rows.max()
        x1, x2 = cols.min(), cols.max()
        crop = img_rgb[y1 : y2 + 1, x1 : x2 + 1].copy()
        mask_crop = mask[y1 : y2 + 1, x1 : x2 + 1]
        crop[~mask_crop] = 0

        crops.append(crop)

    if crops:
        batch = torch.stack(
            [clip_preprocess(Image.fromarray(crop)) for crop in crops],
            dim=0,
        ).to(device)

        with torch.no_grad():
            embeddings = clip_model.encode_image(batch)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        embeddings = embeddings.cpu().numpy()
    else:
        embeddings = np.empty((0, clip_model.visual.output_dim))  # or just np.array([])

    import pickle

    # 결과 폴더 생성
    mask_dir = (
        os.path.join(output_dir, "mask", rel_dir)
        if rel_dir != "."
        else os.path.join(output_dir, "mask")
    )
    meta_dir = (
        os.path.join(output_dir, "meta", rel_dir)
        if rel_dir != "."
        else os.path.join(output_dir, "meta")
    )
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(path))[0]

    # mask는 텐서로 저장
    mask_path = os.path.join(mask_dir, f"{base_name}.pt")
    torch.save(torch.from_numpy(panoptic_map), mask_path)

    # embedding/metadata는 분리해서 저장 (벡터 DB 용도)
    meta_data = {
        "global_metadata": global_meta,
        "segments_info": segments_info,
        "embeddings": embeddings,
    }
    meta_path = os.path.join(meta_dir, f"{base_name}.pkl")

    with open(meta_path, "wb") as f:
        pickle.dump(meta_data, f)


def append_error_log(output_dir, record):
    if not output_dir:
        return
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "extract_errors.jsonl")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    if HEAVY_DEPS_AVAILABLE and setup_logger is not None:
        setup_logger(name="fvcore")
        logger = setup_logger()
    else:
        import logging

        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
        logger = logging.getLogger("regen_meta")

    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)
    qwen_server = QwenServer() if args.output else None

    import threading
    import queue

    q_qwen = queue.Queue(maxsize=16)
    q_save = queue.Queue(maxsize=16)
    qwen_thread = None
    save_thread = None

    def qwen_worker():
        while True:
            task = q_qwen.get()
            if task is None:
                q_qwen.task_done()
                q_save.put(None)
                break
            (
                img,
                predictions,
                path,
                rel_dir,
                visualized_output,
                out_filename,
                args_output,
            ) = task
            try:
                print(
                    f"[Worker] Qwen VLM으로 {os.path.basename(path)} 메타데이터를 추출합니다..."
                )
                global_meta = (
                    qwen_server.process_image(path)
                    if qwen_server
                    else {"error": "Server not initialized"}
                )
                if isinstance(global_meta, dict) and global_meta.get("error"):
                    append_error_log(
                        args_output,
                        {
                            "type": "metadata",
                            "path": path,
                            "error": global_meta.get("error"),
                        },
                    )
                elif is_blank_meta(global_meta):
                    append_error_log(
                        args_output,
                        {
                            "type": "metadata",
                            "path": path,
                            "error": "empty metadata",
                        },
                    )
                q_save.put(
                    (
                        img,
                        predictions,
                        path,
                        rel_dir,
                        visualized_output,
                        out_filename,
                        args_output,
                        global_meta,
                    )
                )
            except Exception as e:
                logger.error(f"[Worker] Error processing {path}: {e}")
                append_error_log(
                    args_output,
                    {
                        "type": "metadata",
                        "path": path,
                        "error": str(e),
                    },
                )
            finally:
                q_qwen.task_done()

    def save_worker():
        while True:
            task = q_save.get()
            if task is None:
                q_save.task_done()
                break
            (
                img,
                predictions,
                path,
                rel_dir,
                visualized_output,
                out_filename,
                args_output,
                global_meta,
            ) = task
            try:
                visualized_output.save(out_filename)
                extract_and_save(
                    img, predictions, args_output, path, global_meta, rel_dir
                )
            except Exception as e:
                logger.error(f"[Worker] Error saving {path}: {e}")
                append_error_log(
                    args_output,
                    {
                        "type": "save",
                        "path": path,
                        "error": str(e),
                    },
                )
            finally:
                q_save.task_done()

    if args.output:
        qwen_thread = threading.Thread(target=qwen_worker, daemon=True)
        save_thread = threading.Thread(target=save_worker, daemon=True)
        qwen_thread.start()
        save_thread.start()

    if args.input:
        img_pths = []
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        for root, dirs, files in os.walk(args.input):
            for img_name in files:
                if os.path.splitext(img_name)[1].lower() in valid_exts:
                    img_pths.append(os.path.join(root, img_name))

        for path in tqdm.tqdm(img_pths):
            if args.output:
                rel_dir = os.path.relpath(os.path.dirname(path), args.input)
                meta_dir = (
                    os.path.join(args.output, "meta", rel_dir)
                    if rel_dir != "."
                    else os.path.join(args.output, "meta")
                )
                base_name = os.path.splitext(os.path.basename(path))[0]
                meta_path = os.path.join(meta_dir, f"{base_name}.pkl")

                if os.path.exists(meta_path):
                    try:
                        import pickle

                        with open(meta_path, "rb") as f:
                            existing_meta = pickle.load(f)
                        if not is_blank_meta(existing_meta.get("global_metadata")):
                            logger.info(f"Skipping {path} as metadata already exists.")
                            continue
                        logger.info(f"Regenerating empty metadata for {path}.")
                    except Exception as e:
                        logger.warning(
                            f"Failed to inspect existing metadata for {path}: {e}. Regenerating."
                        )

            # use PIL, to be consistent with evaluation
            try:
                img = read_image(path, format="BGR")
            except OSError as e:
                logger.error(f"Error reading {path}: {e}")
                append_error_log(
                    args.output,
                    {
                        "type": "segmentation",
                        "path": path,
                        "error": f"read_error: {e}",
                    },
                )
                continue
            except Exception as e:
                logger.error(
                    f"{type(e).__name__}: {e} while processing {path}. Skipping."
                )
                append_error_log(
                    args.output,
                    {
                        "type": "segmentation",
                        "path": path,
                        "error": f"read_error: {e}",
                    },
                )
                continue
            start_time = time.time()
            try:
                predictions, visualized_output = demo.run_on_image(img)
            except Exception as e:
                logger.error(f"Segmentation failed for {path}: {e}")
                append_error_log(
                    args.output,
                    {
                        "type": "segmentation",
                        "path": path,
                        "error": f"run_error: {e}",
                    },
                )
                continue
            # mask = predictions["sem_seg"].argmax(dim=0).cpu().detach().numpy()
            # seg_image = label_to_color_image(mask).astype(np.uint8)
            # cv2.imshow('img', seg_image)
            # cv2.waitKey()
            # print(seg_image.shape)
            # exit()
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    (
                        "detected {} instances".format(len(predictions["instances"]))
                        if "instances" in predictions
                        else "finished"
                    ),
                    time.time() - start_time,
                )
            )

            if args.output:
                rel_dir = os.path.relpath(os.path.dirname(path), args.input)
                out_dir = (
                    os.path.join(args.output, "images", rel_dir)
                    if rel_dir != "."
                    else os.path.join(args.output, "images")
                )
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                out_filename = os.path.join(out_dir, os.path.basename(path))

                # 작업을 큐에 넣어 백그라운드 스레드에서 VLM 추론과 저장/임베딩을 분리 수행
                q_qwen.put(
                    (
                        img,
                        predictions,
                        path,
                        rel_dir,
                        visualized_output,
                        out_filename,
                        args.output,
                    )
                )
                # cv2.imwrite(out_filename, seg_image[:, :, ::-1])
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit

        if args.output:
            logger.info("남은 작업들을 큐에서 모두 처리할 때까지 기다립니다...")
            q_qwen.put(None)
            q_qwen.join()
            q_save.join()
            if qwen_thread is not None:
                qwen_thread.join()
            if save_thread is not None:
                save_thread.join()
            logger.info("모든 백그라운드 작업 완료!")

    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)
        codec, file_ext = (
            ("x264", ".mkv")
            if test_opencv_video_format("x264", ".mkv")
            else ("mp4v", ".mp4")
        )
        if codec == ".mp4v":
            warnings.warn("x264 codec not available, switching to mp4v")
        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + file_ext
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*codec),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
