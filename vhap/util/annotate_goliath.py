# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#


from tqdm import tqdm
import copy
import argparse
import torch
import math
import cv2
import numpy as np
import dlib

from tqdm import tqdm
from torch.utils.data import DataLoader
from vhap.config.base import import_module
from vhap.data.goliath import GoliathHeadDataset, collate_fn, worker_init_fn
from torchvision.utils import save_image
import logging

from star.lib import utility
from star.asset import predictor_path, model_path

from vhap.util.log import get_logger
logger = get_logger(__name__)


class GetCropMatrix():
    """
    from_shape -> transform_matrix
    """

    def __init__(self, image_size, target_face_scale, align_corners=False):
        self.image_size = image_size
        self.target_face_scale = target_face_scale
        self.align_corners = align_corners

    def _compose_rotate_and_scale(self, angle, scale, shift_xy, from_center, to_center):
        cosv = math.cos(angle)
        sinv = math.sin(angle)

        fx, fy = from_center
        tx, ty = to_center

        acos = scale * cosv
        asin = scale * sinv

        a0 = acos
        a1 = -asin
        a2 = tx - acos * fx + asin * fy + shift_xy[0]

        b0 = asin
        b1 = acos
        b2 = ty - asin * fx - acos * fy + shift_xy[1]

        rot_scale_m = np.array([
            [a0, a1, a2],
            [b0, b1, b2],
            [0.0, 0.0, 1.0]
        ], np.float32)
        return rot_scale_m

    def process(self, scale, center_w, center_h):
        if self.align_corners:
            to_w, to_h = self.image_size - 1, self.image_size - 1
        else:
            to_w, to_h = self.image_size, self.image_size

        rot_mu = 0
        scale_mu = self.image_size / (scale * self.target_face_scale * 200.0)
        shift_xy_mu = (0, 0)
        matrix = self._compose_rotate_and_scale(
            rot_mu, scale_mu, shift_xy_mu,
            from_center=[center_w, center_h],
            to_center=[to_w / 2.0, to_h / 2.0])
        return matrix


class TransformPerspective():
    """
    image, matrix3x3 -> transformed_image
    """

    def __init__(self, image_size):
        self.image_size = image_size

    def process(self, image, matrix):
        return cv2.warpPerspective(
            image, matrix, dsize=(self.image_size, self.image_size),
            flags=cv2.INTER_LINEAR, borderValue=0)


class TransformPoints2D():
    """
    points (nx2), matrix (3x3) -> points (nx2)
    """

    def process(self, srcPoints, matrix):
        # nx3
        desPoints = np.concatenate([srcPoints, np.ones_like(srcPoints[:, [0]])], axis=1)
        desPoints = desPoints @ np.transpose(matrix)  # nx3
        desPoints = desPoints[:, :2] / desPoints[:, [2, 2]]
        return desPoints.astype(srcPoints.dtype)


class Alignment:
    def __init__(self, args, model_path, dl_framework, device_ids):
        self.input_size = 256
        self.target_face_scale = 1.0
        self.dl_framework = dl_framework

        # model
        if self.dl_framework == "pytorch":
            # conf
            self.config = utility.get_config(args)
            self.config.device_id = device_ids[0]
            # set environment
            utility.set_environment(self.config)
            self.config.init_instance()
            if self.config.logger is not None:
                self.config.logger.info("Loaded configure file %s: %s" % (args.config_name, self.config.id))
                self.config.logger.info("\n" + "\n".join(["%s: %s" % item for item in self.config.__dict__.items()]))

            net = utility.get_net(self.config)
            if device_ids == [-1]:
                checkpoint = torch.load(model_path, map_location="cpu")
            else:
                checkpoint = torch.load(model_path)
            net.load_state_dict(checkpoint["net"])
            net = net.to(self.config.device_id)
            net.eval()
            self.alignment = net
        else:
            assert False

        self.getCropMatrix = GetCropMatrix(image_size=self.input_size, target_face_scale=self.target_face_scale,
                                           align_corners=True)
        self.transformPerspective = TransformPerspective(image_size=self.input_size)
        self.transformPoints2D = TransformPoints2D()

    def norm_points(self, points, align_corners=False):
        if align_corners:
            # [0, SIZE-1] -> [-1, +1]
            return points / torch.tensor([self.input_size - 1, self.input_size - 1]).to(points).view(1, 1, 2) * 2 - 1
        else:
            # [-0.5, SIZE-0.5] -> [-1, +1]
            return (points * 2 + 1) / torch.tensor([self.input_size, self.input_size]).to(points).view(1, 1, 2) - 1

    def denorm_points(self, points, align_corners=False):
        if align_corners:
            # [-1, +1] -> [0, SIZE-1]
            return (points + 1) / 2 * torch.tensor([self.input_size - 1, self.input_size - 1]).to(points).view(1, 1, 2)
        else:
            # [-1, +1] -> [-0.5, SIZE-0.5]
            return ((points + 1) * torch.tensor([self.input_size, self.input_size]).to(points).view(1, 1, 2) - 1) / 2

    def preprocess(self, image, scale, center_w, center_h):
        matrix = self.getCropMatrix.process(scale, center_w, center_h)
        input_tensor = self.transformPerspective.process(image, matrix)
        input_tensor = input_tensor[np.newaxis, :]

        input_tensor = torch.from_numpy(input_tensor)
        input_tensor = input_tensor.float().permute(0, 3, 1, 2)
        input_tensor = input_tensor / 255.0 * 2.0 - 1.0
        input_tensor = input_tensor.to(self.config.device_id)
        return input_tensor, matrix

    def postprocess(self, srcPoints, coeff):
        # dstPoints = self.transformPoints2D.process(srcPoints, coeff)
        # matrix^(-1) * src = dst
        # src = matrix * dst
        dstPoints = np.zeros(srcPoints.shape, dtype=np.float32)
        for i in range(srcPoints.shape[0]):
            dstPoints[i][0] = coeff[0][0] * srcPoints[i][0] + coeff[0][1] * srcPoints[i][1] + coeff[0][2]
            dstPoints[i][1] = coeff[1][0] * srcPoints[i][0] + coeff[1][1] * srcPoints[i][1] + coeff[1][2]
        return dstPoints

    def analyze(self, image, scale, center_w, center_h):
        input_tensor, matrix = self.preprocess(image, scale, center_w, center_h)

        if self.dl_framework == "pytorch":
            with torch.no_grad():
                output = self.alignment(input_tensor)
            landmarks = output[-1][0]
        else:
            assert False

        landmarks = self.denorm_points(landmarks)
        landmarks = landmarks.data.cpu().numpy()[0]
        landmarks = self.postprocess(landmarks, np.linalg.inv(matrix))

        return landmarks


def draw_pts(img, pts, mode="pts", shift=4, color=(0, 255, 0), radius=10, thickness=10, save_path=None, dif=0,
             scale=0.3, concat=False, ):
    img_draw = copy.deepcopy(img)
    img_draw = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)
    print(img_draw.shape)
    for cnt, p in enumerate(pts):
        if mode == "index":
            cv2.putText(img_draw, str(cnt), (int(float(p[0] + dif)), int(float(p[1] + dif))), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, color, thickness)
        elif mode == 'pts':
            if len(img_draw.shape) > 2:
                # 此处来回切换是因为opencv的bug
                # img_draw = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
                pass
            cv2.circle(img_draw, (int(p[0] * img_draw.shape[1]), int(p[1] * img_draw.shape[0])), radius, color, thickness=thickness)
        else:
            raise NotImplementedError
    if concat:
        img_draw = np.concatenate((img, img_draw), axis=1)
    if save_path is not None:
        cv2.imwrite(save_path, img_draw)
    return img_draw


class LandmarkDetectorSTAR:
    def __init__(
        self,
    ):
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(predictor_path)

        # facial landmark detector
        args = argparse.Namespace()
        args.config_name = 'alignment'
        # could be downloaded here: https://drive.google.com/file/d/1aOx0wYEZUfBndYy_8IYszLPG_D2fhxrT/view
        # model_path = '/path/to/WFLW_STARLoss_NME_4_02_FR_2_32_AUC_0_605.pkl'
        device_ids = '0'
        device_ids = list(map(int, device_ids.split(",")))
        self.alignment = Alignment(args, model_path, dl_framework="pytorch", device_ids=device_ids)

    def detect_single_image(self, img):
        bbox = self.detector(img, 1)

        if len(bbox) == 0:
            bbox = np.zeros(5) - 1
            lmks = np.zeros([68, 3]) - 1  # set to -1 when landmarks is inavailable
        else:
            face = self.shape_predictor(img, bbox[0])
            shape = []
            for i in range(68):
                x = face.part(i).x
                y = face.part(i).y
                shape.append((x, y))
            shape = np.array(shape)
            x1, x2 = shape[:, 0].min(), shape[:, 0].max()
            y1, y2 = shape[:, 1].min(), shape[:, 1].max()
            scale = min(x2 - x1, y2 - y1) / 200 * 1.05
            center_w = (x2 + x1) / 2
            center_h = (y2 + y1) / 2

            scale, center_w, center_h = float(scale), float(center_w), float(center_h)
            lmks = self.alignment.analyze(img, scale, center_w, center_h)

            h, w = img.shape[:2]

            lmks = np.concatenate([lmks, np.ones([lmks.shape[0], 1])], axis=1).astype(np.float32)  # (x, y, 1)
            lmks[:, 0] /= w
            lmks[:, 1] /= h

            bbox = np.array([bbox[0].left(), bbox[0].top(), bbox[0].right(), bbox[0].bottom(), 1.]).astype(np.float32)  # (x1, y1, x2, y2, score)
            bbox[[0, 2]] /= w
            bbox[[1, 3]] /= h

        return bbox, lmks

    def detect_dataset(self, dataloader):
        """
        Annotates each frame with 68 facial landmarks
        :return: dict mapping frame number to landmarks numpy array and the same thing for bboxes
        """
        logger.info("Initialize Landmark Detector (STAR)...")
        # 68 facial landmark detector

        landmarks = {}
        bboxes = {}

        logger.info("Begin annotating landmarks...")
        for item in tqdm(dataloader):
            timestep_id = item["timestep_id"][0]
            camera_id = item["camera_id"][0]

            logger.info(
                f"Annotate facial landmarks for timestep: {timestep_id}, camera: {camera_id}"
            )
            img = item["rgb"][0].numpy()

            bbox, lmks = self.detect_single_image(img)
            if len(bbox) == 0:
                logger.error(
                    f"No bbox found for frame: {timestep_id}, camera: {camera_id}. Setting landmarks to all -1."
                )

            if camera_id not in landmarks:
                landmarks[camera_id] = {}
            if camera_id not in bboxes:
                bboxes[camera_id] = {}
            landmarks[camera_id][timestep_id] = lmks
            bboxes[camera_id][timestep_id] = bbox
        return landmarks, bboxes

    def annotate_landmarks(self, dataloader):
        """
        Annotates each frame with landmarks for face and iris. Assumes frames have been extracted
        :return:
        """
        lmks_face, bboxes_faces = self.detect_dataset(dataloader)

        # construct final json
        for camera_id, lmk_face_camera in lmks_face.items():
            bounding_box = []
            face_landmark_2d = []
            for timestep_id in lmk_face_camera.keys():
                bounding_box.append(bboxes_faces[camera_id][timestep_id][None])
                face_landmark_2d.append(lmks_face[camera_id][timestep_id][None])

            lmk_dict = {
                "bounding_box": bounding_box,
                "face_landmark_2d": face_landmark_2d,
            }

            for k, v in lmk_dict.items():
                if len(v) > 0:
                    lmk_dict[k] = np.concatenate(v, axis=0)
            out_path = dataloader.dataset.get_property_path(
                "landmark2d/STAR", camera_id=camera_id
            )
            logger.info(f"Saving landmarks to: {out_path}")
            if not out_path.parent.exists():
                out_path.parent.mkdir(parents=True)
            np.savez(out_path, **lmk_dict)


def main(chunk_id: int, n_chunks: int):
    logger = get_logger(__name__, root=True, level=logging.WARN)

    cam_subset_front = [
        "401643",
        "401650",
        "401653",
        "401659",
        "401892",
        "401949",
        "401955",
        "401957",
        "401961",
        "401962",
        "401964",
        "402597",
        "402598",
        "402601",
        "402792",
        "402800",
        "402803",
        "402805",
        "402807",
        "402862",
        "402866",
        "402871",
        "402875",
        "402878",
        "402879",
        "402957",
        "402959",
        "402965",
        "402966",
        "402967",
        "402968",
        "402979",
        "402983",
        "403072",
        "403073",
        "403077",
        "403078",
    ]

    chunk_size = math.ceil(len(cam_subset_front) / n_chunks)
    cam_subset_chunk = [cam_subset_front[i:i + chunk_size] for i in range(0, len(cam_subset_front), chunk_size)][chunk_id]

    dataset = GoliathHeadDataset(
        root_path="/cluster/pegasus/jschmidt/goliath/m--20230306--0707--AXE977--pilot--ProjectGoliath--Head",
        shared_assets_path="/cluster/pegasus/jschmidt/goliath/shared/static_assets_head.pt",
        split=None,
        fully_lit_only=True,
        cameras_subset=cam_subset_chunk,
        # frames_subset=[2858, 2888, 2918],
        downsample_factor=2.
    )

    print(f"Num cameras: {len(dataset.camera_ids)} / {len(cam_subset_front)}")
    print(f"Num frames: {len(dataset.frame_list)}")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn, worker_init_fn=worker_init_fn)

    star = LandmarkDetectorSTAR()

    for idx, batch in enumerate(tqdm(dataloader)):
        if batch is None:
            print("Empty batch")
        
        frame_id = batch["frame_id"][0]
        cam_id = batch["camera_id"][0]
        image = batch["image"].squeeze(0).clamp(0, 1).permute(1, 2, 0).numpy() * 255
        image = image.astype(np.uint8)

        bbox, lmks = star.detect_single_image(image)

        # draw_pts(image, lmks, save_path="./lmks.png")

        save_path = dataset.root_path / "landmark2d" / "STAR" / f"cam{cam_id}" / f"{frame_id:06d}.npz"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(save_path, bbox=bbox, lmks=lmks)

if __name__ == "__main__":
    import tyro
    
    tyro.cli(main)
    # if idx % 1000 == 0:
    #     print(f"Processed {idx} / {len(dataset)} frames")