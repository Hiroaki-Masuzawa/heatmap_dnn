#!
import os
import time
import numpy as np
import yaml
import random
import argparse
from setuptools._distutils.util import strtobool
from scipy.spatial.transform import Rotation

import cv2

import torch


def get_localmax(pred_sig, kernelsize, th=0.3):
    """Get localmax"""
    padding_size = (kernelsize - 1) // 2
    max_v = torch.nn.functional.max_pool2d(
        pred_sig, kernelsize, stride=1, padding=padding_size
    )
    ret = pred_sig.clone()
    ret[pred_sig != max_v] = 0
    ret[ret < th] = 0
    ret[ret != 0] = 1
    return ret


# 頂点を法線に対して右回りになるように並べ替える
def sort_vertex(model_vertex, plane_vector, plane_normal):
    """Get sorted vertex"""
    mean_vertex = np.mean(model_vertex, axis=0)
    tmp_vec = model_vertex - mean_vertex
    tmp_vec = tmp_vec / np.linalg.norm(tmp_vec, axis=1, keepdims=True)
    point_args = np.arctan2(
        np.dot(tmp_vec, plane_vector),
        np.dot(np.cross(tmp_vec, plane_vector), plane_normal),
    )
    sorted_idxs = np.argsort(point_args)
    model_vertex = model_vertex[sorted_idxs]
    return model_vertex


def gen_fitting_model(blockdata, issort=True):
    """Generrate fitting model from config"""
    ret = {}
    eps = 1e-3
    
    for block in blockdata:
        # print(block['name'])
        convex_vertex_pos = np.array(
            block["convex_vertex_pos"]).reshape((-1, 3))
        concave_vertex_pos = np.array(
            block["concave_vertex_pos"]).reshape((-1, 3))

        model_vertexs = []
        z_max = np.max(convex_vertex_pos[:, 2])
        plane_normal = np.array([0, 0, 1])
        plane_vector = np.array([1, 0, 0])
        target_ids = np.where(convex_vertex_pos[:, 2] > (z_max - eps))[0]
        model_vertex = convex_vertex_pos[target_ids]
        if issort:
            model_vertexs.append(sort_vertex(
                model_vertex, plane_vector, plane_normal))
        else:
            model_vertexs.append(model_vertex)

        z_min = np.min(convex_vertex_pos[:, 2])
        plane_normal = np.array([0, 0, -1])
        plane_vector = np.array([1, 0, 0])
        target_ids = np.where(convex_vertex_pos[:, 2] < (z_min + eps))[0]
        model_vertex = convex_vertex_pos[target_ids]
        if issort:
            model_vertexs.append(sort_vertex(
                model_vertex, plane_vector, plane_normal))
        else:
            model_vertexs.append(model_vertex)

        x_max = np.max(convex_vertex_pos[:, 0])
        plane_normal = np.array([1, 0, 0])
        plane_vector = np.array([0, 1, 0])
        target_ids = np.where(convex_vertex_pos[:, 0] > (x_max - eps))[0]
        model_vertex = convex_vertex_pos[target_ids]
        if issort:
            model_vertexs.append(sort_vertex(
                model_vertex, plane_vector, plane_normal))
        else:
            model_vertexs.append(model_vertex)

        x_min = np.min(convex_vertex_pos[:, 0])
        plane_normal = np.array([-1, 0, 0])
        plane_vector = np.array([0, 1, 0])
        target_ids = np.where(convex_vertex_pos[:, 0] < (x_min + eps))[0]
        model_vertex = convex_vertex_pos[target_ids]
        if issort:
            model_vertexs.append(sort_vertex(
                model_vertex, plane_vector, plane_normal))
        else:
            model_vertexs.append(model_vertex)

        y_max = np.max(convex_vertex_pos[:, 1])
        plane_normal = np.array([0, 1, 0])
        plane_vector = np.array([0, 0, 1])
        target_ids = np.where(convex_vertex_pos[:, 1] > (y_max - eps))[0]
        model_vertex = convex_vertex_pos[target_ids]
        if issort:
            model_vertexs.append(sort_vertex(
                model_vertex, plane_vector, plane_normal))
        else:
            model_vertexs.append(model_vertex)

        y_min = np.min(convex_vertex_pos[:, 1])
        plane_normal = np.array([0, -1, 0])
        plane_vector = np.array([0, 0, 1])
        target_ids = np.where(convex_vertex_pos[:, 1] < (y_min + eps))[0]
        model_vertex = convex_vertex_pos[target_ids]
        if issort:
            model_vertexs.append(sort_vertex(
                model_vertex, plane_vector, plane_normal))
        else:
            model_vertexs.append(model_vertex)

        # print(model_vertexs)
        # bounding_point = [x_min, x_max, y_min, y_max, z_min, z_max]
        bounding_point = np.array(
            [
                [x_min, y_min, z_min],
                [x_max, y_min, z_min],
                [x_max, y_max, z_min],
                [x_min, y_max, z_min],
                [x_min, y_min, z_max],
                [x_max, y_min, z_max],
                [x_max, y_max, z_max],
                [x_min, y_max, z_max],
            ]
        )
        ret[block["name"]] = {
            "model_vertexs": model_vertexs,
            "convex_vertex_pos": convex_vertex_pos,
            "concave_vertex_pos": concave_vertex_pos,
            "bounding_point": bounding_point,
        }
    return ret


def gen_featrue_map(image, model, device, score_ths=[0.5, 0.3]):
    """Generate feature map"""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    image_rgb = np.transpose(image_rgb, (2, 0, 1))[np.newaxis]
    image_tensor = torch.Tensor(image_rgb)
    pred = (model(image_tensor.to(device)).sigmoid()).detach()
    pred_convex = pred[:, 0:1].clone()
    pred_concave = pred[:, 1:2].clone()

    for score_th in score_ths:
        pred_convex_lm = get_localmax(pred_convex, 13, score_th)
        pred_concave_lm = get_localmax(pred_concave, 13, score_th)
        # pred_convex_np = (pred_convex.cpu()*255).numpy().astype(np.uint8)[0,0]
        # pred_concave_np = (pred_concave.cpu()*255).numpy().astype(np.uint8)[0,0]
        convex_pos = np.array(
            np.where(pred_convex_lm[0, 0].cpu().numpy() != 0)).T[:, ::-1]
        concave_pos = np.array(
            np.where(pred_concave_lm[0, 0].cpu().numpy() != 0)).T[:, ::-1]
        # TODO 画面周囲には特徴点がないと仮定するとマッチングがよくなるので実施しているがこの過程良いのか？
        convex_pos = convex_pos[convex_pos[:, 0] != 0]
        convex_pos = convex_pos[convex_pos[:, 0] != 511]
        convex_pos = convex_pos[convex_pos[:, 1] != 0]
        convex_pos = convex_pos[convex_pos[:, 1] != 511]
        if len(convex_pos) > 5:
            break
    return (
        pred_convex,
        pred_concave,
        pred_convex_lm,
        pred_concave_lm,
        convex_pos,
        concave_pos,
    )


# モデルに対してfittingをおこなう
def fit_model(target_model, convex_pos, pred_convex, cam_mat, dist, itr_num=2000, issort=True):
    """Fitting Model"""
    model_vertexs = target_model["model_vertexs"]
    convex_vertex_pos = target_model["convex_vertex_pos"]
    # concave_vertex_pos = target_model['concave_vertex_pos']

    min_score = 1e-2
    min_logscore = np.log(min_score)
    max_score = -1e100
    max_score_dat = {}

    # start_time = time.time()
    score_map = pred_convex.cpu()[0, 0].numpy()
    convex_pos_list = convex_pos.tolist()
    for i in range(itr_num):
        # 当てはめる面のサンプル
        model_vertex = random.sample(model_vertexs, 1)[0]
        # 点のサンプル
        sample_dat = random.sample(convex_pos_list, len(model_vertex))
        if issort:
            # サンプリングした点を-zの単位ベクトルに対して右回りに並べなおす
            img_pos = np.array(sample_dat).astype(np.float64)
            rel_img_pos = img_pos - np.mean(img_pos, axis=0)
            rel_img_pos = rel_img_pos / \
                np.linalg.norm(rel_img_pos, axis=1, keepdims=True)
            cost = np.dot(rel_img_pos, np.array([1, 0]))
            sint = np.dot(
                np.cross(
                    np.hstack((rel_img_pos, np.zeros((len(model_vertex), 1)))),
                    np.array([1, 0, 0]),
                ),
                np.array([0, 0, -1]),
            )
            t = np.arctan2(cost, sint)
            sort_idx = np.argsort(t)
            img_pos = img_pos[sort_idx]
        else :
            img_pos = np.array(sample_dat).astype(np.float64)

        for j in range(len(img_pos)):
            # 順番を変えながらフィッティングを行う
            if j != 0:
                img_posd = np.array(
                    img_pos.tolist()[j:] + img_pos.tolist()[:j])
            else:
                img_posd = img_pos
            # PnPで並進，回転ベクトルを求める
            ret, rvec, tvec = cv2.solvePnP(
                model_vertex, img_posd, cam_mat, dist)
            # モデルの頂点の投影位置を求める
            point, _ = cv2.projectPoints(
                convex_vertex_pos, rvec, tvec, cam_mat, dist)

            # モデルの投影位置のヒートマップ値の積を求めたいが，
            # アンダーフローする可能性があるので，ヒートマップの積ではなくlogの和を取る
            point_int = point.astype(np.int64).reshape((-1, 2))
            score_list = []
            for p in point_int:
                if p[0] >= 0 and p[1] >= 0 and p[0] < 512 and p[1] < 512:
                    value = float(score_map[p[1], p[0]])
                    if value >= min_score:
                        score_list.append(np.log(value))
                    else:
                        score_list.append(min_logscore)
                else:
                    score_list.append(min_logscore)
            # 出現した中で最大のものを保存する
            score_sum = np.sum(score_list)
            if score_sum > max_score:
                max_score = score_sum
                max_score_dat = {
                    "rvec": rvec,
                    "tvec": tvec,
                    "point": point.reshape((-1, 2)),
                }
    return max_score, max_score_dat


def quat2rvec(quat):
    """クオータニオンをrotation vectorに変換する"""
    theta = 2 * np.arccos(quat[3])
    vec = np.array(quat[:3])
    vec = vec / np.linalg.norm(vec)
    return theta * vec


def get_Pscore(Pmat):
    diff_trans = np.linalg.norm(Pmat[0:3, 3])
    diff_rot = np.linalg.norm(Rotation.from_matrix(Pmat[0:3, 0:3]).as_rotvec())
    return diff_trans, diff_rot


def main():
    parser = argparse.ArgumentParser(
        prog="",  # プログラム名
        usage="",  # プログラムの利用方法
        add_help=True,  # -h/–help オプションの追加
    )
    parser.add_argument("--dnnmodel", type=str, default="")
    parser.add_argument("--input", type=str, default="")
    parser.add_argument("--blockmodel", type=str, default="yellow_block")
    parser.add_argument(
        "--annotatefile", type=str, default="/dataset/puzzle_block/test/annotation.yaml"
    )
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--device", type=str,
                        default="cuda", choices=["cpu", "cuda"])
    parser.add_argument('--score_th', type=float,
                        nargs='*', default=[0.5, 0.3])
    parser.add_argument('--itr_num', type=int, default=2000)
    parser.add_argument('--issort', type=strtobool, default=1)
    args = parser.parse_args()

    device = args.device
    issort = args.issort == 1

    with open(args.annotatefile, "r", encoding="utf-8") as f:
        anno_dat = yaml.safe_load(f)

    model = torch.load(args.dnnmodel)
    model.to(device)
    model.eval()

    fitting_model = gen_fitting_model(anno_dat["blocks"], issort=issort)
    lines = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )

    if args.input != "":
        cam_mat = np.array(anno_dat["annotations"]
                           [0]["camera_matrix"]).reshape(3, 3)  # TODO
        dist = np.zeros((5))
        image = cv2.imread(args.input)
        (
            pred_convex,
            pred_concave,
            pred_convex_lm,
            pred_concave_lm,
            convex_pos,
            concave_pos,
        ) = gen_featrue_map(image, model, device, score_ths=args.score_th)

        max_score, max_score_dat = fit_model(
            fitting_model[args.blockmodel], convex_pos, pred_convex, cam_mat, dist, itr_num=args.itr_num, issort=issort
        )

        result = image.copy()
        bounding_point = fitting_model[args.blockmodel]["bounding_point"]
        bounding_img_point, _ = cv2.projectPoints(
            bounding_point, max_score_dat["rvec"], max_score_dat["tvec"], cam_mat, dist
        )
        bounding_img_point = bounding_img_point.reshape(
            (-1, 2)).astype(np.int64)
        for line in lines:
            result = cv2.line(
                result,
                tuple(bounding_img_point[line[0]]),
                tuple(bounding_img_point[line[1]]),
                (255, 255, 255),
                2,
            )
        output_image = cv2.hconcat(
            [image,
             cv2.cvtColor(
                 (pred_convex.cpu() * 255).numpy().astype(np.uint8)[0, 0], cv2.COLOR_GRAY2BGR),
             cv2.cvtColor(
                 (pred_concave.cpu() * 255).numpy().astype(np.uint8)[0, 0], cv2.COLOR_GRAY2BGR),
             cv2.cvtColor(
                 (pred_convex_lm.cpu() * 255).numpy().astype(np.uint8)[0, 0], cv2.COLOR_GRAY2BGR),
             result,
             ]
        )
        if args.output != '':
            cv2.imwrite(args.output, output_image)
    else:
        # 評価用コード
        for anno in anno_dat["annotations"]:
            dist = np.zeros((5))
            image_file = os.path.join(os.path.dirname(
                args.annotatefile), anno["imagefile"])
            cam_mat = np.array(anno["camera_matrix"]).reshape(3, 3)
            image = cv2.imread(image_file)
            blockmodel = anno["block_name"]
            (
                pred_convex,
                pred_concave,
                pred_convex_lm,
                pred_concave_lm,
                convex_pos,
                concave_pos,
            ) = gen_featrue_map(image, model, device, score_ths=args.score_th)
            max_score, max_score_dat = fit_model(
                fitting_model[blockmodel], convex_pos, pred_convex, cam_mat, dist, itr_num=args.itr_num, issort=issort)
            result = image.copy()
            bounding_point = fitting_model[blockmodel]["bounding_point"]
            bounding_img_point, _ = cv2.projectPoints(
                bounding_point,
                max_score_dat["rvec"],
                max_score_dat["tvec"],
                cam_mat,
                dist,
            )
            bounding_img_point = bounding_img_point.reshape(
                (-1, 2)).astype(np.int64)
            for line in lines:
                result = cv2.line(
                    result,
                    tuple(bounding_img_point[line[0]]),
                    tuple(bounding_img_point[line[1]]),
                    (255, 255, 255),
                    2,
                )
            output_image = cv2.hconcat(
                [image,
                 cv2.cvtColor(
                     (pred_convex.cpu() * 255).numpy().astype(np.uint8)[0, 0], cv2.COLOR_GRAY2BGR,),
                 cv2.cvtColor(
                     (pred_concave.cpu() * 255).numpy().astype(np.uint8)[0, 0], cv2.COLOR_GRAY2BGR,),
                 result,
                 ]
            )
            if args.output != '':
                cv2.imwrite(args.output, output_image)
            else:
                os.makedirs('result', exist_ok=True)
                output_file = os.path.basename(
                    image_file).replace('image', 'result')
                cv2.imwrite(os.path.join('result', output_file), output_image)

            camera_rvec = quat2rvec(anno["camera_orientation"])
            R1, _ = cv2.Rodrigues(camera_rvec)
            Proj1 = np.zeros((4, 4))
            Proj1[0:3, 0:3] = R1
            Proj1[0:3, 3] = anno["camera_position"]
            Proj1[3, 3] = 1.0

            estimate_rvec = max_score_dat["rvec"].reshape(-1)
            R2, _ = cv2.Rodrigues(estimate_rvec)
            Proj2 = np.zeros((4, 4))
            Proj2[0:3, 0:3] = R2
            Proj2[0:3, 3] = max_score_dat["tvec"].reshape(-1)
            Proj2[3, 3] = 1.0
            Proj = Proj2 @ Proj1
            diff_trans, diff_rot = get_Pscore(Proj)
            if blockmodel == "cyan_block":
                alt_proj = Proj2 @ np.array([[-1, 0, 0, 0], [0, -1, 0, 0],
                                            [0, 0, 1, 0], [0, 0, 0, 1]]) @ Proj1
                alt_diff_trans, alt_diff_rot = get_Pscore(alt_proj)
                if alt_diff_rot < diff_rot and alt_diff_trans < diff_trans:
                    diff_rot = alt_diff_rot
                    diff_trans = alt_diff_trans
            elif blockmodel == "red_block":
                alt_proj = Proj2 @ np.array([[1, 0, 0, 0], [0, -1, 0, 0],
                                            [0, 0, -1, 0.03], [0, 0, 0, 1]]) @ Proj1
                alt_diff_trans, alt_diff_rot = get_Pscore(alt_proj)
                if alt_diff_rot < diff_rot and alt_diff_trans < diff_trans:
                    diff_rot = alt_diff_rot
                    diff_trans = alt_diff_trans
            elif blockmodel == "green_block":
                alt_proj1 = Proj2 @ np.array([[0, 0, -1, 0.03], [1, 0, 0, 0], [
                                             0, -1, 0, 0.03], [0, 0, 0, 1]]) @ Proj1
                alt_proj2 = Proj2 @ np.array(
                    [[0, 1, 0, 0], [0, 0, -1, 0.03], [-1, 0, 0, 0.03], [0, 0, 0, 1]]) @ Proj1
                for alt_proj in [alt_proj1, alt_proj2]:
                    alt_diff_trans, alt_diff_rot = get_Pscore(alt_proj)
                    if alt_diff_rot < diff_rot and alt_diff_trans < diff_trans:
                        diff_rot = alt_diff_rot
                        diff_trans = alt_diff_trans
            elif blockmodel == "brown_block":
                alt_proj1 = Proj2 @ np.array(
                    [[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0.03], [0, 0, 0, 1]]) @ Proj1
                alt_proj2 = Proj2 @ np.array([[-1, 0, 0, 0.03], [0, 1, 0, 0], [
                                             0, 0, -1, 0.03], [0, 0, 0, 1]]) @ Proj1
                alt_proj3 = Proj2 @ np.array([[0, 0, -1, 0.03],
                                             [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]) @ Proj1
                alt_proj4 = Proj2 @ np.array(
                    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0.03], [0, 0, 0, 1]]) @ Proj1
                alt_proj5 = Proj2 @ np.array(
                    [[0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]) @ Proj1
                alt_proj6 = Proj2 @ np.array([[-1, 0, 0, 0.03], [0, -1, 0, 0], [
                                             0, 0, 1, 0], [0, 0, 0, 1]]) @ Proj1
                alt_proj7 = Proj2 @ np.array(
                    [[0, 0, -1, 0.03], [0, -1, 0, 0], [-1, 0, 0, 0.03], [0, 0, 0, 1]]) @ Proj1
                for alt_proj in [alt_proj1, alt_proj2, alt_proj3, alt_proj4, alt_proj5, alt_proj6, alt_proj7]:
                    alt_diff_trans, alt_diff_rot = get_Pscore(alt_proj)
                    if alt_diff_rot < diff_rot and alt_diff_trans < diff_trans:
                        diff_rot = alt_diff_rot
                        diff_trans = alt_diff_trans
            elif blockmodel == "purple_block":
                alt_proj = Proj2 @ np.array([[-1, 0, 0, 0.03], [0, 0, 1, -0.03], [
                                            0, 1, 0, 0.03], [0, 0, 0, 1]]) @ Proj1
                alt_diff_trans, alt_diff_rot = get_Pscore(alt_proj)
                if alt_diff_rot < diff_rot and alt_diff_trans < diff_trans:
                    diff_rot = alt_diff_rot
                    diff_trans = alt_diff_trans
            elif blockmodel == "lightgreen_block":
                alt_proj = Proj2 @ np.array([[-1, 0, 0, 0], [0, 0, 1, -0.03], [
                                            0, 1, 0, 0.03], [0, 0, 0, 1]]) @ Proj1
                alt_diff_trans, alt_diff_rot = get_Pscore(alt_proj)
                if alt_diff_rot < diff_rot and alt_diff_trans < diff_trans:
                    diff_rot = alt_diff_rot
                    diff_trans = alt_diff_trans

            print(image_file, max_score, diff_trans, diff_rot)

            # _ = input()


if __name__ == "__main__":
    main()
