#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-30 上午10:04
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : lanenet_postprocess.py
# @IDE: PyCharm Community Edition
"""
LaneNet model post process
"""
import os.path as ops
import sys

import math

import cv2
import glog as log
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from lanenet_model.util_get_four_pointv2 import get_final_x
# from lanenet_model.util_get_four_point import get_final_x


from config import global_config

CFG = global_config.cfg


def _morphological_process(image, kernel_size=5):
    """
    morphological process to fill the hole in the binary segmentation result
    :param image:
    :param kernel_size:
    :return:
    """
    if len(image.shape) == 3:
        raise ValueError('Binary segmentation result image should be a single channel image')

    if image.dtype is not np.uint8:
        image = np.array(image, np.uint8)

    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(kernel_size, kernel_size))

    # close operation fille hole
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)

    return closing


def _connect_components_analysis(image):
    """
    connect components analysis to remove the small components 连接组件分析以删除小组件
    :param image:
    :return:
    """
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    #  cv2.connectedComponentsWithStats
    # 该函数输入为一个二值化图像，输出为一个长为4的tuple,第一个是连通区域的个数，第二个是一整张图的label，
    # 第三个是(x, y, width, height, area)，即每个区域的每个区域的左上角坐标,宽和高，面积。第四个是每个连通区域的中心点
    return cv2.connectedComponentsWithStats(gray_image, connectivity=8, ltype=cv2.CV_32S)


class _LaneFeat(object):
    """

    """
    def __init__(self, feat, coord, class_id=-1):
        """
        lane feat object
        :param feat: lane embeddng feats [feature_1, feature_2, ...]
        :param coord: lane coordinates [x, y]
        :param class_id: lane class id
        """
        self._feat = feat
        self._coord = coord
        self._class_id = class_id

    @property
    def feat(self):
        """

        :return:
        """
        return self._feat

    @feat.setter
    def feat(self, value):
        """

        :param value:
        :return:
        """
        if not isinstance(value, np.ndarray):
            value = np.array(value, dtype=np.float64)

        if value.dtype != np.float32:
            value = np.array(value, dtype=np.float64)

        self._feat = value

    @property
    def coord(self):
        """

        :return:
        """
        return self._coord

    @coord.setter
    def coord(self, value):
        """

        :param value:
        :return:
        """
        if not isinstance(value, np.ndarray):
            value = np.array(value)

        if value.dtype != np.int32:
            value = np.array(value, dtype=np.int32)

        self._coord = value

    @property
    def class_id(self):
        """

        :return:
        """
        return self._class_id

    @class_id.setter
    def class_id(self, value):
        """

        :param value:
        :return:
        """
        if not isinstance(value, np.int64):
            raise ValueError('Class id must be integer')

        self._class_id = value


class _LaneNetCluster(object):
    """
     Instance segmentation result cluster
    """

    def __init__(self):
        """

        """
        self._color_map = [np.array([255, 0, 0]),
                           np.array([0, 255, 0]),
                           np.array([0, 0, 255]),
                           np.array([125, 125, 0]),
                           np.array([0, 125, 125]),
                           np.array([125, 0, 125]),
                           np.array([50, 100, 50]),
                           np.array([100, 50, 100])]

    @staticmethod
    def _embedding_feats_dbscan_cluster(embedding_image_feats):
        """
        dbscan cluster
        :param embedding_image_feats:
        :return:
        """
        db = DBSCAN(eps=CFG.POSTPROCESS.DBSCAN_EPS, min_samples=CFG.POSTPROCESS.DBSCAN_MIN_SAMPLES)
        try:
            features = StandardScaler().fit_transform(embedding_image_feats)
            db.fit(features)
        except Exception as err:
            log.error(err)
            ret = {
                'origin_features': None,
                'cluster_nums': 0,
                'db_labels': None,
                'unique_labels': None,
                'cluster_center': None
            }
            return ret
        # db_labels 每个像素属于哪条车道线标签
        db_labels = db.labels_
        # np.unique()该函数是去除数组中的重复数字，并进行排序之后输出
        unique_labels = np.unique(db_labels)
        # 聚类的簇的个数
        num_clusters = len(unique_labels)
        cluster_centers = db.components_

        ret = {
            'origin_features': features,
            'cluster_nums': num_clusters,
            'db_labels': db_labels,
            'unique_labels': unique_labels,
            'cluster_center': cluster_centers
        }

        return ret

    @staticmethod
    def _get_lane_embedding_feats(binary_seg_ret, instance_seg_ret):
        """
        get lane embedding features according the binary seg result
        :param binary_seg_ret:
        :param instance_seg_ret:
        :return:
        """
        # print('instance_seg_ret.shape',instance_seg_ret.shape) (256,512,4)
        idx = np.where(binary_seg_ret == 255)
        # print('idx.shape',len(idx))
        # print('idx',idx)(idx (array([152, 152, 153, ..., 255, 255, 255], dtype=int64),
        # array([258, 259, 256, ..., 420, 421, 422], dtype=int64))
        # 输出的是二值图中白色（即车道线）的坐标(152,258),(152,259)...
        lane_embedding_feats = instance_seg_ret[idx]
        # print('lane_embedding_feats',len(lane_embedding_feats))2463
        # print('lane_embedding_feats', lane_embedding_feats.shape)
        # print('instance_seg_ret[idx]',len(instance_seg_ret[idx]))
        # idx_scale = np.vstack((idx[0] / 256.0, idx[1] / 512.0)).transpose()
        # lane_embedding_feats = np.hstack((lane_embedding_feats, idx_scale))

        lane_coordinate = np.vstack((idx[1], idx[0])).transpose()

        assert lane_embedding_feats.shape[0] == lane_coordinate.shape[0]

        ret = {
            'lane_embedding_feats': lane_embedding_feats,
            'lane_coordinates': lane_coordinate
        }
        # print(ret['lane_embedding_feats'].shape)  (2463,4)
        # print(ret['lane_coordinates'].shape)  (2463,2)
        return ret
    # 获取到坐标及对应坐标像素对应的向量表示.
    #
    # np.where(condition)
    # 只有条件 (condition)，没有 x 和 y ，则输出满足条件 (即非0) 元素的坐标 (等价于numpy.nonzero)。
    # 这里的坐标以tuple的形式给出，通常原数组有多少维，输出的tuple中就包含几个数组，分别对应符合条件元素的各维坐标。

    def apply_lane_feats_cluster(self, binary_seg_result, instance_seg_result):
        """

        :param binary_seg_result:
        :param instance_seg_result:
        :return:
        """
        # get embedding feats and coords 得到特征向量对应的像素坐标

        # 得到特征向量对应的像素坐标
        get_lane_embedding_feats_result = self._get_lane_embedding_feats(
            binary_seg_ret=binary_seg_result,
            instance_seg_ret=instance_seg_result
        )
        # print(binary_seg_result.shape) (256,512)
        # print(instance_seg_result.shape) (256,512,4)
        # print(type(get_lane_embedding_feats_result)) dict
        # dbscan cluster 聚类 对特征向量进行聚类
        dbscan_cluster_result = self._embedding_feats_dbscan_cluster(
            embedding_image_feats=get_lane_embedding_feats_result['lane_embedding_feats']
        )
        # print(dbscan_cluster_result) 'cluster_nums': 6
        # print(dbscan_cluster_result['db_labels'])
        # print(dbscan_cluster_result['db_labels'].shape) (9836,1)
        mask = np.zeros(shape=[binary_seg_result.shape[0], binary_seg_result.shape[1], 3], dtype=np.uint8)
        db_labels = dbscan_cluster_result['db_labels']
        unique_labels = dbscan_cluster_result['unique_labels']
        # coord ：特征向量对应的坐标
        coord = get_lane_embedding_feats_result['lane_coordinates']
        print(type(coord))
        if db_labels is None:
            return None, None

        lane_coords = []
        # 存储的是 mask 图中的坐标
        all_line = []
        for index, label in enumerate(unique_labels.tolist()):
            # print(db_labels.shape) (9836,1)
            # print(label) -1,0,1,2,3,4
            if label == -1:
                continue
            idx = np.where(db_labels == label)
            # print(idx)
            # (array([   0,    1,    2, ..., 9829, 9830, 9832], dtype=int64),)
            pix_coord_idx = tuple((coord[idx][:, 1], coord[idx][:, 0]))
            # print('coordy',coord[idx][:,1])
            # print('coordx',coord[idx][:,0])
            all_line.append([coord[idx][:, 0], coord[idx][:, 1]])

            mask[pix_coord_idx] = self._color_map[index]
            lane_coords.append(coord[idx])

        ''' 根据曲线求四个点的坐标'''
        # 这个地方修改的值是在输出的 mask 图上进行绘制的
        x_final_left_down, \
        x_final_left_up, \
        x_final_right_down, \
        x_final_right_up = get_final_x(all_line, W=512, H=256, y_stead=150) # 根据 y 值选取感兴趣区域，y 值是先验值
        points = []
        points.append(list(x_final_left_up))
        points.append(list(x_final_right_up))
        points.append(list(x_final_right_down))
        points.append(list(x_final_left_down))
        # 输出 mask 图上的四个点
        print(type(x_final_right_up))
        print(x_final_left_down, x_final_left_up, x_final_right_down, x_final_right_up)
        # '''根据四个点，绘制区域'''
        # source_image = cv2.imread("E:/laneDetection/lane_pic/instance/instance_2.png")
        # pts = np.array([list(x_final_left_up), list(x_final_right_up), list(x_final_right_down), list(x_final_left_down)], np.int32)
        # # 顶点个数：4，矩阵变成4*1*2维
        # pts = pts.reshape((-1, 1, 2))
        # cv2.polylines(source_image, [pts], True, (0, 255, 255))
        # cv2.imshow('source_image', source_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print(mask.shape)  (256, 512, 3)
        # lane_coords 是车道线的条数，每条车道线的坐标 len(lane_coords)==5
        return mask, lane_coords, points


class LaneNetPostProcessor(object):
    """
    lanenet post process for lane generation
    """
    def __init__(self, ipm_remap_file_path='./data/tusimple_ipm_remap.yml'):
        """

        :param ipm_remap_file_path: ipm generate file path
        """
        assert ops.exists(ipm_remap_file_path), '{:s} not exist'.format(ipm_remap_file_path)

        self._cluster = _LaneNetCluster()
        self._ipm_remap_file_path = ipm_remap_file_path

        remap_file_load_ret = self._load_remap_matrix()
        self._remap_to_ipm_x = remap_file_load_ret['remap_to_ipm_x']
        self._remap_to_ipm_y = remap_file_load_ret['remap_to_ipm_y']

        self._color_map = [np.array([255, 0, 0]),
                           np.array([0, 255, 0]),
                           np.array([0, 0, 255]),
                           np.array([125, 125, 0]),
                           np.array([0, 125, 125]),
                           np.array([125, 0, 125]),
                           np.array([50, 100, 50]),
                           np.array([100, 50, 100])]
    #     红，石灰，蓝，无色名，青绿色，紫色，深绿色，深紫色

    def _load_remap_matrix(self):
        """

        :return:
        """
        fs = cv2.FileStorage(self._ipm_remap_file_path, cv2.FILE_STORAGE_READ)

        remap_to_ipm_x = fs.getNode('remap_ipm_x').mat()
        remap_to_ipm_y = fs.getNode('remap_ipm_y').mat()

        ret = {
            'remap_to_ipm_x': remap_to_ipm_x,
            'remap_to_ipm_y': remap_to_ipm_y,
        }

        fs.release()

        return ret

    def postprocess(self, binary_seg_result, instance_seg_result=None,
                    min_area_threshold=100, source_image=None,
                    data_source='tusimple'):
        """

        :param binary_seg_result:
        :param instance_seg_result:
        :param min_area_threshold:
        :param source_image:
        :param data_source:
        :return:
        """
        # convert binary_seg_result 转换二进制seg结果
        binary_seg_result = np.array(binary_seg_result * 255, dtype=np.uint8)

        # apply image morphology operation to fill in the hold and reduce the small area
        # 应用图像形态学操作来填充保留区域并缩小小区域
        morphological_ret = _morphological_process(binary_seg_result, kernel_size=5)

        connect_components_analysis_ret = _connect_components_analysis(image=morphological_ret)
        # 第二个是一整张图的label，
        # 第三个是(x, y, width, height, area)，即每个区域的每个区域的左上角坐标,宽和高，面积
        # labels ：对原始图中的每一个像素都打上标签，背景为0，连通域打上1，2，3。。。的标签，同一个连通域的像素打上同样的标签。相当与对每一个像素进行了分类（分割）
        # stats：每一连通域的信息，表示每个连通区域的外接矩形（起始点的x、y、宽和高)和面积
        labels = connect_components_analysis_ret[1]

        stats = connect_components_analysis_ret[2]
        for index, stat in enumerate(stats):
            # 连通域的面积小于给定的最小值,设置为黑色，视为背景
            if stat[4] <= min_area_threshold:
                # 只有条件 (condition)，没有x和y，则输出满足条件 (即非0) 元素的坐标 (等价于numpy.nonzero)。
                # 这里的坐标以tuple的形式给出，通常原数组有多少维，输出的tuple中就包含几个数组，分别对应符合条件元素的各维坐标。
                idx = np.where(labels == index)
                # 0是黑色
                morphological_ret[idx] = 0

        # apply embedding features cluster
        mask_image, lane_coords , points = self._cluster.apply_lane_feats_cluster(
            binary_seg_result=morphological_ret,
            instance_seg_result=instance_seg_result,
        )

        if mask_image is None:
            return {
                'mask_image': None,
                'fit_params': None,
                'source_image': None,
            }

        # lane line fit
        fit_params = []
        src_lane_pts = []  # lane pts every single lane
        # print(lane_coords) lane_coords每一条车道线的像素的位置
        # print(len(lane_coords)) 5（共5条车道线）
        # 这个列表中存的是真图中的坐标信息
        all_real_list=[]
        for lane_index, coords in enumerate(lane_coords):
            # print(lane_index)
            # print(coords) 每条车道线上的像素
            if data_source == 'tusimple':
                tmp_mask = np.zeros(shape=(720, 1280), dtype=np.uint8)
                # 转换到原图后的坐标
                # all_real_list.append([list(np.int_(coords[:, 0] * 1280 / 512)), list(np.int_(coords[:, 1] * 720 / 256))])
                all_real_list.append([list(np.int_(coords[:, 0] * 1280 / 512)), list(np.int_(coords[:, 1] * 720 / 256))])

                # print(type(list(np.int_(coords[:, 1] * 720 / 256))))
                # print('y',np.int_(coords[:, 1] * 720 / 256))
                # print('x', np.int_(coords[:, 0] * 1280 / 512))
                tmp_mask[tuple((np.int_(coords[:, 1] * 720 / 256), np.int_(coords[:, 0] * 1280 / 512)))] = 255
            elif data_source == 'beec_ccd':
                tmp_mask = np.zeros(shape=(1350, 2448), dtype=np.uint8)
                # coords 第一维取全部数据，取第二维中第0个或第1个数据
                tmp_mask[tuple((np.int_(coords[:, 1] * 1350 / 256), np.int_(coords[:, 0] * 2448 / 512)))] = 255
            else:
                raise ValueError('Wrong data source now only support tusimple and beec_ccd')
            # cv2.remap()重映射
            # void remap(InputArray src, //输入图像
            #            OutputArraydst, //输出图像
            #            InputArray map1,//第一个映射
            #            InputArray map2, //第二个映射
            #            int interpolation, //插值 最邻近插值
            #
            tmp_ipm_mask = cv2.remap(
                tmp_mask,
                self._remap_to_ipm_x,
                self._remap_to_ipm_y,
                interpolation=cv2.INTER_NEAREST
            )
            # print(tmp_ipm_mask.shape)  (640,640)
            nonzero_y = np.array(tmp_ipm_mask.nonzero()[0])
            # print('nonezero_y',nonzero_y)

            nonzero_x = np.array(tmp_ipm_mask.nonzero()[1])
            # print('nonezero_x', nonzero_x)
            # np.polyfit（二项式拟合）
            fit_param = np.polyfit(nonzero_y, nonzero_x, 2)
            fit_params.append(fit_param)

            [ipm_image_height, ipm_image_width] = tmp_ipm_mask.shape
            plot_y = np.linspace(10, ipm_image_height, ipm_image_height - 10)
            fit_x = fit_param[0] * plot_y ** 2 + fit_param[1] * plot_y + fit_param[2]
            # fit_x = fit_param[0] * plot_y ** 3 + fit_param[1] * plot_y ** 2 + fit_param[2] * plot_y + fit_param[3]

            lane_pts = []
            for index in range(0, plot_y.shape[0], 5):
                src_x = self._remap_to_ipm_x[
                    int(plot_y[index]), int(np.clip(fit_x[index], 0, ipm_image_width - 1))]
                if src_x <= 0:
                    continue
                src_y = self._remap_to_ipm_y[
                    int(plot_y[index]), int(np.clip(fit_x[index], 0, ipm_image_width - 1))]
                src_y = src_y if src_y > 0 else 0

                lane_pts.append([src_x, src_y])

            src_lane_pts.append(lane_pts)
        ''' 求原图中的四个交点'''
        # 这个是在原图中进行绘制的
        x_real_final_left_down, \
        x_real_final_left_up, \
        x_real_final_right_down, \
        x_real_final_right_up = get_final_x(all_real_list, W=1280, H=720, y_stead=300)
        real_points = []
        real_points.append(list(x_real_final_left_up))
        real_points.append(list(x_real_final_right_up))
        real_points.append(list(x_real_final_right_down))
        real_points.append(list(x_real_final_left_down))
        # tusimple test data sample point along y axis every 10 pixels 每 10 个像素沿 y 轴简化测试数据采样点
        source_image_width = source_image.shape[1]
        for index, single_lane_pts in enumerate(src_lane_pts):
            single_lane_pt_x = np.array(single_lane_pts, dtype=np.float32)[:, 0]
            single_lane_pt_y = np.array(single_lane_pts, dtype=np.float32)[:, 1]
            if data_source == 'tusimple':
                start_plot_y = 240
                end_plot_y = 720
            elif data_source == 'beec_ccd':
                start_plot_y = 820
                end_plot_y = 1350
            else:
                raise ValueError('Wrong data source now only support tusimple and beec_ccd')
            step = int(math.floor((end_plot_y - start_plot_y) / 10))
            for plot_y in np.linspace(start_plot_y, end_plot_y, step):
                diff = single_lane_pt_y - plot_y
                fake_diff_bigger_than_zero = diff.copy()
                fake_diff_smaller_than_zero = diff.copy()
                fake_diff_bigger_than_zero[np.where(diff <= 0)] = float('inf')
                fake_diff_smaller_than_zero[np.where(diff > 0)] = float('-inf')
                idx_low = np.argmax(fake_diff_smaller_than_zero)
                idx_high = np.argmin(fake_diff_bigger_than_zero)

                previous_src_pt_x = single_lane_pt_x[idx_low]
                previous_src_pt_y = single_lane_pt_y[idx_low]
                last_src_pt_x = single_lane_pt_x[idx_high]
                last_src_pt_y = single_lane_pt_y[idx_high]

                if previous_src_pt_y < start_plot_y or last_src_pt_y < start_plot_y or \
                        fake_diff_smaller_than_zero[idx_low] == float('-inf') or \
                        fake_diff_bigger_than_zero[idx_high] == float('inf'):
                    continue

                interpolation_src_pt_x = (abs(previous_src_pt_y - plot_y) * previous_src_pt_x +
                                          abs(last_src_pt_y - plot_y) * last_src_pt_x) / \
                                         (abs(previous_src_pt_y - plot_y) + abs(last_src_pt_y - plot_y))
                interpolation_src_pt_y = (abs(previous_src_pt_y - plot_y) * previous_src_pt_y +
                                          abs(last_src_pt_y - plot_y) * last_src_pt_y) / \
                                         (abs(previous_src_pt_y - plot_y) + abs(last_src_pt_y - plot_y))

                if interpolation_src_pt_x > source_image_width or interpolation_src_pt_x < 10:
                    continue
                # tolist()将数组或矩阵转换成列表

                lane_color = self._color_map[index].tolist()
                #  cv2.circle(image, center_coordinates, radius, color, thickness)
                # image:它是要在其上绘制圆的图像。
                # center_coordinates：它是圆的中心坐标。坐标表示为两个值的元组，即(X坐标值，Y坐标值)。
                # radius:它是圆的半径。
                # color:它是要绘制的圆的边界线的颜色。对于BGR，我们通过一个元组。例如：(255，0，0)为蓝色。
                # thickness:它是圆边界线的粗细像素。厚度-1像素将以指定的颜色填充矩形形状。
                #
                #
                # 返回值：它返回一个图像。

                cv2.circle(source_image, (int(interpolation_src_pt_x),
                                          int(interpolation_src_pt_y)), 5, lane_color, -1)
        ret = {
            'mask_image': mask_image,
            'fit_params': fit_params,
            'source_image': source_image,
            'points':points,
            'real_points':real_points

        }

        return ret
