import numpy as np
import os
import tensorflow as tf
from utils import label_map_util
from utils import visualization_utils as vis_util
from PIL import Image
import cv2 as cv
from keras.models import load_model
import json
import time

def image_classifier(img_path):
    with open('data/object_num_name.json') as file:
        object_num_name = json.load(file)

    model = load_model('pretrained_models/model_trained.h5')

    inputing = []
    img = cv.imread(img_path)
    img = cv.resize(img, (299, 299))
    img = np.array(img)
    inputing.append(img)

    prediction_num = np.argmax(model.predict(np.array(inputing)), asix=-1)[0]

    return object_num_name[prediction_num]


def get_results(image_path):
    object_detection_result, image_classifier_result = [], []

    # 重置图
    tf.compat.v1.reset_default_graph()
    '''
    载入模型以及数据集样本标签，加载待测试的图片文件
    '''
    # 指定要使用的模型的路径  包含图结构，以及参数
    PATH_TO_CKPT = './pretrained_models/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb'

    # 数据集对应的label mscoco_label_map.pbtxt文件保存了index到类别名的映射
    PATH_TO_LABELS = os.path.join('./data', 'mscoco_label_map.pbtxt')
    NUM_CLASSES = 90

    # 重新定义一个图
    output_graph_def = tf.compat.v1.GraphDef()

    with tf.compat.v1.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        # 将*.pb文件读入serialized_graph
        serialized_graph = fid.read()
        # 将serialized_graph的内容恢复到图中
        output_graph_def.ParseFromString(serialized_graph)
        # print(output_graph_def)
        # 将output_graph_def导入当前默认图中(加载模型)
        tf.import_graph_def(output_graph_def, name='')

    # 载入coco数据集标签文件
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    def load_image_into_numpy_array(image):
        im_width, im_height = image.size
        return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint0)

    # 使用默认图，此时已经加载了模型
    detection_graph = tf.compat.v1.get_default_graph()

    with tf.compat.v1.Session(graph=detection_graph) as sess:
        image = Image.open(image_path)
        # 将图片转换为numpy格式
        image_np = load_image_into_numpy_array(image)

        '''
        定义节点，运行并可视化
        '''
        # 将图片扩展一维，最后进入神经网络的图片格式应该是[1,?,?,3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        '''
        获取模型中的tensor
        '''
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # boxes用来显示识别结果
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Echo score代表识别出的物体与标签匹配的相似程度，在类型标签后面
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        # 开始检查
        boxes, scores, classes, num_detections = sess.run([boxes, scores, classes, num_detections],
                                                          feed_dict={image_tensor: image_np_expanded})
        # 可视化结果
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)

        for i in range(len(np.squeeze(scores))):
            if np.squeeze(scores)[i] > 0.7:
                object_detection_result.append(category_index[np.squeeze(classes).astype(np.int32)[i]]["name"])
                print("Objection Detection!")
                break
                # print(category_index[np.squeeze(classes).astype(np.int32)[i]]["name"], np.squeeze(scores)[i])
            else:
                if len(object_detection_result) == 0:
                    image_classifier_result.append(image_classifier(image_path))
                print("Image Classifier!")
                break

        return object_detection_result, image_classifier_result


if __name__ == '__main__':
    object_detection_results, image_classifier_results = get_results('test_images/bottle.jpg')

    print(object_detection_results)
    print(image_classifier_results)

