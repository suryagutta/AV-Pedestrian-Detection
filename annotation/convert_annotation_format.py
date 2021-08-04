from __future__ import print_function
from pathlib import Path
import argparse
import json
import cv2
import os
import sys

# Convert FLIR (Forward Looking Infrared) ADAS annotations
def convert_flir_annotations(target_format, input_folder, output_folder, debug, debug_obj):

    # Create output folders if doesn't exist
    output_folder = output_folder+'_'+target_format;
    path = Path(output_folder)
    if not path.exists():
        path.mkdir(parents=True)

    print("Reading json file from folder", input_folder)
    json_files_path = Path(input_folder)
    # There is only one file
    json_files = json_files_path.glob('*.json')
    json_files = list(json_files)

    # There are around 80 categories in thermal_annotations.json
    # Only three categories are kept as FLIR evaluates the model based on only three categories(Person, Bicycle, and Car)
    # Note: FLIR train and Val has category id 17 which is 'dog'
    flir_id_to_category = {'1': 'person', '2': 'bicycle', '3': 'car' }
    debug_id = -1
    if debug:
        # list out keys and values separately
        flir_id_list = list(flir_id_to_category.keys())
        flir_category_list = list(flir_id_to_category.values())
        position = flir_category_list.index(debug_obj)
        debug_id = flir_id_list[position]

    print("Debug ID: ",debug_id)
    total_count = 0

    if(target_format == 'voc'):
        print('TODO: Not yet implemented')

    else: # Assume as yolo
        # Initialize the count for categories
        categories = {0: 0, 1: 0, 2: 0}
        # Cache debug images
        debug_images = set()
        extra_cat_ids = {}
        for json_file in json_files:
            with open(json_file) as f:
                data = json.load(f)
                images = data['images']
                # Note: image_id starts from 0 whereas id starts from 1 in teh annotations file
                print('Total images count: %d' % len(images))
                annotations = data['annotations']
                width = float(images[0]['width']) #640.0
                height = float(images[0]['height']) # 512.0

                for i in range(0, len(images)):
                    image_name = images[i]['file_name']  # thermal_8_bit/FLIR_00001.jpeg
                    image_name = image_name[14:-5]  # FLIR_00001
                    converted_annotations = []
                    for ann in annotations:
                        cat_id = int(ann['category_id'])
                        image_id = ann['image_id']
                        # FLIR data set only upto 3
                        if image_id == i and cat_id <= 3:
                            # Convert to float type
                            left, top, bbox_width, bbox_height = map( float, ann['bbox'])
                            if debug and cat_id == int(debug_id):
                                debug_images.add(image_name)
                            # FLIR ID starts from 1, YOLO ID starts from 0
                            cat_id -= 1
                            categories[cat_id] += 1
                            total_count += 1
                            # Center point coordinates
                            x_center, y_center = (left + bbox_width / 2, top + bbox_height / 2)
                            # yolo expects relative values with respect to image width and height (normalized)
                            # The coordinates are normalized/scaled [0, 1]
                            x_rel, y_rel = (x_center / width, y_center / height)
                            w_rel, h_rel = (bbox_width / width, bbox_height / height)
                            # Each row (one bounding box) in the annotation file will have label_idx/category_id x_center y_center width height
                            converted_annotations.append((cat_id, x_rel, y_rel, w_rel, h_rel))
                        else:
                            if image_id == i and cat_id > 3:
                                if cat_id in extra_cat_ids.keys():
                                    extra_cat_ids[cat_id] += 1
                                else:
                                    print("Extra cat id %s with image file name %s" %(cat_id,image_name))
                                    extra_cat_ids[cat_id] = 1

                    with open(os.path.join(output_folder, image_name + '.txt'), 'w') as fp:
                        fp.write('\n'.join('%d %.6f %.6f %.6f %.6f' % res for res in converted_annotations))

                print("Converted files have been written to folder", output_folder)

            print("Extra Category IDs in FLIR: ", extra_cat_ids)
            # Debug
            print("Converted annotations count by category:",{flir_id_to_category[str(cat+1)]: categories[cat] for cat in categories})
            print('Total converted annotations count: %d'% total_count)
            if debug:
                debug_folder =output_folder+'_debug';
                path = Path(debug_folder)
                if not path.exists():
                    path.mkdir(parents=True)
                print("Debug %s images count: %d" %(debug_obj,len(debug_images)))
                #debug_images = list(debug_images)[:10]
                with open(debug_folder+'/'+debug_obj+'.txt', 'w') as f:
                    f.write('\n'.join("%s.jpeg" % b_img for b_img in debug_images))
                    f.write('\n')

    return ''


def convert_kaist_annotations(target_format, input_folder,output_folder, debug, debug_obj):

    if (target_format == 'voc'):
        print('TODO: Not yet implemented')

    else:  # Assume as yolo

        train_set = ['set00', 'set01', 'set02','set03', 'set04', 'set05']
        test_set = ['set06', 'set07', 'set08','set09', 'set10', 'set11']

        # Create output folders if doesn't exist
        output_folder_train = output_folder + '/train' + '_' + target_format
        path = Path(output_folder_train)
        if not path.exists():
            path.mkdir(parents=True)
        output_folder_test = output_folder + '/test' + '_' + target_format
        path = Path(output_folder_test)
        if not path.exists():
            path.mkdir(parents=True)

        output_folders = {
            'train': output_folder_train,
            'test': output_folder_test
        }

        '''
        registering labeling to number
        0 = person, 1 = people, 2 = cyclist
        '''
        kaist_id_to_category = {'0': 'person', '1': 'people', '2': 'cyclist'}

        #kaist_names_dic_key = ['person', 'people', 'cyclist']
        # Only person detection
        kaist_names_dic_key = ['person']

        values = range(len(kaist_names_dic_key))
        kaist_names_num = dict(zip(kaist_names_dic_key, values))

        for phase in ['train', 'test']:  # train_set, test_set
            print(phase)
            if phase == 'train':
                allset = train_set
                f = open('{0}/{1}.txt'.format(output_folder,phase), 'w')  # create list of images for training
            elif phase == 'test':
                allset = test_set
                f = open('{0}/{1}.txt'.format(output_folder, phase), 'w')  # create list of images for test
            for set_ in allset:  # set0x
                kaist_img_path_set = input_folder + '/images/' + set_
                print(kaist_img_path_set)

                all_V = os.listdir(kaist_img_path_set)
                all_V.sort()

                for V00_ in all_V:

                    kaist_images = os.listdir(kaist_img_path_set + '/' + str(V00_) + '/lwir/')  # infrared images
                    print(input_folder+'/annotations/' + set_ + '/' + str(V00_) + '/')
                    kaist_labels = os.listdir(input_folder+'/annotations/' + set_ + '/' + str(V00_) + '/')
                    kaist_labels.sort()
                    kaist_images.sort()
                    print(V00_)
                    # list of images
                    for indexi, img in enumerate(kaist_images):  # img is the name file, indexi is the iter

                        filename = str(img).split('.jpg')
                        filename = filename[0]

                        kaist_img_totest_path = kaist_img_path_set + '/' + V00_ + '/visible/' + img  # rgb
                        #kaist_img_totest_path = kaist_img_path_set + '/' + V00_ + '/lwir/' + img  # infrared camera
                        kaist_label_totest_path = input_folder+'/annotations/'  + set_ + '/' + V00_ + '/' + filename + '.txt'

                        kaist_label_totest = open(kaist_label_totest_path, 'r')

                        label_contents = kaist_label_totest.readlines()

                        save_path = output_folders[phase] + '/' + set_ + '/' + V00_ + '/visible/'  #make a folder of V00/visible/
                        #save_path = output_folders[phase] + '/' + set_ + '/' + V00_ + '/lwir/'  # make a folder of V00/lwir/
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)

                        for line in label_contents:

                            if (line == '% bbGt version=3\n'):
                                continue  # % bbGt version =3\n skip this part the first line
                            if line != '':

                                kaist_img_totest = cv2.imread(kaist_img_totest_path)  # infrared

                                img_height, img_width = kaist_img_totest.shape[0], kaist_img_totest.shape[1]

                                kaist_label_tosave = save_path + filename + '.txt'

                                data = line.split(' ')

                                x = y = w = h = 0
                                if (len(data) == 12):
                                    class_str = data[0]  # class label
                                    if class_str == 'person?' or class_str == 'person':
                                        class_str = 'person'
                                    else:  # uncomment this for excluding people and cyclist
                                        continue  # go to next line
                                    # create a txt file for annotation
                                    if os.path.exists(kaist_label_tosave):
                                        real_label = open(kaist_label_tosave, 'a')  # append if it exists
                                    else:
                                        real_label = open(kaist_label_tosave, 'w')  # make a new file if it doesn't exists
                                        # save the image set if there is an object
                                        # print("writing to {1}{0}.txt".format(phase, output_folder))
                                        f.write(kaist_img_totest_path + '\n')

                                    # (x,y) center (w,h) size
                                    x1 = float(data[1])
                                    y1 = float(data[2])
                                    x2 = float(data[3])  # object width
                                    y2 = float(data[4])  # object height

                                    bbox_center_x = float((x1 + (x2 / 2.0)) / img_width)  # anchor x
                                    bbox_center_y = float((y1 + (y2 / 2.0)) / img_height)  # anchor y
                                    bbox_width = float(x2 / img_width)
                                    bbox_height = float(y2 / img_height)

                                    line_to_write = str(kaist_names_num[class_str]) + ' ' + str(
                                        bbox_center_x) + ' ' + str(bbox_center_y) + ' ' + str(bbox_width) + ' ' + str(
                                        bbox_height) + '\n'  # 1 line of all texts
                                    #print('Writing {0} to {1}'.format(line_to_write, str(real_label)))
                                    real_label.write(line_to_write)  # write to the new file
                                    sys.stdout.write(str(
                                        int((indexi / len(kaist_images)) * 100)) + '% ' + '*******************->' "\r")
                                    sys.stdout.flush()
                                    real_label.close()

            f.close()
        print("Labels transform finished!")

        return ''


if __name__ == "__main__":
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')


    dataset_choices = ['flir', 'kaist']
    target_formats = ['yolo','voc']
    object_types = ['person', 'bicycle', 'car']
    parser = argparse.ArgumentParser(
        description='This script provides multiple util functions which are used for data creation/processing.')
    parser.add_argument('--dataset', type=str, choices=dataset_choices,
                        help="'flir' or 'kaist'")
    parser.add_argument('--target_format', type=str, choices=target_formats,
                        help="'yolo' or 'voc'")
    parser.add_argument('--input_folder', type=str, help="Input folder containing the files to be converted")
    parser.add_argument('--output_folder', type=str, help="Ouput folder to which the converted files to be written")
    parser.add_argument('--debug', type=str2bool, default='false', help="true or false (default=false)")
    parser.add_argument('--debug_obj', type=str, choices=object_types,
                        default='person',
                        help="'person', 'bicycle', or 'car'")


    args = parser.parse_args()

    if args.dataset == 'flir':
        print('Coverting FLIR ADAS annotations to '+args.target_format+' format ---START')
        print(convert_flir_annotations(args.target_format,args.input_folder, args.output_folder, args.debug, args.debug_obj ))
        print('Coverting FLIR ADAS annotations to '+args.target_format+' format ---END')
    else:
        # Assume ADAS
        print('Coverting KAIST annotations to '+args.target_format+' format ---START')
        print(convert_kaist_annotations(args.target_format,args.input_folder, args.output_folder, args.debug, args.debug_obj ))
        print('Coverting KAIST annotations to '+args.target_format+' format ---END')
