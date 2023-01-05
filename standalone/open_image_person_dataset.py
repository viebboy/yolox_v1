import fiftyone.zoo as foz
import fiftyone as fo
import argparse
import json
import shutil
import copy
from tqdm import tqdm
import os

def parse_args():
    parser = argparse.ArgumentParser("human detection dataset preparation")
    parser.add_argument("--output-path", required=True, type=str, help="path to save data")
    parser.add_argument("--split", default='train', type=str, help="split type", choices=['train', 'validation'])
    parser.add_argument("--min-area", default=0.0, type=float, help="min area of bounding box")
    parser.add_argument("--max-area", default=0.25, type=float, help="max area of bounding box")

    return parser.parse_args()

def check_existence(path, split):
    if split == 'train':
        img_dir = os.path.join(path, 'train2017')
        anno_file = os.path.join(path, 'annotations', 'train.json')
    else:
        img_dir = os.path.join(path, 'val2017')
        anno_file = os.path.join(path, 'annotations', 'validation.json')
    if os.path.exists(img_dir) and os.path.exists(anno_file):
        return True
    else:
        return False


def download_data(path, split):
    dataset = foz.load_zoo_dataset(
        'open-images-v6',
        split=split,
        classes = ['Person'],
        dataset_dir=path,
        label_types=['detections'],
    )

    dataset.export(
        export_dir=path,
        dataset_type=fo.types.COCODetectionDataset,
        label_field='detections',
    )

    # remove the train split
    shutil.rmtree(os.path.join(path, split))
    # rename the directory data to train2017 or val2017
    target_dir = os.path.join(path, 'train2017') if split == 'train' else os.path.join(path, 'val2017')
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.rename(os.path.join(path, 'data'), target_dir)


def clean_data(path, split, min_area, max_area):
    # params here
    filepath = os.path.join(path, 'labels.json')
    class_names = ['Person']
    class_names.sort()

    with open(filepath, 'r') as fid:
        info = json.loads(fid.read())

    if split == 'train':
        sub_dir = 'train2017'
    else:
        sub_dir = 'val2017'

    # update categories to only include specified classes
    old_category_ids = []
    old_to_new_id = {}
    updated_categories = []
    for cat in info['categories']:
        if cat['name'] in class_names:
            old_id = copy.deepcopy(cat['id'])
            name = cat['name']
            new_id = class_names.index(name)
            cat['id'] = new_id
            old_to_new_id[old_id] = new_id
            old_category_ids.append(old_id)
            updated_categories.append(cat)

    # update annotations
    updated_image_list = []
    updated_annotations = []
    for item in info['annotations']:
        old_id = item['category_id']
        if old_id in old_category_ids:
            new_id = old_to_new_id[old_id]
            item['category_id'] = new_id
            updated_image_list.append(item['image_id'])
            updated_annotations.append(item)

    updated_image_list = set(updated_image_list)
    updated_images = []
    for item in info['images']:
        if item['id'] in updated_image_list:
            updated_images.append(item)

    # update categories about the class
    info['categories'] = updated_categories
    info['annotations'] = updated_annotations
    info['images'] = updated_images

    # now we want to remove too close images
    removed_image_ids = set()
    image_to_size = {}
    for item in updated_images:
        image_to_size[item['id']] = item['height'] * item['width']

    for item in updated_annotations:
        box_size = item['area']
        img_size = image_to_size[item['image_id']]
        if (box_size / img_size < min_area) or (box_size / img_size > max_area):
            removed_image_ids.add(item['image_id'])

    print(f'total number of images: {len(updated_images)}')
    print(f'number of removed images: {len(removed_image_ids)}')

    for idx in range(len(updated_images)-1, -1, -1):
        if updated_images[idx]['id'] in removed_image_ids:
            updated_images.pop(idx)

    for idx in range(len(updated_annotations)-1, -1, -1):
        if updated_annotations[idx]['image_id'] in removed_image_ids:
            updated_annotations.pop(idx)

    info['annotations'] = updated_annotations
    info['images'] = updated_images

    files_to_keep = []
    for item in tqdm(updated_images):
        f = item['file_name']
        files_to_keep.append(os.path.join(path, sub_dir, f))

    all_files = [os.path.join(path, sub_dir, f) for f in os.listdir(os.path.join(path, sub_dir))]
    # remove files not needed
    for f in all_files:
        if f not in files_to_keep:
            os.remove(f)

    # write label file
    annotation_dir = os.path.join(path, 'annotations')
    if not os.path.exists(annotation_dir):
        os.mkdir(annotation_dir)
    annotation_file = os.path.join(annotation_dir, f'{split}.json')

    with open(annotation_file, 'w') as fid:
        fid.write(json.dumps(info, indent=2))


def main():
    args = parse_args()
    if check_existence(args.output_path, args.split):
        print('data was downloaded and prepared')
    else:
        download_data(args.output_path, args.split)
        clean_data(args.output_path, args.split, args.min_area, args.max_area)


if __name__ == '__main__':
    main()
