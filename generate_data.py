import os
import shutil
import numpy as np
import numpy.random as npr
from PIL import Image
from pathlib import Path
from tqdm import tqdm

annot_file = Path('build_dataset/wider_face_split/wider_face_train_bbx_gt.txt')
images_dir = Path('build_dataset/WIDER_train/images')

def calc_iou(box, boxes):
    x, y, w, h = box
    xs, ys, ws, hs = np.transpose(boxes)

    box_area = (w + 1) * (h + 1)
    boxes_area = (ws + 1) * (hs + 1)

    nx1 = np.maximum(x, xs)
    ny1 = np.maximum(y, ys)

    nx2 = np.minimum(x + w, xs + ws)
    ny2 = np.minimum(y + h, ys + hs)

    nw = np.maximum(0, nx2 - nx1 + 1)
    nh = np.maximum(0, ny2 - ny1 + 1)

    overlap = nw * nh
    union = box_area + boxes_area - overlap

    return overlap / union

def generate_data(out_size):
    # generated image size
    save_dir = Path(f'data/{out_size}')

    classes = {
        'neg': 0,
        'pos': 1,
        'part': 2
    }

    # scaling of box area for cropping part & pos images
    min_face_ratio = 0.8
    max_face_ratio = 1.25

    # how far the box area should be shifted
    shift_ratio = 0.2

    # minimum bbox size to be considered a face
    min_face_size = 20

    if save_dir.exists():
        while True:
            print(f'{out_size}x{out_size}: save dir already exist, delete? [Y/n]: ', end='')

            ans = input().lower()

            if ans in ['', 'y']:
                shutil.rmtree(save_dir)
                break

            if ans == 'n':
                print('skipping..')
                return

    save_dir.mkdir(parents=True)

    files, saves = dict(), dict()

    for cat in classes.keys():
        files[cat] = (save_dir / f'{cat}.txt').open('w')
        saves[cat] = save_dir / cat
        saves[cat].mkdir()

    with annot_file.open('r') as f:
        annotations = f.readlines()

    # just count how many annotated images there are
    annot_count = 0
    line = 0
    while line < len(annotations):
        box_count = max(1, int(annotations[line+1]))
        line += 2 + box_count
        annot_count += 1

    indices = {'pos': 0, 'neg': 0, 'part': 0, 'current': 0}
    skips = {'small': 0, 'oob': 0}

    line = 0
    pbar = tqdm(total=annot_count)

    while line < len(annotations):
        img_name = annotations[line].rstrip('\n')
        box_count = max(1, int(annotations[line+1]))
        line += 2

        boxes = []
        for i in range(box_count):
            box = np.array(annotations[line+i].split(' ')[:4], dtype=np.int64)
            boxes.append(box)

        line += box_count

        img_path = images_dir / img_name

        with Image.open(img_path) as img:
            ih, iw = img.size

            # Generate negative images (IoU < 0.3)
            for i in range(2):
                size = npr.randint(out_size, min(iw, ih) / 2)

                nx = npr.randint(0, iw - size)
                ny = npr.randint(0, ih - size)

                crop_box = np.array([nx, ny, size, size])
                iou = calc_iou(crop_box, boxes)

                best_iou = np.max(iou)

                if best_iou < 0.3:
                    crop_img = img.crop((nx, ny, nx+size, ny+size))
                    res_img = crop_img.resize((out_size, out_size), Image.ANTIALIAS)

                    save_file = str(saves['neg'] / (f'{indices["neg"]}' + '.jpg'))

                    res_img.save(save_file, format='jpeg')
                    files['neg'].write(save_file + f' {classes["neg"]}\n')

                    indices['neg'] += 1

            for box in boxes:
                x, y, w, h = box

                if min(w, h) < min_face_size:
                    skips['small'] += 1
                    continue

                if x < 0 or y < 0 or x + w > iw or y + h > ih:
                    skips['oob'] += 1
                    continue

                # Generate part & positive images
                size = npr.randint(min(w, h) * min_face_ratio, min(min(iw, ih), max(w, h) * max_face_ratio))

                dx = npr.randint(-w * shift_ratio, w * shift_ratio)
                dy = npr.randint(-h * shift_ratio, h * shift_ratio)

                nx = int(max(0, x + w/2 - size/2 + dx))
                ny = int(max(0, y + h/2 - size/2 + dy))

                if nx + size > iw:
                    nx -= nx + size - iw

                if ny + size > ih:
                    ny -= ny + size - ih

                crop_box = np.array([nx, ny, size, size])
                iou = calc_iou(crop_box, boxes)

                best_iou = np.max(iou)

                # offset for bounding box regression
                off_x1 = (x - nx) / float(size)
                off_y1 = (y - ny) / float(size)
                off_x2 = ((x + w) - (nx + size)) / float(size)
                off_y2 = ((y + h) - (ny + size)) / float(size)

                if best_iou >= 0.65:
                    crop_img = img.crop((nx, ny, nx+size, ny+size))
                    res_img = crop_img.resize((out_size, out_size), Image.ANTIALIAS)

                    save_file = str(saves['pos'] / (f'{indices["pos"]}' + '.jpg'))

                    res_img.save(save_file, format='jpeg')
                    files['pos'].write(save_file + f' {classes["pos"]} {off_x1} {off_y1} {off_x2} {off_y2}\n')

                    indices['pos'] += 1

                elif best_iou >= 0.4:
                    crop_img = img.crop((nx, ny, nx+size, ny+size))
                    res_img = crop_img.resize((out_size, out_size), Image.ANTIALIAS)

                    save_file = str(saves['part'] / (f'{indices["part"]}' + '.jpg'))

                    res_img.save(save_file, format='jpeg')
                    files['part'].write(save_file + f' {classes["part"]} {off_x1} {off_y1} {off_x2} {off_y2}\n')

                    indices['part'] += 1

        indices['current'] += 1

        pbar.update()
        pbar.set_description(f'{out_size}x{out_size}: {indices["current"]} images done, {indices["pos"]} positive, {indices["neg"]} negative, {indices["part"]} part')

    pbar.close()

    for file in files.values():
        file.close()

    print(f'{out_size}x{out_size}: {indices["current"]} images processed in total. small faces: {skips["small"]}, oob: {skips["oob"]}')

if __name__ == '__main__':
    generate_data(out_size=12)
    generate_data(out_size=24)
    generate_data(out_size=48)