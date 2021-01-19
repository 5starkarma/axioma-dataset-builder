import os
import pickle
import numpy as np
import random
import time
from random import randrange
import uuid
import statistics
 
import cv2
from tqdm import tqdm
 
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage 
 
 
class Scene():
    """
    Augments and applies card images and bboxes to random background images.
    """
    def __init__(self, cfg, bg_loader, ann='yolo', debug=False):
        self.cfg = cfg
        self.debug = debug
 
        self.iou_thresh = self.cfg['scene']['iou_thresh']
        self.max_cards = self.cfg['scene']['max_cards_in_scene']
        self.num_cards_in_scene = randrange(self.max_cards) + 1
        self.annotation_format = ann
 
        self.bg_loader = bg_loader
 
        self.zoom = self.cfg['card']['zoom']
        self.cards = pickle.load(open(self.cfg['card']['pck'],'rb'))
        self.card_h = self.cfg['card']['height'] * self.zoom
        self.card_w = self.cfg['card']['width'] * self.zoom

        self.aug_scale = self.cfg['card']['aug']['scale']
        self.aug_rotate = self.cfg['card']['aug']['rotate']
        self.aug_trans_x = self.cfg['card']['aug']['translate_x']
        self.aug_trans_y = self.cfg['card']['aug']['translate_y']
 
        self.dims = self.cfg['output']['dims']
        self.scene_w = self.dims[0]
        self.scene_h = self.dims[1]
 
        self.decal_x = int((self.scene_w - self.card_w) / 2)
        self.decal_y = int((self.scene_h - self.card_h) / 2)
 
    def get_random_card(self):
        return random.choice(list(self.cards.values()))[0]
 
    def change_bbox_loc(self, bbs):
        bbs.x1, bbs.x2, bbs.y1, bbs.y2 = [self.decal_x, 
                                          self.decal_x + self.card_w, 
                                          self.decal_y, 
                                          self.decal_y + self.card_h]
        return bbs
 
    def get_card_and_bbox(self, card_dict):
        return card_dict[0], card_dict[1]
 
    def create_scene_of_zeros(self):
        return np.zeros((self.scene_h, self.scene_w, 4), dtype=np.uint8)
 
    def apply_card_to_scene_of_zeros(self, card_1, card):
        card_1[self.decal_y:self.decal_y + self.card_h, 
               self.decal_x:self.decal_x + self.card_w, :] = card
        return card_1
 
    def set_up_card_and_bbox(self):
        card_dict = self.get_random_card()
        card, bbox = self.get_card_and_bbox(card_dict)
        card_1 = self.create_scene_of_zeros()
        card_1 = self.apply_card_to_scene_of_zeros(card_1, card)
        bbox = self.change_bbox_loc(bbox)
        return card_1, bbox
 
    def plot_bboxes(self, card, bbs):
        if card.shape[2] == 4:
            card = cv2.cvtColor(card, cv2.COLOR_BGRA2BGR)
        cv2.rectangle(card, 
                      (int(bbs.x1), int(bbs.y1)), 
                      (int(bbs.x2), int(bbs.y2)), 
                      self.cfg['card']['bbox']['color'], 
                      self.cfg['card']['bbox']['thickness'])
        return card
 
    def augment(self, card, bbox):
        # Put augmentation settings in yaml
        seq = iaa.Sequential([
            
            iaa.Affine(scale=self.aug_scale),
            iaa.Affine(rotate=self.aug_rotate),
            iaa.Affine(translate_percent={"x":self.aug_trans_x,
                                          "y":self.aug_trans_y}),
            iaa.PerspectiveTransform(scale=(0.01, 0.2)),
        ])
        card_aug, bbs_aug = seq(image=card, bounding_boxes=bbox)
 
        if self.debug:
            card = self.plot_bboxes(card_aug, bbs_aug)
            debug_dir = self.cfg['debug']['dir']
            debug_file = os.path.join(debug_dir, 'card_augmented.jpg')
            cv2.imwrite(debug_file, card_aug)
            
        return card_aug, bbs_aug

    def init_empty_bbox_list(self):
        shape = (self.scene_h, self.scene_w, 4)
        return BoundingBoxesOnImage([], shape=shape)
 
    def fit_card_in_scene(self, bg, bbox_list):
        card, bbox = self.set_up_card_and_bbox()
        # Augment and check that IoU is small
        for i in range(self.cfg['scene']['aug_max_tries']):
            card_aug, bbox_aug = self.augment(card, bbox)
            # Check if bbox is fully within image
            if bbox_aug.is_fully_within_image(card_aug): 
                # If bbox list is empty no need to check IoU
                if not len(bbox_list.items):
                    bbox_list.items.append(bbox_aug)
                    break
                # Otherwise check IoU with all other bboxes
                elif len(bbox_list.items):
                    # For each box in bbox_list
                    for i in range(len(bbox_list.items)):
                        # Get the IoU versus all other bboxes
                        iou = bbox_aug.iou(bbox_list.bounding_boxes[i])
                        # If IoU is over threshold break the loop
                        if iou > self.cfg['scene']['iou_thresh']: 
                            break
                    # If the last IoU is < thresh then append, else retry aug
                    if iou < self.cfg['scene']['iou_thresh']: 
                        bbox_list.items.append(bbox_aug)     
                        break               
         # Apply card to bg
        mask1 = card_aug[: ,: ,3]
        mask1 = np.stack([mask1] * 3, -1)
        bg = np.where(mask1, card_aug[:, :, 0:3], bg)

        if self.debug:
            card = self.plot_bboxes(card_aug, bbox_aug)
            debug_dir = self.cfg['debug']['dir']
            debug_file = os.path.join(debug_dir, 'card_applied_to_bg.jpg')
            cv2.imwrite(debug_file, card)
 
        return bg, bbox_list

    def create_filename(self):
        # Create random filename using UUID
        return str(uuid.uuid4())
 
    def save_image(self, scene, filename):
        output_dir = self.cfg['output']['dir']
        output_file = os.path.join(output_dir, filename + '.jpg')
        return cv2.imwrite(output_file, scene)
 
    def create_yolo_file(self, filename, bbox_list):
        # Create yolo annotation file
        # Each row is class x_center y_center width height format.
        # Box coordinates are normalized xywh format (from 0 - 1). 
        # If your boxes are in pixels, divide x_center and width by image 
        # width, and y_center and height by image height.
        # Class numbers are zero-indexed (start from 0).
        output_dir = self.cfg['output']['dir']
        output_file = os.path.join(output_dir.replace('images', 'labels'), filename + '.txt')
        classes = self.cfg['card']['classes']
        with open(output_file, 'w') as writer:
            for bbox in bbox_list.items:
                x_coords = [bbox.x1, bbox.x2]
                y_coords = [bbox.y1, bbox.y2]
                x_center = statistics.mean(x_coords) / self.scene_w
                y_center = statistics.mean(y_coords) / self.scene_h
                width = (bbox.x2 - bbox.x1) / self.scene_w
                height = (bbox.y2 - bbox.y1) / self.scene_h
                name = str(classes.index(bbox.label))
                writer.write('{} {} {} {} {}\n'.format(name, 
                                                       x_center,
                                                       y_center,
                                                       width,
                                                       height))
        writer.close()
 
    def create_voc_file(self, filename, bbox_list):
        # Create pascal voc annotation file
        return # File save loc
 
    def save_coords(self, filename, bbox_list):
        if self.annotation_format == 'yolo':
            self.create_yolo_file(filename, bbox_list)
        elif self.annotation_format == 'voc':
            self.create_voc_file(filename, bbox_list)

    def create(self):
        if self.debug:
            # Start time
            tic = time.perf_counter()
        # Get random bg
        bg = self.bg_loader.get_random()
        bbox_list = self.init_empty_bbox_list()
        if self.debug:
            print(f'Num cards in scene: {self.num_cards_in_scene}')
        # For num_cards_in_scene:
        for i in range(self.num_cards_in_scene):
            # Try to augment card and fit in scene
            bg, bbox_list = self.fit_card_in_scene(bg, bbox_list)
        
        filename = self.create_filename()
        image_file = self.save_image(bg, filename)
        coord_file = self.save_coords(filename, bbox_list)        

        if self.debug:
            toc = time.perf_counter()
            print(f"Operation ran in {toc - tic:0.4f} seconds")
            print(f'Bboxes: {bbox_list.items}')
            print(f'Final cards in scene: {len(bbox_list.items)}')
            for i in bbox_list.items:
                self.plot_bboxes(bg, i)
            debug_dir = self.cfg['debug']['dir']
            debug_file = os.path.join(debug_dir, 'scene.jpg')
            cv2.imwrite(debug_file, bg)
 

        