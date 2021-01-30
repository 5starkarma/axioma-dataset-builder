import os
import pickle
import numpy as np
import random
import time
from random import randrange
import uuid
import statistics
 
import cv2
import yaml

from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBoxesOnImage
from tqdm import trange

from loaders.alpha_mask import CardAlphaMask
from loaders.background import BackgroundLoader
from loaders.card import CardLoader
from utils.arg_parser import parse_args


class DatasetBuilder:
    """
    Augments and applies card images and bboxes to random background images.
    """
    def __init__(self):
        self.args = parse_args()

        with open(self.args.cfg_file, 'r') as file:
            self.cfg = yaml.load(file, Loader=yaml.FullLoader)

        self.bg_loader = BackgroundLoader(self.cfg, debug=self.args.debug)
        self.alpha_mask = CardAlphaMask(self.cfg, debug=self.args.debug)
        self.card_loader = CardLoader(self.cfg, debug=self.args.debug)

        if self.args.debug:
            print('Starting scene builder...')

        self.cards = pickle.load(open(self.cfg['card']['pck'], 'rb'))
        self.card_h = self.cfg['card']['height'] * self.cfg['card']['zoom']
        self.card_w = self.cfg['card']['width'] * self.cfg['card']['zoom']

        self.scene_w = self.cfg['output']['dims'][0]
        self.scene_h = self.cfg['output']['dims'][1]
 
        self.decal_x = int((self.scene_w - self.card_w) / 2)
        self.decal_y = int((self.scene_h - self.card_h) / 2)

    def create_output_directory(self):
        img_dir = self.cfg['output']['dir']
        lbl_dir = self.cfg['output']['dir'].replace('images', 'labels')
        dirs = [img_dir, lbl_dir]
        for folder in dirs:
            if not os.path.exists(folder):
                os.makedirs(folder)
 
    def get_random_card(self):
        """Retrieves a random card from dictionary of cards."""
        card, bbox = random.choice(list(self.cards.values()))[0]
        return card, bbox
 
    def change_bbox_loc(self, bbox):
        """Adjusts the bbox coords after card is applied to empty scene."""
        bbox.x1, bbox.y1, bbox.x2, bbox.y2 = [self.decal_x,
                                              self.decal_y,
                                              self.decal_x + self.card_w,
                                              self.decal_y + self.card_h]
        return bbox
 
    def unpack_card_and_bbox(self, card_dict):
        # ---------------------------------------------------------THIS WAS DELETED AND UNPACKED IN GET_RANDOM_CARD
        return card_dict[0], card_dict[1]
 
    def create_scene_of_zeros(self):
        """Creates an emtpy scene before background and cards are applied."""
        return np.zeros((self.scene_h, self.scene_w, 4), dtype=np.uint8)
 
    def apply_card_to_empty_scene(self, scene, card):
        """Applies card to center of scene."""
        scene[self.decal_y:self.decal_y + self.card_h,
              self.decal_x:self.decal_x + self.card_w, :] = card
        return scene
 
    def set_up_card_scene(self):
        """Gets card, creates scene, applies card to scene."""
        card, bbox = self.get_random_card()
        zeros_scene = self.create_scene_of_zeros()
        card_scene = self.apply_card_to_empty_scene(zeros_scene, card)
        bbox = self.change_bbox_loc(bbox)
        return card_scene, bbox
 
    def plot_bbox(self, scene, bbox):
        """Plots a single bbox on a scene."""
        if scene.shape[2] == 4:
            scene = cv2.cvtColor(scene, cv2.COLOR_BGRA2BGR)
        cv2.rectangle(scene,
                      (int(bbox.x1), int(bbox.y1)),
                      (int(bbox.x2), int(bbox.y2)),
                      self.cfg['card']['bbox']['color'], 
                      self.cfg['card']['bbox']['thickness'])
        return scene

    def debug_file_save(self, img, bbox, name='debug.jpg'):
        """Applies bbox then saves img to debug folder."""
        if self.args.debug:
            img = self.plot_bbox(img, bbox)
            debug_dir = self.cfg['debug']['dir']
            debug_file = os.path.join(debug_dir, name)
            cv2.imwrite(debug_file, img)
 
    def augment_card(self, card, bbox):
        """Augments a single card and associated bbox."""
        seq = iaa.Sequential([
            iaa.Affine(scale=self.cfg['card']['aug']['scale']),
            iaa.Affine(rotate=self.cfg['card']['aug']['rotate']),
            iaa.Affine(translate_percent={"x": self.cfg['card']['aug']['trans_x'],
                                          "y": self.cfg['card']['aug']['trans_y']}),
            iaa.PerspectiveTransform(scale=(0.01, 0.2))])
        card_aug, bbox_aug = seq(image=card, bounding_boxes=bbox)
        self.debug_file_save(card_aug, bbox_aug, name='card_step1_aug.jpg')
        return card_aug, bbox_aug

    def shrink_card(self, card, bbox):
        seq = iaa.Sequential([
            iaa.Affine(scale=self.cfg['card']['aug']['init_scale']),
            iaa.Sometimes(0.5, iaa.Affine(rotate=self.cfg['card']['aug']['init_flip'])),
        ])
        return seq(image=card, bounding_boxes=bbox)

    def set_double_down_card(self, card, bbox):
        seq = iaa.Sequential([
            iaa.Affine(rotate=self.cfg['card']['aug']['dd_rotate']),
        ])
        return seq(image=card, bounding_boxes=bbox)

    def init_translate_card(self, card, bbox):
        seq = iaa.Sequential([iaa.Affine(translate_percent={"x": self.cfg['card']['aug']['init_translate'],
                                                            "y": self.cfg['card']['aug']['init_translate'] * -1})])
        return seq(image=card, bounding_boxes=bbox)

    def translate_card_for_dealer_hand(self, i, card, bbox):
        distance = random.uniform(1.05, 1.1)
        x_pos = self.card_w / self.scene_w * self.cfg['card']['aug']['init_scale'][0] * -1 * distance
        x_trans_percent = i * x_pos + x_pos
        x = (x_trans_percent, x_trans_percent)

        seq = iaa.Sequential([iaa.Affine(translate_percent={"x": x})])

        return seq(image=card, bounding_boxes=bbox)

    def translate_card_for_player_hand(self, i, card, bbox):
        x_pos = self.card_w / self.scene_w / -2 * self.cfg['card']['aug']['init_scale'][0]
        y_pos = self.card_h / self.scene_h / 2 * self.cfg['card']['aug']['init_scale'][0]

        x_trans_percent = i * x_pos + x_pos
        y_trans_percent = i * y_pos + y_pos
        x = (x_trans_percent, x_trans_percent)
        y = (y_trans_percent, y_trans_percent)

        seq = iaa.Sequential([iaa.Affine(translate_percent={"x": x, "y": y})])

        return seq(image=card, bounding_boxes=bbox)

    def augment_hand(self, card, bbox):
        seq = iaa.Sequential([
            iaa.Affine(scale=self.cfg['hand']['aug']['scale']),
            # iaa.Affine(translate_percent={"x": self.cfg['hand']['aug']['translate_x'],
            #                               "y": self.cfg['hand']['aug']['translate_y']}),
            iaa.Affine(rotate=self.cfg['hand']['aug']['rotate']),
            iaa.Sometimes(0.5, iaa.PerspectiveTransform(scale=(0.0, 0.25))),
        ])
        card_aug, bbox_list = seq(image=card, bounding_boxes=bbox)

        if self.args.debug:
            for bbox in bbox_list.items:
                card = self.plot_bbox(card, bbox)
            debug_dir = self.cfg['debug']['dir']
            debug_file = os.path.join(debug_dir, 'card_augmented.jpg')
            cv2.imwrite(debug_file, card)

        return card_aug, bbox_list

    def init_bbox_list(self):
        """Creates empty bbox list with the shape of the scene."""
        shape = (self.scene_h, self.scene_w, 4)
        return BoundingBoxesOnImage([], shape=shape)

    def create_hand_bbox(self, bbox_list, label):
        bbox = None
        for i in bbox_list.items:
            if not bbox:
                bbox = i.copy()
            bbox.x1 = min(i.x1, bbox.x1)
            bbox.y1 = min(i.y1, bbox.y1)
            bbox.x2 = max(i.x2, bbox.x2)
            bbox.y2 = max(i.y2, bbox.y2)
        bbox.label = label
        return bbox

    def set_cards_in_player_hand(self, num_cards):
        """Takes several random cards and translates them to
            the orientation of a players blackjack hand."""
        bbox_list = self.init_bbox_list()
        card, bbox = self.set_up_card_scene()
        card, bbox = self.shrink_card(card, bbox)
        card, bbox = self.init_translate_card(card, bbox)

        bbox_list.items.append(bbox)

        # Randomly split between PLAYER, PLAYER_DD, DEALER
        random_choice = random.random()

        # PLAYER OR PLAYER_DD
        if 0.0 <= random_choice <= 0.33:
            label = 'PLAYER'
            for i in range(num_cards):  # min: 2, max: 8

                new_card, new_bbox = self.set_up_card_scene()
                new_card, new_bbox = self.shrink_card(new_card, new_bbox)

                # Move card to top of scene
                new_card, new_bbox = self.init_translate_card(new_card, new_bbox)
                # Augment: translate x += 50%-60% of card, y += 50%-60% of card
                new_card, new_bbox = self.translate_card_for_player_hand(i, new_card, new_bbox)
                # Apply augmented card to new card
                mask = new_card[:, :, 3]
                full_mask = np.stack([mask] * 4, -1)
                card = np.where(full_mask, new_card[:, :, 0:4], card)
                # Save bbox to list of bboxes
                bbox_list.items.append(new_bbox)
                # Create bbox around the whole hand
        # PLAYER_DD should only have 2 cards to double down
        elif 0.33 <= random_choice <= 0.66:
            label = 'PLAYER_DD'
            num_cards = 2

            for i in range(num_cards):

                new_card, new_bbox = self.set_up_card_scene()
                new_card, new_bbox = self.shrink_card(new_card, new_bbox)

                # PLAYER_DD randomly (50%) apply 90 degree rotation to the last card (double_down)
                if i == num_cards - 1:
                    new_card, new_bbox = self.set_double_down_card(new_card, new_bbox)

                # Move card to top of scene
                new_card, new_bbox = self.init_translate_card(new_card, new_bbox)
                # Augment: translate x += 50%-60% of card, y += 50%-60% of card
                new_card, new_bbox = self.translate_card_for_player_hand(i, new_card, new_bbox)
                # Apply augmented card to new card
                mask = new_card[:, :, 3]
                full_mask = np.stack([mask] * 4, -1)
                card = np.where(full_mask, new_card[:, :, 0:4], card)
                # Save bbox to list of bboxes
                bbox_list.items.append(new_bbox)
                # Create bbox around the whole hand
        else:
            label = 'DEALER'
            for i in range(num_cards):  # min: 2, max: 8

                new_card, new_bbox = self.set_up_card_scene()
                new_card, new_bbox = self.shrink_card(new_card, new_bbox)

                # Move card to top of scene
                new_card, new_bbox = self.init_translate_card(new_card, new_bbox)
                # Augment: translate x += 100% of card
                new_card, new_bbox = self.translate_card_for_dealer_hand(i, new_card, new_bbox)
                # Apply augmented card to new card
                mask = new_card[:, :, 3]
                full_mask = np.stack([mask] * 4, -1)
                card = np.where(full_mask, new_card[:, :, 0:4], card)
                # Save bbox to list of bboxes
                bbox_list.items.append(new_bbox)
                # Create bbox around the whole hand

        if self.args.debug:
            # for bbox in bbox_list.items:
            #     print(bbox.label)
            #     card = self.plot_bbox(card, bbox)
            debug_dir = self.cfg['debug']['dir']
            debug_file = os.path.join(debug_dir, 'player_hand.jpg')
            cv2.imwrite(debug_file, card)

        return card, bbox_list, label
 
    def apply_card_to_scene(self, background, bbox_list):
        """Applies augmentation to card and applies it to the background."""
        card_scene, bbox = self.set_up_card_scene()
        card_aug, bbox_aug = None, None
        for _ in range(self.cfg['scene']['aug_max_tries']):
            card_aug, bbox_aug = self.augment_card(card_scene, bbox)
            if bbox_aug.is_fully_within_image(card_aug):
                if not len(bbox_list.items):
                    bbox_list.items.append(bbox_aug)
                    break
                elif len(bbox_list.items):
                    iou = False
                    for j in range(len(bbox_list.items)):
                        iou = bbox_aug.iou(bbox_list.bounding_boxes[j])
                        if iou:
                            break
                    if not iou:
                        bbox_list.items.append(bbox_aug)     
                        break

        card_mask_sc = card_aug[:, :, 3]  # single-channel mask
        card_mask = np.stack([card_mask_sc] * 3, -1)  # three-channel mask
        background = np.where(card_mask, card_aug[:, :, 0:3], background)

        self.debug_file_save(background, bbox_aug, name='card_step2_scene.jpg')
        return background, bbox_list

    def apply_hand_to_scene(self, bg, bbox_list1, num_cards):
        added_to_scene = False
        iou = False

        for i in range(self.cfg['scene']['aug_max_tries']):
            # Create and augment hand
            hand, hand_bbox_list, label = self.set_cards_in_player_hand(num_cards)
            hand_aug, bbox_list_aug = self.augment_hand(hand, hand_bbox_list)
            bbox1 = self.create_hand_bbox(bbox_list_aug, label=label)
            bbox_list_aug.items.append(bbox1)

            # Make sure bbox_list_aug items are inside image
            for bbox_aug in bbox_list_aug.items:
                if not bbox_aug.is_fully_within_image(hand_aug):
                    break
            else:
                # If bbox_list1 is empty
                if not len(bbox_list1):
                    added_to_scene = True
                    for bbox_aug in bbox_list_aug.items:
                        bbox_list1.items.append(bbox_aug)
                    break
                # Otherwise check IoU with all other bboxes
                else:
                    iou = False
                    # For each box in bbox_list
                    for box1 in bbox_list1.items:
                        for box2 in bbox_list_aug.items:
                            if box1.iou(box2):
                                iou = True
                    # For each box in bbox_list
                    for box1 in bbox_list_aug.items:
                        for box2 in bbox_list1.items:
                            if box2.iou(box1):
                                iou = True
                    if not iou:
                        for bbox_aug in bbox_list_aug.items:
                            # Add augmented boxes to list
                            bbox_list1.items.append(bbox_aug)
                        added_to_scene = True
                        break

        if added_to_scene and not iou:
            mask1 = hand_aug[:, :, 3]
            mask1 = np.stack([mask1] * 3, -1)
            bg = np.where(mask1, hand_aug[:, :, 0:3], bg)

        if self.args.debug:
            for bbox_aug in bbox_list1.items:
                hand = self.plot_bbox(bg, bbox_aug)
            debug_dir = self.cfg['debug']['dir']
            debug_file = os.path.join(debug_dir, 'hand_applied_to_bg.jpg')
            cv2.imwrite(debug_file, hand)

        return bg, bbox_list1
 
    def save_image(self, img, filename):
        """Saves image to directory."""
        output_dir = self.cfg['output']['dir']
        output_file = os.path.join(output_dir, filename + '.jpg')
        cv2.imwrite(output_file, img)
 
    def create_yolo_file(self, filename, bbox_list):
        """Create YOLOv5 annotation file."""
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
                writer.write(f'{name} {x_center} {y_center} {width} {height}\n')
        writer.close()

    def create_scenes(self, bg_loader):
        """Creates a scene with augmented cards."""
        if self.args.debug:
            tic = time.perf_counter()

        scene = bg_loader.get_random()
        bbox_list = self.init_bbox_list()

        if self.args.single:
            for i in range(randrange(self.cfg['scene']['max_cards']) + 1):
                scene, bbox_list = self.apply_card_to_scene(scene,
                                                            bbox_list)
        if self.args.blackjack:
            for i in range(randrange(self.cfg['scene']['max_hands']) + 1):
                num_cards = randrange(5) + 1
                scene, bbox_list = self.apply_hand_to_scene(scene,
                                                            bbox_list,
                                                            num_cards)

        filename = str(uuid.uuid4())
        self.save_image(scene, filename)
        self.create_yolo_file(filename, bbox_list)

        if self.args.debug:
            toc = time.perf_counter()
            print(f"Operation ran in {toc - tic:0.4f} seconds")
            print(f'Bboxes: {bbox_list.items}')
            print(f'Final cards in scene: {len(bbox_list.items)}')
            for i in bbox_list.items:
                self.plot_bbox(scene, i)
            debug_dir = self.cfg['debug']['dir']
            debug_file = os.path.join(debug_dir, 'scene.jpg')
            cv2.imwrite(debug_file, scene)

    def run(self):

        self.create_output_directory()

        if self.args.extract:
            alpha_mask = self.alpha_mask.create()
            self.card_loader.extract_all(alpha_mask)

        for _ in trange(self.args.size):
            if self.args.single:
                self.create_scenes(self.bg_loader)
            if self.args.blackjack:
                self.create_scenes(self.bg_loader)


if __name__ == '__main__':
    builder = DatasetBuilder()
    builder.run()
