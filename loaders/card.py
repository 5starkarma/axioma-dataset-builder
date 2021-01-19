import os
import random
import numpy as np
import pickle

from tqdm import tqdm
import cv2

import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox


class CardLoader:
    """
    For loading and preprocessing playing card images.
    """

    def __init__(self, cfg, debug=False):

        self.debug = debug
        self.cfg = cfg

        self.img_dir = self.cfg['card']['dir']
        self.outpath = self.cfg['card']['extracted']

        self.zoom = self.cfg['card']['zoom']
        self.card_names = [v + s for v in self.cfg['card']['values'] for s in self.cfg['card']['suits']]
        self.card_h = self.cfg['card']['height'] * self.zoom
        self.card_w = self.cfg['card']['width'] * self.zoom
        self.card_pck = self.cfg['card']['pck']

        self.filenames = [os.path.join(self.img_dir, filename) for filename in os.listdir(self.img_dir)]
        self.num_images = len(self.filenames)
        print("Num of card images:", self.num_images)

    @staticmethod
    def resize_image(image, width=None, height=None):
        # https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
        dim = None
        (h, w) = image.shape[:2]
        if width is None and height is None:
            return image
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        return resized

    def remove_image_bg(self, image):
        """Removes green chroma-key background"""
        mask = cv2.inRange(image,
                           np.array(self.cfg['card']['mask']['lower_thresh']),
                           np.array(self.cfg['card']['mask']['upper_thresh']))
        image[mask != 0] = [0, 0, 0]
        return image

    def get_random_image(self, remove_bg=False):
        """Retrieves a random image from the dataset"""
        filename = self.filenames[random.randint(0, self.num_images - 1)]
        img = cv2.imread(filename)
        if remove_bg:
            img = self.remove_image_bg(img)
            img = self.resize_image(img, width=self.cfg['output']['dims'][0])
        if self.debug:
            debug_dir = self.cfg['debug']['dir']
            debug_file = os.path.join(debug_dir, 'card_random.jpg')
            cv2.imwrite(debug_file, img)
            print(f'Random card saved: {debug_file}')
        return img

    def get_all_images(self):
        """Retrieves and removes bg from all images in the dataset"""
        images_without_bg = []
        for filename in self.filenames:
            image = cv2.imread(filename)
            img_no_bg = self.remove_image_bg(image)
            img_no_bg = self.resize_image(img_no_bg,
                                          width=self.cfg['output']['dims'][0])
            images_without_bg.append(img_no_bg)
        if self.debug:
            debug_dir = self.cfg['debug']['dir']
            debug_file = os.path.join(debug_dir, 'card.jpg')
            cv2.imwrite(debug_file, images_without_bg[0])
            print(f'Random card resized w/o bg saved: {debug_file}')
        return images_without_bg

    @staticmethod
    def convert_to_grayscale(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def reduce_noise(img):
        return cv2.bilateralFilter(img, 11, 17, 17)

    @staticmethod
    def find_edges(img):
        return cv2.Canny(img, 30, 200)

    @staticmethod
    def find_contours(img):
        return cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    @staticmethod
    def get_largest_contour(contours):
        return sorted(contours, key=cv2.contourArea, reverse=True)[0]

    @staticmethod
    def get_contour_box(contour):
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        return np.int0(box), rect

    @staticmethod
    def find_contour_area(contour):
        return cv2.contourArea(contour)

    @staticmethod
    def find_box_area(box):
        return cv2.contourArea(box)

    def validate_areas_are_similar(self, contour_area, box_area):
        return contour_area / box_area > self.cfg['card']['contour_similarity_thresh']

    def get_card_shape(self):
        return np.array([[0, 0],
                         [self.card_w, 0],
                         [self.card_w, self.card_h],
                         [0, self.card_h]],
                        dtype=np.float32)

    def get_rotated_card_shape(self):
        return np.array([[self.card_w, 0],
                         [self.card_w, self.card_h],
                         [0, self.card_h],
                         [0, 0]],
                        dtype=np.float32)

    def extract_card(self, img, alpha_mask):
        gray_img = self.convert_to_grayscale(img)
        gray_img = self.reduce_noise(gray_img)
        edge = self.find_edges(gray_img)
        contours, _ = self.find_contours(edge.copy())
        contour = self.get_largest_contour(contours)
        box, rect = self.get_contour_box(contour)
        contour_area = self.find_contour_area(contour)
        box_area = self.find_box_area(box)
        areas_are_similar = self.validate_areas_are_similar(contour_area, box_area)
        card_shape = self.get_card_shape()
        rotated_card_shape = self.get_rotated_card_shape()

        if areas_are_similar:
            # We want transform the zone inside the contour into the 
            # reference rectangle of dimensions (self.card_w,self.card_h)
            ((xr, yr), (wr, hr), thetar) = rect
            # Determine 'mp' the transformation that transforms 'box' 
            # into the reference rectangle
            if wr > hr:
                mp = cv2.getPerspectiveTransform(np.float32(box),
                                                 card_shape)
            else:
                mp = cv2.getPerspectiveTransform(np.float32(box),
                                                 rotated_card_shape)
            # Determine the warped image by applying the transformation 
            # to the image
            warped_img = cv2.warpPerspective(img, mp, (self.card_w, self.card_h))
            # Add alpha layer
            warped_img = cv2.cvtColor(warped_img, cv2.COLOR_BGR2BGRA)
            # Shape (n, 1, 2), with n = number of points reshape into (1, n, 2)
            cnta = contour.reshape(1, -1, 2).astype(np.float32)
            # Apply the transformation 'mp' to the contour
            cntwarp = cv2.perspectiveTransform(cnta, mp)
            cntwarp = cntwarp.astype(np.int)
            # Initialize alpha channel
            alphachannel = np.zeros(warped_img.shape[:2], dtype=np.uint8)
            # Then fill in the contour to make opaque this zone of the card 
            cv2.drawContours(alphachannel, cntwarp, 0, 255, -1)
            # Apply the alpha_mask onto the alpha channel to clean it
            alphachannel = cv2.bitwise_and(alphachannel, alpha_mask)
            # Add the alphachannel to the warped image
            warped_img[:, :, 3] = alphachannel
            extra_bg = cv2.copyMakeBorder(warped_img,
                                          9, 9, 9, 9,
                                          cv2.BORDER_REFLECT_101)
        return areas_are_similar, warped_img

    def define_bbox(self, label):
        """Creates and returns imgaug bbox"""
        return BoundingBox(x1=0,
                           x2=self.card_w,
                           y1=0,
                           y2=self.card_h,
                           label=label)

    def save_to_pickle(self, cards):
        pickle.dump(cards, open(self.card_pck, 'wb'))

    def extract_all(self, alpha_mask):
        print('Extracting all cards...')
        # For img in card images
        img_files = os.listdir(self.img_dir)
        with tqdm(total=len(img_files), desc="Extract cards") as pbar:
            cards = dict()
            for img in img_files:
                read_img = cv2.imread(os.path.join(self.img_dir, img))
                img_no_bg = self.remove_image_bg(read_img)
                img_no_bg = self.resize_image(img_no_bg, width=self.cfg['output']['dims'][0])
                areas_similar, card = self.extract_card(img_no_bg, alpha_mask)

                card_name = os.path.splitext(img)[0]
                bbox = self.define_bbox(card_name)
                cards[card_name] = list()
                cards[card_name].append((card, bbox))

                pbar.update(1)
            self.save_to_pickle(cards)
        pbar.close()

        if self.debug:
            debug_dir = self.cfg['debug']['dir']
            debug_file = os.path.join(debug_dir, 'card_extracted.png')
            cv2.imwrite(debug_file, card)
            print(f'Extracted all cards: {debug_file}')
