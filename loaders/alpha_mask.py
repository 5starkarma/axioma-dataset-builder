import numpy as np
import os

import cv2


class CardAlphaMask():
    
    def __init__(self, cfg, debug=False):
        self.debug = debug
        self.cfg = cfg
        self.zoom = self.cfg['card']['zoom']
        self.card_h = self.cfg['card']['height'] * self.zoom
        self.card_w = self.cfg['card']['width'] * self.zoom
        self.border_size = self.cfg['card']['mask']['border_size']

    def get_mask_size(self):
        if self.debug: print('Creating alpha mask..')
        return np.ones((self.card_h, self.card_w), dtype=np.uint8) * 255
    
    def create(self):
        alpha_mask = self.get_mask_size()
        cv2.rectangle(alpha_mask,
                      (0, 0),
                      (self.card_w - 1, self.card_h - 1),
                      0,
                      self.border_size)
        cv2.line(alpha_mask,
                 (self.border_size * 3, 0),
                 (0, self.border_size * 3), 
                 0,
                 self.border_size)
        cv2.line(alpha_mask,
                 (self.card_w - self.border_size * 3, 0),
                 (self.card_w, self.border_size * 3),
                 0,
                 self.border_size)
        cv2.line(alpha_mask,
                 (0, self.card_h - self.border_size * 3),
                 (self.border_size * 3,self.card_h),
                 0,
                 self.border_size)
        cv2.line(alpha_mask,
                 (self.card_w - self.border_size * 3, self.card_h),
                 (self.card_w, self.card_h - self.border_size * 3),
                 0,
                 self.border_size)
        if self.debug: 
            debug_dir = self.cfg['debug']['dir']
            debug_file = os.path.join(debug_dir, 'alpha_mask.jpg')
            cv2.imwrite(debug_file, alpha_mask)
            print(f'Alpha mask saved: {debug_file}')
            print('----------------------------------')
        return alpha_mask