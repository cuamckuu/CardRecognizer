#!/usr/bin/env python3

import cv2
import numpy as np
from transform import *
from os.path import exists, basename, splitext
from os import listdir
from random import random

def get_thresh(img, num, c=0, inv=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 100)

    type1 = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    type2 = cv2.THRESH_BINARY if not inv else cv2.THRESH_BINARY_INV
    thresh = cv2.adaptiveThreshold(gray, 255, type1, type2, num, c)

    return thresh


def get_sorted_contours(thresh):
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contours = list(filter(lambda x: cv2.contourArea(x) > 100, contours))

    return contours

def guess_card(final, ref_vals, ref_suits):
    val  = final[2:27, 2:18]
    suit = final[27:47, 2:18]

    val = cv2.cvtColor(val, cv2.COLOR_BGR2GRAY)
    suit = cv2.cvtColor(suit, cv2.COLOR_BGR2GRAY)

    _, val  = cv2.threshold(val,  125, 255, cv2.THRESH_BINARY)
    _, suit = cv2.threshold(suit, 125, 255, cv2.THRESH_BINARY)

# Add func
    min_diff = 1e9
    best_val = None
    for val_name, val_img in ref_vals.items():
        diff_img = cv2.absdiff(val, val_img)
        diff = np.sum(diff_img)

        if diff < min_diff:
            min_diff = diff
            best_val = val_name

    min_diff = 1e9
    best_suit = None
    for suit_name, suit_img in ref_suits.items():
        diff_img = cv2.absdiff(suit, suit_img)
        diff = np.sum(diff_img)

        if diff < min_diff:
            min_diff = diff
            best_suit = suit_name

#     print(best_val, best_suit)
#     cv2.imshow("final", final)
#     cv2.waitKey(0)

    return best_val + best_suit

def init_reference():
    s_path = "img/suits/"
    v_path = "img/values/"
    if exists(s_path) and exists(v_path):
        suits = listdir(s_path)
        suits_dict = dict()
        for suit in suits:
            suit_name = splitext(suit)[0]
            suits_dict[suit_name] = cv2.imread(s_path + suit, 0)

        values = listdir(v_path)
        values_dict = dict()
        for value in values:
            val_name = splitext(value)[0]
            values_dict[val_name] = cv2.imread(v_path + value, 0)

    return (values_dict, suits_dict)

if __name__ == "__main__":
    ref_vals, ref_suits = init_reference()

    img = cv2.imread("values.png")

    thresh   = get_thresh(img, 15, -10)
    contours = get_sorted_contours(thresh)

    if contours:
        sumArea = 0

        for i, card in enumerate(contours):
            eps = 0.03 * cv2.arcLength(card, True)
            while len(card) > 4:
                card = cv2.approxPolyDP(card, eps, True)

            area = cv2.contourArea(card)
            sumArea += area
            meanArea = sumArea / (i + 1)

            if 0.7 * meanArea <= area <= 1.3 * meanArea:
                warp_contours = np.array(list(map(lambda x: x[0], card)))
                final = four_point_transform(img, warp_contours)

                h, w, _ = final.shape
                if w > h:
                    final = np.rot90(final, 1)
                final = cv2.resize(final, (150, 180))

                ans = guess_card(final, ref_vals, ref_suits)
                cv2.drawContours(img, [card], -1, (0, 255, 0), 2)

                mid_x = np.sum(warp_contours[:, 0]) // 4 - 35
                mid_y = np.sum(warp_contours[:, 1]) // 4

                cv2.putText(img, ans, (mid_x, mid_y), 1, 1, (0, 255, 0), 2)

    cv2.imshow("img", img)
    # cv2.imshow("thresh", thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
