#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 10/12/17 8:16 PM 

@author: Hantian Liu
"""

from drawmask import draw_mask

def maskImage(img):
	mask, bbox = draw_mask(img)
	return mask