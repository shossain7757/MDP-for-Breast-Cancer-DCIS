#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 16:05:24 2022

@author: shossain
"""
from gym.envs.registration import register


register(
    id = 'breastcancer-v1',
    entry_point = 'gym_breastcancer.envs:BreastCancerDCISBandit',
    kwargs = {}
    )


register(
    id = 'breastcancer-v2',
    entry_point = 'gym_breastcancer.envs:BreastCancerDCISCoach',
    kwargs = {}
    )
