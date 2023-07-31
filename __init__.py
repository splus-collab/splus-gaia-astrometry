#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S-PLUS - Gaia astrometric comparison
====================================
"""
import os
import git

__path__ = os.path.dirname(os.path.abspath(__file__))
repo = git.Repo(__path__)
try:
    latest_tag = repo.git.describe('--tags').lstrip('v')
    __version__ = latest_tag.lstrip('v')
except Exception:
    __version__ = 'unknown'

__author__ = 'Fabio R Hepich'
__credits__ = ['Fabio R Hepich', 'S-PLUS Collaboration']
