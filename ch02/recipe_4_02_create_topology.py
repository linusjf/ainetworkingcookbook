#!/usr/bin/env python
"""
Recipe402CreateTopology.

######################################################################
# @author      : Linus Fernandes (linusfernandes at gmail dot com)
# @file        : recipe_4_02_create_topology
# @created     : Saturday Mar 14, 2026 16:33:38 IST
# @description :
# -*- coding: utf-8 -*-'
######################################################################
"""
"""
Simple script to convert dot file to PNG topology visualization
"""
import subprocess
import sys
dot_file = sys.argv[1]
output_file = "topology.png"
# Generate PNG from dot file
subprocess.run(['dot', '-Tpng', dot_file, '-o', output_file])
print(f"Generated {output_file}")
