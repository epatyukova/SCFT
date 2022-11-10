#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 02:53:33 2022

@author: patyukoe
"""
import argparse
import json

from supervisor import Supervisor


def main(args):
    with open(args.config_filename) as f:
        supervisor_config = json.load(f)

    supervisor = Supervisor(**supervisor_config)

    supervisor.run()


if __name__ == '__main__':
    print('This script solves SCFT equations for the mixture of diblock copolymers imitating random correlated copolymer')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default='config.json', type=str,
                        help='Configuration filename')
    args = parser.parse_args()
    main(args)