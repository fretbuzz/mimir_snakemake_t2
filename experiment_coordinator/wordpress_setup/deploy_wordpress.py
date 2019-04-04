import argparse
import os

import scale_wordpress

def main(autoscale_p=False, cpu_percent_cuttoff=80):
    os.chdir("..")
    deployment_scaling = {"pxc": {"min": 7, "max": 7}, "wordpreses": {"min": 10, "max": 23}}
    scale_wordpress.scale_wordpress(autoscale_p, cpu_percent_cuttoff, deployment_scaling)
    scale_wordpress.deploy_wp(deployment_scaling)


if __name__== "__main__":
    parser = argparse.ArgumentParser(description='deploys the wordpress application')

    parser.add_argument('--cpu_cutoff',dest="cpu_cutoff", default='80')
    parser.add_argument('--autoscale_p', dest='autoscale_p', action='store_true',
                        default=False,
                        help='should we autoscale the wordpress service')
    args = parser.parse_args()
    main(autoscale_p=args.autoscale_p, cpu_percent_cuttoff=args.cpu_cutoff)
