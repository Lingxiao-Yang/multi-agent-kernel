"""Command line for the registered public-kernel study."""
import argparse
import os
from pathlib import Path

from .campaign.preflight import run_preflight
from .campaign.jobs import run_campaign
from .campaign.report import run_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)

    preflight = commands.add_parser('preflight', help='check a shared starting program')
    preflight.add_argument('phase',choices=['preflight','calibrate','hold','hand','stress'])
    preflight.add_argument('--case',required=True)
    preflight.add_argument('--output',type=Path,required=True)
    preflight.set_defaults(handler=run_preflight, parser=preflight)

    run = commands.add_parser('run', help='run one registered search')
    report_phases = ['matched-smoke','matvec-development','matvec-recheck','row-qualification','development','holdout']
    run.add_argument('phase',choices=['smoke','multi-smoke','calibration',*report_phases])
    run.add_argument('--preflight',type=Path,required=True)
    run.add_argument('--output',type=Path,required=True)
    run.add_argument('--counters',type=Path,required=True)
    run.add_argument('--retry-transport',action='store_true',help='Run one new search after a recorded transport interruption')
    run.add_argument('--job',type=int,required=True,help='Zero-based index in the phase')
    run.add_argument('--model',default=os.environ.get('KERNEL_BASELINE_MODEL'))
    run.add_argument('--endpoint',default=os.environ.get('OPENAI_BASE_URL','https://api.openai.com/v1'))
    run.add_argument('--input-price',type=float,default=os.environ.get('KERNEL_BASELINE_INPUT_PRICE'))
    run.add_argument('--output-price',type=float,default=os.environ.get('KERNEL_BASELINE_OUTPUT_PRICE'))
    run.set_defaults(handler=run_campaign, parser=run)

    report = commands.add_parser('report', help='summarize directly paired finalists')
    report.add_argument('phase',choices=report_phases)
    report.add_argument('output',type=Path)
    report.add_argument('--freeze',action='store_true',help='Register the measured single implementation before holdout')
    report.add_argument('--measure-pairs',action='store_true',help='Measure missing finalist pairs on the registered GPU')
    report.add_argument('--require-seed',type=int,help='Require a completed matvec recheck with matched starting evidence')
    report.add_argument('--matvec-sweep',type=Path,help='Rank matvec recheck endpoints against a completed launch sweep')
    report.set_defaults(handler=run_report, parser=report)

    args = parser.parse_args(argv)
    args.handler(args, args.parser)
