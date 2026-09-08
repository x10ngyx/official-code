"""Delegate fixed-protocol generation to the local benchmark adapter."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'Wan21Benchmark'))
from generation import main
if __name__=='__main__':
    sys.argv[1:1]=['--method','dicache']
    main()
