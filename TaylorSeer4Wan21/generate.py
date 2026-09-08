"""TaylorSeer fixed-protocol resident Wan21 entry point."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'Wan21Benchmark'))
from generation import main
if __name__=='__main__':
    sys.argv[1:1]=['--method','taylorseer']
    main()
