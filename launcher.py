#!/usr/bin/python3
from multiprocessing import current_process
import EyetrackingFramework as etf


if (__name__ == "__main__"):
    from multiprocessing import freeze_support
    freeze_support()
    mp = etf.mainparser
    util = etf.utilities
    parser = mp.MainParser()
    namespaces = parser.parse()
    parser.visualize()
    indata = []
    result_args = []
    for ns in namespaces:
        if ((ns.intype is not None and ns.intype == "any") or util.test_dtypes(indata, ns.intype)):
            indata = ns.func(ns, indata)
        else:
            print(indata)
            print("Dtype {} for {} not in indata".format(ns.intype, ns.command))
            break
elif (current_process().name == "MainProcess"):
    # if is needed since windows reimports all modules when starting a new process which would cause this message to pop up
    print("To use the framework in your python code import the directory directly instead of importing the launcher")
