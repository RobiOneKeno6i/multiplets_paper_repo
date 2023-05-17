# seismic multiplets scan routine

The repository contains two identical python versions of the core catalog scanning routine: one is the colab ipython porting an the other is the python pure code. The code has been checked to behave exactly as the mathematica original code.  No GUI programming is anyway present: I/O is actually based on read and write of files on disk. Any user interface can be easily added by experienced Python programmers. There are two files: 
An example of input catalog file and some output files have been also added. They are regular ASCII text files.
The input file (Simulated_100k_005-02-2_geo.txt) is a table of 6 columns, each row being a seismic event.
The columns are (from left to right): time(decimal year), latitude(deg), longitude(deg), depth(km), magnitude, id(integer).
The last id column is an event identifier, but it is actually not used by this routine.
The code is open source, not maintained and hopefully "citation-ware" i.e. it can be used completely free, and a citation to the author is more than welcome.
