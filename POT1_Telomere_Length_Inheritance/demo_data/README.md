
The fastq files in the demo data folder were derived from the same sample id ‘_supporting_reads’ e.g. PBC41886-a76cb5a7-6406-4d96-8e8c-20ca68b66ce5... (from Supp Table 2).

**The code was developed in VS Code v1.100+ and run-tested only on Apple Silicon (M1+) MacOS.**

Before running Telogator2, make sure to create folders matching the sample name and correctly specify the stdout to match the sample name to avoid ValueError in 1_BulkTelLengthPlots.ipynb.

**Example command:**
```
python telogator2.py -i 24-03685.fastq.gz  -o 24-03685/ -r ont -n 7 --minimap2 minimap2/minimap2 --filt-sub 200 -l 1000 -p 10 > 24-03685/24-03685.out&&

python telogator2.py -i 24-08417.fastq.gz  -o 24-08417/ -r ont -n 7 --minimap2 minimap2/minimap2 --filt-sub 200 -l 1000 -p 10 > 24-08417/24-08417.out&&

python telogator2.py -i 24-08418.fastq.gz  -o 24-08418/ -r ont -n 7 --minimap2 minimap2/minimap2 --filt-sub 200 -l 1000 -p 10 > 24-08418/24-08418.out
```