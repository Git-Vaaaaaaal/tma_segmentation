#!/bin/ksh 
#$ -q gpu
#$ -j y
#$ -o result.out
#$ -N clamv2
cd $WORKDIR
cd /beegfs/data/work/c-2iia/vb710264/tma_segmentation
source /beegfs/data/work/c-2iia/vb710264/tma_segmentation/tma_segmentation_venv/venv/bin/activate
module load python
export PYTHONPATH=/work/c-2iia/vb710264/tma_segmentation/venv/lib/python3.9/site-packages:$PYTHONPATH
export MPLCONFIGDIR=/work/c-2iia/vb710264/.cache/matplotlib

python /beegfs/data/work/c-2iia/vb710264/tma_segmentation/train.py
python /beegfs/data/work/c-2iia/vb710264/tma_segmentation/predict.py