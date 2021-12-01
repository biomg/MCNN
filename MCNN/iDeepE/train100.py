import subprocess
class RunCmd(object):
  def cmd_run(self, cmd):
    self.cmd = cmd
    subprocess.call(self.cmd, shell=True)

for i in range(10):
    a = RunCmd()
    a.cmd_run('CUDA_VISIBLE_DEVICES=0 python ideepe.py \
    --posi=../RNA-data/GraphProt_CLIP_sequences/CAPRIN1_Baltz2012.train.positives.fa \
    --nega=../RNA-data/GraphProt_CLIP_sequences/CAPRIN1_Baltz2012.train.negatives.fa \
    --model_type=CNN --train=True --n_epochs=50')
    for j in range(10):
        a = RunCmd()
        a.cmd_run('CUDA_VISIBLE_DEVICES=0 python ideepe.py \
        --testfile=../RNA-data/GraphProt_CLIP_sequences/CAPRIN1_Baltz2012.ls.positives.fa \
        --nega=../RNA-data/GraphProt_CLIP_sequences/CAPRIN1_Baltz2012.ls.negatives.fa \
        --model_type=CNN --predict=True')
