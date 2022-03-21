import subprocess
class RunCmd(object):
  def cmd_run(self, cmd):
    self.cmd = cmd
    subprocess.call(self.cmd, shell=True)

# Train the model ten times and model predicts the result ten times on the test set
for i in range(10):
    a = RunCmd()
    a.cmd_run('CUDA_VISIBLE_DEVICES=0 python mcnn.py \
    --posi=../GraphProt_CLIP_sequences/ALKBH5_Baltz2012.train.positives.fa \
    --nega=../GraphProt_CLIP_sequences/ALKBH5_Baltz2012.train.negatives.fa \
    --model_type=CNN --train=True --n_epochs=50')
    for j in range(10):
        a = RunCmd()
        a.cmd_run('CUDA_VISIBLE_DEVICES=0 python mcnn.py \
        --testfile=../GraphProt_CLIP_sequences/ALKBH5_Baltz2012.ls.positives.fa \
        --nega=../GraphProt_CLIP_sequences/ALKBH5_Baltz2012.ls.negatives.fa \
        --model_type=CNN --predict=True')
