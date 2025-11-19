create a venv file inside the lmms_eval folder under the root.
python3.12 -m venv .venv

in the lmms_eval 
pip install -e . 
in the LLava-Next 
pip install -e 

in the lmms_eval 
pip uninstall transformers 
pip install transformers==4.38.2

now download the model using this command 
huggingface-cli download \
    lmms-lab/LLaVA-NeXT-Video-7B-Qwen2 \
    --local-dir LLaVA-NeXT-Video-7B-Qwen2 \
    --local-dir-use-symlinks False \
    --include "*.json" "*.safetensors" "*.bin" "*.model" "*.txt" "*.py"
this should be done in the project root, i.e the downloaded folder should be a sibling of lmms_eval 

Now create this 
mkdir -p ~/.cache/huggingface/custom_video_qa_cache/data/data

go to the above folder and download the dataset 
hf download lmms-lab/Video-MME --repo-type dataset --local-dir .
# you can use git lfs for faster download or some other way aswell 
something like this should work

mkdir -p ~/VideoMME_download
cd ~/VideoMME_download
git clone https://huggingface.co/datasets/lmms-lab/Video-MME
 now extract them 

Now we move them to the cache folder you dont need it if you already extracted them to the cache folder 
mkdir -p ~/.cache/huggingface/custom_video_qa_cache/data/data
for example this is the command in my machine 
mv ~/VideoMME_download/Video-MME/extracted/data/* ~/.cache/huggingface/custom_video_qa_cache/data/data/


now you can run the scripts


You have to download the longVideoBench dataset as well 
git clone https://huggingface.co/datasets/longvideobench/LongVideoBench
remember you have to login with hf auth before you do this, and you need dataset access aswell 

now after download extract with this command 
cd /home/train01/LVBench/LongVideoBench
cat videos.tar.part.* | tar -xv

now again we have to move the files to the cache 
from where you extracted the videos run this this command modify the path accordingly 
mv /home/train01/LVBench/LongVideoBench/videos /home/train01/.cache/huggingface/longvideobench_custom_cache/



