
# **Setup Instructions**

## **1. Create a virtual environment**

Create a venv inside the `lmms_eval` folder under the root:

```bash
cd lmms_eval
python3.12 -m venv .venv
```
This also works with python3.13

## **2. Install packages**

Inside `lmms_eval`:

```bash
pip install -e .
```

Inside `LLaVA-NeXT`:

```bash
pip install -e .
```

## **3. Install the correct Transformers version**

Inside `lmms_eval`:

```bash
pip uninstall transformers
pip install transformers==4.38.2
```

---

## **4. Download the model**

Run this command in the **project root** (the downloaded model folder must be a sibling of `lmms_eval`):

```bash
huggingface-cli download \
lmms-lab/LLaVA-NeXT-Video-7B-Qwen2 \
--local-dir LLaVA-NeXT-Video-7B-Qwen2 \
--local-dir-use-symlinks False \
--include ".json" ".safetensors" ".bin" ".model" ".txt" ".py"
```

---

## **5. Prepare the cache directory for Video-MME**

Create the cache folder:

```bash
mkdir -p ~/.cache/huggingface/custom_video_qa_cache/data/data
```

Go to the above folder and download the dataset:

```bash
hf download lmms-lab/Video-MME --repo-type dataset --local-dir .
```

You can use Git LFS for faster download or other methods as well.
Something like this should work:

```bash
mkdir -p ~/VideoMME_download
cd ~/VideoMME_download
git clone https://huggingface.co/datasets/lmms-lab/Video-MME
```

Now extract them.

---

## **6. Move the extracted Video-MME data into the cache**

You don’t need this step if you already extracted them directly into the cache folder.

Example command (modify path if needed):

```bash
mv ~/VideoMME_download/Video-MME/extracted/data/* \
~/.cache/huggingface/custom_video_qa_cache/data/data/
```

Now you can run the scripts.

---

## **7. Download the LongVideoBench dataset**

Clone the dataset:

```bash
git clone https://huggingface.co/datasets/longvideobench/LongVideoBench
```

Remember:

* You must run `hf login` before downloading.
* You need dataset access granted on Hugging Face.

After download, extract using:

```bash
cd /home/train01/LVBench/LongVideoBench
cat videos.tar.part.* | tar -xv
```

---

## **8. Move the LongVideoBench videos to the cache**

Modify the path accordingly:

```bash
mv /home/train01/LVBench/LongVideoBench/videos \
/home/train01/.cache/huggingface/longvideobench_custom_cache/
```

## **9. How to run 
To run use this command 
```bash
bash total_run.sh
```
you can use the nohup if you need 
also for invidivual run of the dataset i.e videomme and longvideo bench use 'generate_and_run_LV.sh' and 'generate_and_run_custom_videoqa.sh'
