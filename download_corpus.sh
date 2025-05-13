save_path=/mnt/tidal-alsh01/usr/yuanxiaowei/search-r1/wiki
python scripts/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz