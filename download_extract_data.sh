#!/bin/sh
echo "Downloading..."
wget --no-check-certificate -r 'https://docs.google.com/uc?export=download&id=1JKAluJEagidnUYin77yjoiN_FW63zuZj' -O img_align_celeba.zip
echo "Download completed."
echo "Extracting..."
unzip img_align_celeba.zip -d ../dataset/celebA
echo "Extraction completed."
