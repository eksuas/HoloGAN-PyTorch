def download_dataset():
    import os
    os.system("pip install gdown")
    os.system("gdown https://drive.google.com/uc?id=1JKAluJEagidnUYin77yjoiN_FW63zuZj")
def extract_dataset():
    from zipfile import ZipFile
    path_to_zip = "img_align_celeba.zip"
    path_to_extract = "../dataset/fake/celebA/"
    with ZipFile(path_to_zip, 'r') as zipObj:
        zipObj.extractall(path_to_extract)

if __name__ == '__main__':
    download_dataset()
    extract_dataset()