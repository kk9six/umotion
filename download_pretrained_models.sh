echo -e "The pretrained models will stored in the 'models' folder"
gdown "https://drive.google.com/uc?export=download&id=1U6mEPhplIsvp67EkKB7w95jzLsEPbRwH"
echo -e "Extracting"
unzip -q models.zip
echo -e "Cleaning"
rm models.zip
echo -e "Downloading done!"
