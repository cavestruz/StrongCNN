import os, sys
import os.path
sys.path.insert(0, os.path.abspath("./model_training/StrongCNN/pipeline/"))
from pipeline_image_processors import PreprocessHST

os.chdir('./02/YES')
images = os.listdir("./")
if not os.path.exists("../../processed/YES/PreprocessHST"):
    os.makedirs("../../processed/YES/PreprocessHST")

images = [fits.open("./" + file)[0].data for file in images]
phst = PreprocessHST()
phsted = proc.fit_transform(images)


os.chdir("../../processed/YES/PreprocessHST")
x = 0
for name in images:
	new_name = "phsted_" + name
    fits.writeto(new_name, phsted[x], output_verify='exception', overwrite=True, checksum=False)
    x += 1



