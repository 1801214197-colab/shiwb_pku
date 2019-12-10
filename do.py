import os
str1=('python generate_h5.py')
str2=('python model_cgru_senswords.py')
p= os.system(str1)
q= os.system(str2)

print('generate_h5 > ', p)
print('model > ', q)
