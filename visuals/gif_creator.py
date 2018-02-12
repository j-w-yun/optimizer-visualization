import imageio

images = []
filenames = [str(i) + ".png" for i in range(103)]

for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('movie3.gif', images)