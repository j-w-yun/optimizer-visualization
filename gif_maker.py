import imageio

images = []
filenames = ['figures/' + str(i) + '.png' for i in range(65)]

imageio.plugins.freeimage.download()

for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('figures/movie.gif', images, format='GIF-FI', duration=0.001)
