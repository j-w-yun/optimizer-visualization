import imageio

images = []
filenames = ['figures/' + str(i) + '.png' for i in range(120)]

for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('figures/movie.gif', images)