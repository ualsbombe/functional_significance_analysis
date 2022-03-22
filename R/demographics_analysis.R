rm(list=ls())
path <- '/home/lau/projects/functional_cerebellum/scratch/demographics.csv'

demographics <- read.csv(path)

MEAN <- mean(demographics$Age)
RANGE <- range(demographics$Age)
SD = sd(demographics$Age)
n.males <- sum(demographics$Sex == 'M')
n.females <- sum(demographics$Sex == 'F')

print(c(MEAN, SD, RANGE, n.males, n.females))